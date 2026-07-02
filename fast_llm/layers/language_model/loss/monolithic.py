import typing

import torch

from fast_llm.core.distributed import ProcessGroup
from fast_llm.engine.base_model.config import LossDef
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.functional.entropy_loss import softmax_base
from fast_llm.layers.language_model.loss.config import MonolithicLossConfig
from fast_llm.layers.language_model.loss.loss import CombinableLoss, LanguageModelLoss
from fast_llm.utils import safe_merge_dicts


@torch.compile
def _monolithic_core(
    children: tuple["LanguageModelLoss", ...],
    logits: torch.Tensor,  # (*batch, vocab)
    group: ProcessGroup | None,
    logits_scale_factor: float,
    grad_logits: torch.Tensor | None,
    arguments: tuple[tuple, ...],
) -> tuple[list, torch.Tensor | None]:
    """
    One shared softmax over the logits, then each child loss's `fused_core` consuming it. The child
    list is fixed per config, so the loop unrolls inside this single `@torch.compile` boundary and each
    `fused_core` dispatches (and inlines) to its loss type's math — every enabled loss is fused over
    one softmax. Gradient contributions accumulate in fp32 and cast to `logits.dtype` once at the end.
    """
    logits_norm, exp_logits, sum_exp_logits, logits_max = softmax_base(logits, logits_scale_factor, group)
    grad = None
    results = []
    for child, child_arguments in zip(children, arguments):
        loss, child_grad, extra = child.fused_core(
            logits_norm, exp_logits, sum_exp_logits, logits_max, group, logits_scale_factor, child_arguments
        )
        results.append((loss, extra))
        if child_grad is not None:
            grad = child_grad if grad is None else grad + child_grad
    return results, CombinableLoss._accumulate_grad(grad, logits.dtype, grad_logits)


class MonolithicLoss[ConfigType: MonolithicLossConfig](LanguageModelLoss[ConfigType]):
    """
    A composite loss that runs the vocabulary softmax once and shares it across its combinable child
    losses (cross-entropy, z-loss, distillation, GRPO), emitting each child's scalar / metrics and the
    combined logits gradient in a single `@torch.compile` boundary. It is an ordinary head loss: the head
    loops over it like any other and threads the same gradient buffer, so non-combinable losses (e.g. DPO)
    are plain siblings in the head's loss list.
    """

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        name: str,
        prediction_distance: int = 1,
        prediction_heads: int = 1,
        vocab_parallel: bool = False,
        num_splits: int = 1,
        logits_scale_factor: float = 1.0,
        weight: float = 1.0,
        register_loss: bool = False,
    ):
        super().__init__(
            config,
            distributed_config,
            name=name,
            prediction_distance=prediction_distance,
            prediction_heads=prediction_heads,
            vocab_parallel=vocab_parallel,
            num_splits=num_splits,
            logits_scale_factor=logits_scale_factor,
            weight=weight,
            register_loss=register_loss,
        )
        # Register children as distinct losses, unless a single child equals the head total (logged anyway).
        children_register = register_loss or len(config.losses) > 1
        self._children: typing.Sequence[LanguageModelLoss] = torch.nn.ModuleList(
            [
                child_config.get_layer(
                    distributed_config,
                    name=child_name if prediction_distance == 1 else f"{child_name}_{prediction_distance}",
                    prediction_distance=prediction_distance,
                    prediction_heads=prediction_heads,
                    vocab_parallel=vocab_parallel,
                    num_splits=num_splits,
                    logits_scale_factor=logits_scale_factor,
                    weight=self._weight,
                    register_loss=children_register,
                )
                for child_name, child_config in config.losses.items()
            ]
        )
        # The shared softmax serves one effective scale; the config validates the children agree on it.
        self._softmax_scale_factor = self._children[0]._logits_scale_factor

    def forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor | None, torch.Tensor | None]":
        register = losses is not None
        arguments = tuple(child.get_inputs(kwargs, split_index, register) for child in self._children)
        group = self._parallel_dim.group if self._vocab_parallel else None
        results, grad_logits = _monolithic_core(
            tuple(self._children), logits, group, self._softmax_scale_factor, grad_logits, arguments
        )

        total_loss = None
        for child, (loss, extra) in zip(self._children, results, strict=True):
            if child._do_register_loss:
                child._register_loss(child.name, loss, losses)
            child.register_combinable_extras(extra, kwargs, losses)
            weighted = loss if child.weight == 1 else loss * child.weight
            total_loss = weighted if total_loss is None else total_loss + weighted
        return total_loss, grad_logits

    def get_preprocessing_config(self) -> dict[str, typing.Any]:
        return safe_merge_dicts(*(child.get_preprocessing_config() for child in self._children))

    def get_loss_definitions(self) -> list[LossDef]:
        return [loss_def for child in self._children for loss_def in child.get_loss_definitions()]
