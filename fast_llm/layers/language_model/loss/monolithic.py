import dataclasses
import typing

import torch

from fast_llm.core.distributed import ProcessGroup
from fast_llm.engine.base_model.config import LossDef
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.functional.config import TritonConfig
from fast_llm.functional.entropy_loss import softmax_base
from fast_llm.layers.language_model.loss.config import MonolithicLossConfig
from fast_llm.layers.language_model.loss.loss import CombinableLoss, LanguageModelLoss
from fast_llm.utils import safe_merge_dicts


@dataclasses.dataclass
class _TritonContext:
    """Scratch threaded through the triton monolithic path. Each child packs its kernel-slot inputs (and the
    shared labels / divisor) in `triton_add_inputs`, and reads its outputs back in `triton_finish`, so the
    driver iterates children with no per-kind branching. The single `@triton.jit` kernel keeps its fixed
    per-slot signature — that boundary is the one place the loss kinds are still named."""

    # Shared inputs (first non-`None` child wins) and per-slot kernel inputs.
    labels: torch.Tensor | None = None
    divisor: float | None = None
    ce: tuple | None = None
    z: tuple | None = None
    grpo: tuple | None = None
    gspo_coeff: torch.Tensor | None = None
    # Per-slot kernel outputs; GSPO's loss / new-log-probs instead come from its eager seam.
    ce_loss: torch.Tensor | None = None
    z_loss: torch.Tensor | None = None
    grpo_loss: torch.Tensor | None = None
    grpo_new_logprobs: torch.Tensor | None = None
    gspo_loss: torch.Tensor | None = None
    gspo_new_logprobs: torch.Tensor | None = None
    # Shared metrics precursors from the kernel's softmax + `Σ exp·logits_norm` (set only when metrics are on).
    new_log_probs: torch.Tensor | None = None
    entropy_per_token: torch.Tensor | None = None


@torch.compile
def _monolithic_core(
    children: tuple[LanguageModelLoss, ...],
    logits: torch.Tensor,  # (*batch, vocab)
    group: ProcessGroup | None,
    logits_scale_factor: float,
    grad_logits: torch.Tensor | None,
    arguments: tuple[tuple, ...],
) -> tuple[list[tuple[torch.Tensor, typing.Any]], torch.Tensor | None]:
    """
    One shared softmax over the logits, then each child loss's `fused_core` consuming it. The child
    list is fixed per config, so the loop unrolls inside this single `@torch.compile` boundary and each
    `fused_core` dispatches (and inlines) to its loss type's math — every enabled loss is fused over
    one softmax. Gradient contributions accumulate in fp32 and cast to `logits.dtype` once at the end.
    """
    logits_norm, exp_logits, sum_exp_logits, logits_max = softmax_base(logits, logits_scale_factor, group)
    grad = None
    results = []
    for child, child_arguments in zip(children, arguments, strict=True):
        loss, child_grad, extra = child.fused_core(
            logits_norm, exp_logits, sum_exp_logits, logits_max, group, logits_scale_factor, child_arguments
        )
        results.append((loss, extra))
        if child_grad is not None:
            grad = child_grad if grad is None else grad + child_grad
    return results, CombinableLoss._accumulate_grad(grad, logits.dtype, grad_logits)


class MonolithicLoss[ConfigType: MonolithicLossConfig](LanguageModelLoss[ConfigType]):
    """
    A composite loss that runs the vocabulary softmax once and shares it across its combinable child losses,
    emitting each child's scalar / metrics and the combined logits gradient in a single `@torch.compile`
    boundary. It is an ordinary head loss: the head loops over it like any other and threads the same gradient
    buffer, so non-combinable losses remain plain siblings in the head's loss list.
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
                    logits_scale_factor=self._logits_scale_factor,
                    weight=self._weight,
                    register_loss=children_register,
                )
                for child_name, child_config in config.losses.items()
            ]
        )
        # The shared softmax serves one effective scale; the config validates the children agree on it.
        self._softmax_scale_factor = self._children[0]._logits_scale_factor
        # The triton kernel fuses at most one loss per kind; anything else falls back to the compiled path.
        triton_kinds = [child._config.triton_kind for child in self._children]
        self._triton_valid = None not in triton_kinds and len(set(triton_kinds)) == len(triton_kinds)

    def forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor | None, torch.Tensor | None]":
        if self._triton_valid and TritonConfig.enabled(logits.device, self._config.use_triton):
            return self._triton_forward_backward(logits, kwargs, losses, split_index, grad_logits)
        register = losses is not None
        arguments = tuple(child.get_inputs(kwargs, split_index, register) for child in self._children)
        group = self._parallel_dim.group if self._vocab_parallel else None
        results, grad_logits = _monolithic_core(
            tuple(self._children), logits, group, self._softmax_scale_factor, grad_logits, arguments
        )

        total_loss = None
        for child, (loss, extra) in zip(self._children, results, strict=True):
            # A child whose gradient can't be produced inside the compiled boundary (an eager seam) had
            # `fused_core` return `(None, None, forward_state)`; `finish` completes its loss/gradient here.
            loss, extra, grad_logits = child.finish(loss, extra, kwargs, split_index, grad_logits, logits.dtype)
            if child._do_register_loss:
                child._register_loss(child.name, loss, losses)
            child.register_combinable_extras(extra, kwargs, losses)
            weighted = loss if child.weight == 1 else loss * child.weight
            total_loss = weighted if total_loss is None else total_loss + weighted
        return total_loss, grad_logits

    def _triton_forward_backward(
        self,
        logits: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict | None,
        split_index: int,
        grad_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        from fast_llm.functional.triton.monolithic_loss import (
            _monolithic_forward_reduce,
            triton_monolithic_loss_forward_backward,
        )

        register = losses is not None
        group = self._parallel_dim.group if self._vocab_parallel else None

        # Each child packs its own kernel slot (and the shared labels / divisor) — no per-kind branching.
        context = _TritonContext()
        for child in self._children:
            child.triton_add_inputs(context, kwargs, split_index, register)
        compute_metrics = any(child.triton_metrics_enabled(register) for child in self._children)

        # A child with an eager segment seam (GSPO) can't produce its gradient in one kernel launch: it needs
        # the shared softmax first, so run the reduced forward once, then let each such child add its
        # per-token backward coefficient the kernel superposes with the in-kernel losses.
        softmax = None
        if any(child.triton_needs_forward for child in self._children):
            softmax = _monolithic_forward_reduce(logits, context.labels, group, self._softmax_scale_factor)
            for child in self._children:
                child.triton_seam(context, softmax, kwargs, split_index)

        (
            context.ce_loss,
            context.z_loss,
            context.grpo_loss,
            context.grpo_new_logprobs,
            grad_logits,
            softmax,
            weighted_logits_sum,
        ) = triton_monolithic_loss_forward_backward(
            logits,
            None if context.labels is None else context.labels.contiguous(),
            grad_logits,
            self._softmax_scale_factor,
            group,
            1.0 if context.divisor is None else context.divisor,
            ce=context.ce,
            z=context.z,
            grpo=context.grpo,
            gspo_coeff=context.gspo_coeff,
            softmax=softmax,
            compute_metrics=compute_metrics,
        )

        # Metrics reuse the kernel's shared softmax: per-token new log-probs and the entropy from the kernel's
        # `Σ exp·logits_norm`, so no second softmax pass. Derived once here and shared by each metric loss.
        if compute_metrics:
            max_logits, sum_exp_logits, predicted_logits = softmax
            log_sum_exp_logits = sum_exp_logits.log()
            context.new_log_probs = predicted_logits - max_logits - log_sum_exp_logits
            context.entropy_per_token = log_sum_exp_logits - weighted_logits_sum / sum_exp_logits

        total_loss = None
        for child in self._children:
            loss, extra = child.triton_finish(context, kwargs, split_index, register)
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
