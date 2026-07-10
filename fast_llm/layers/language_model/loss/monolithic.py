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
        # `(cross_entropy, z, grpo, gspo)` children when every child has a triton fused kernel, else `None`.
        self._triton_children = self._classify_triton_children()

    def _classify_triton_children(self) -> tuple | None:
        from fast_llm.layers.language_model.loss.entropy_loss import LanguageModelLabelEntropyLoss
        from fast_llm.layers.language_model.loss.policy_gradient import LanguageModelGRPOLoss, LanguageModelGSPOLoss
        from fast_llm.layers.language_model.loss.z_loss import LanguageModelZLoss

        ce = z = grpo = gspo = None
        for child in self._children:
            if isinstance(child, LanguageModelGSPOLoss):
                if gspo is not None:
                    return None
                gspo = child
            elif isinstance(child, LanguageModelGRPOLoss):
                if grpo is not None:
                    return None
                grpo = child
            elif isinstance(child, LanguageModelZLoss):
                if z is not None:
                    return None
                z = child
            elif isinstance(child, LanguageModelLabelEntropyLoss):
                if ce is not None:
                    return None
                ce = child
            else:
                return None
        return ce, z, grpo, gspo

    def forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor | None, torch.Tensor | None]":
        if self._triton_children is not None and TritonConfig.enabled(logits.device, self._config.use_triton):
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
        from fast_llm.layers.language_model.loss.config import PolicyMetricsLevel

        register = losses is not None
        ce, z, grpo, gspo = self._triton_children
        group = self._parallel_dim.group if self._vocab_parallel else None

        labels = divisor = ce_arg = z_arg = grpo_arg = None
        if ce is not None:
            labels, ce_grad_output, divisor = ce.get_inputs(kwargs, split_index, register)
            ce_arg = (ce_grad_output,)
        if z is not None:
            z_loss_mask, z_grad_output, z_divisor = z.get_inputs(kwargs, split_index, register)
            z_arg = (z_loss_mask, z_grad_output)
            divisor = z_divisor if divisor is None else divisor
        if grpo is not None:
            grpo_target, advantages, old_log_probabilities, grpo_grad_output, grpo_divisor, *grpo_tail = (
                grpo.get_inputs(kwargs, split_index, register)
            )
            epsilon_low, epsilon_high, num_labels_in_seq = grpo_tail[:3]
            labels = grpo_target if labels is None else labels
            grpo_arg = (
                advantages,
                old_log_probabilities,
                grpo_grad_output,
                epsilon_low,
                epsilon_high,
                num_labels_in_seq,
            )
            divisor = grpo_divisor if divisor is None else divisor

        grpo_metrics_on = grpo is not None and register and grpo._metrics_level != PolicyMetricsLevel.none
        gspo_metrics_on = gspo is not None and register and gspo._metrics_level != PolicyMetricsLevel.none
        compute_metrics = grpo_metrics_on or gspo_metrics_on

        # GSPO runs its forward and eager segment seam here, then hands its per-token backward coefficient
        # (and its already-reduced softmax) to the kernel, which superposes it with the in-kernel losses.
        gspo_coeff = softmax = gspo_loss = gspo_new_logprobs = None
        if gspo is not None:
            gspo_target, _ = gspo.get_inputs(kwargs, split_index, register)
            labels = gspo_target if labels is None else labels
            softmax = _monolithic_forward_reduce(logits, labels, group, self._softmax_scale_factor)
            gspo_loss, gspo_new_logprobs, gspo_coeff = gspo.compute_triton_seam(kwargs, split_index, *softmax)

        ce_loss, z_loss, grpo_loss, grpo_new_logprobs, grad_logits, softmax, weighted_logits_sum = (
            triton_monolithic_loss_forward_backward(
                logits,
                None if labels is None else labels.contiguous(),
                grad_logits,
                self._softmax_scale_factor,
                group,
                1.0 if divisor is None else divisor,
                ce=ce_arg,
                z=z_arg,
                grpo=grpo_arg,
                gspo_coeff=gspo_coeff,
                softmax=softmax,
                compute_metrics=compute_metrics,
            )
        )

        # Metrics reuse the kernel's shared softmax: per-token new log-probs and the entropy from the
        # kernel's `Σ exp·logits_norm`, so no second softmax pass.
        grpo_metrics = gspo_metrics = None
        if compute_metrics:
            max_logits, sum_exp_logits, predicted_logits = softmax
            log_sum_exp_logits = sum_exp_logits.log()
            new_log_probs = predicted_logits - max_logits - log_sum_exp_logits
            entropy_per_token = log_sum_exp_logits - weighted_logits_sum / sum_exp_logits
            if grpo_metrics_on:
                grpo_metrics = grpo.triton_metrics(new_log_probs, entropy_per_token, kwargs, split_index)
            if gspo_metrics_on:
                gspo_metrics = gspo.triton_metrics(new_log_probs, entropy_per_token, kwargs, split_index)

        results = {}
        if ce is not None:
            results[ce] = (ce_loss, None)
        if z is not None:
            results[z] = (z_loss, None)
        if grpo is not None:
            results[grpo] = (grpo_loss, (grpo_new_logprobs, grpo_metrics))
        if gspo is not None:
            results[gspo] = (gspo_loss, (gspo_new_logprobs, gspo_metrics))

        total_loss = None
        for child in self._children:
            loss, extra = results[child]
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
