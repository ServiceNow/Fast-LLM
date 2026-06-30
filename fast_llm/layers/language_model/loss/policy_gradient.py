import functools
import typing

import torch

from fast_llm.engine.base_model.config import LossDef, ReductionType
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.functional.config import TritonConfig
from fast_llm.functional.entropy_loss import (
    fused_predicted_logits_from_labels,
    fused_softmax_base,
    predicted_logits_from_labels,
    softmax_base,
)
from fast_llm.functional.utils import reduce_losses
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.language_model.loss.config import (
    GRPOMetricsLevel,
    LanguageModelGRPOLossConfig,
    LanguageModelGSPOLossConfig,
    LanguageModelLossKwargs,
    LanguageModelPolicyGradientLossConfig,
)
from fast_llm.layers.language_model.loss.grpo_metrics import GRPOMetrics, grpo_metrics_core
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss
from fast_llm.layers.language_model.loss.monolithic import (
    MonolithicLossOutput,
    MonolithicLossSpec,
    gspo_backward_core,
    gspo_segment_seam,
)
from fast_llm.utils import Assert


class LanguageModelPolicyGradientLoss[ConfigType: LanguageModelPolicyGradientLossConfig](
    LanguageModelLoss[ConfigType]
):
    """Shared scaffolding for policy-gradient losses (GRPO, GSPO)."""

    def _register_new_logprobs(
        self,
        new_logprobs_mean: torch.Tensor | None,
        kwargs: dict[str, typing.Any],
        losses: dict | None,
    ) -> None:
        if new_logprobs_mean is not None:
            new_logprobs_mean = new_logprobs_mean / kwargs[LanguageModelKwargs.num_documents_in_batch]
        self._register_loss(
            self._logprob_metric_name, new_logprobs_mean, losses, reduce_op=torch.distributed.ReduceOp.SUM
        )

    def get_loss_definitions(self) -> list[LossDef]:
        defs = super().get_loss_definitions()
        defs.append(LossDef(self._logprob_metric_name))
        return defs

    def get_preprocessing_config(self) -> dict[str, typing.Any]:
        return {"use_grpo_data": True, "return_label_counts": True, "return_document_count": True}

    @functools.cached_property
    def _logprob_metric_name(self) -> str:
        return f"{self._name}_new_logprobs"


class LanguageModelGRPOLoss[ConfigType: LanguageModelGRPOLossConfig](LanguageModelPolicyGradientLoss[ConfigType]):
    """GRPO: per-token IS-ratio clipping."""

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
        Assert.custom(
            lambda metrics, pipeline_parallel: metrics == GRPOMetricsLevel.none or pipeline_parallel == 1,
            config.metrics,
            distributed_config.pipeline_parallel,
        )

    def _forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if TritonConfig.enabled(logits.device, self._config.use_triton):
            from fast_llm.functional.triton.grpo_loss import triton_grpo_loss_forward_backward

            fn = triton_grpo_loss_forward_backward
        else:
            fn = fused_grpo_loss_forward_backward
        loss, grad, new_logprobs_mean = fn(
            logits,
            self._get_labels(kwargs, split_index),
            self._prepare_target(kwargs[LanguageModelLossKwargs.advantages], split_index),
            self._prepare_target(kwargs[LanguageModelLossKwargs.old_log_probabilities], split_index),
            grad_logits=grad_logits,
            grad_output=self._get_grad_output(kwargs),
            group=self._parallel_dim.group if self._vocab_parallel else None,
            epsilon_low=self._config.epsilon_low,
            epsilon_high=self._config.epsilon_high,
            logits_scale_factor=self._logits_scale_factor,
            num_labels_in_seq=(
                None
                if losses is None
                else self._prepare_target(kwargs[LanguageModelLossKwargs.label_counts], split_index)
            ),
            divisor=self._get_label_count(kwargs),
        )

        self._register_new_logprobs(new_logprobs_mean, kwargs, losses)

        # Skip the extra softmax pass when there is nothing to register.
        if losses is not None and self._config.metrics != GRPOMetricsLevel.none:
            self._register_extra_metrics(logits, kwargs, losses, split_index)

        return loss, grad

    def get_monolithic_spec(self, kwargs: dict[str, typing.Any], split_index: int = 0) -> MonolithicLossSpec | None:
        return MonolithicLossSpec(
            kind="grpo",
            name=self.name,
            weight=self._weight,
            logits_scale_factor=self._logits_scale_factor,
            grad_output=self._get_grad_output(kwargs),
            divisor=self._get_label_count(kwargs),
            target=self._get_labels(kwargs, split_index),
            advantages=self._prepare_target(kwargs[LanguageModelLossKwargs.advantages], split_index),
            old_log_probabilities=self._prepare_target(
                kwargs[LanguageModelLossKwargs.old_log_probabilities], split_index
            ),
            epsilon_low=self._config.epsilon_low,
            epsilon_high=self._config.epsilon_high,
            num_labels_in_seq=self._prepare_target(kwargs[LanguageModelLossKwargs.label_counts], split_index),
            compute_metrics=self._config.metrics != GRPOMetricsLevel.none,
            compute_entropy=self._config.metrics == GRPOMetricsLevel.with_entropy,
        )

    def register_monolithic_outputs(
        self, output: MonolithicLossOutput, kwargs: dict[str, typing.Any], losses: dict | None
    ) -> None:
        super().register_monolithic_outputs(output, kwargs, losses)
        self._register_new_logprobs(output.new_logprobs_mean, kwargs, losses)
        if output.metrics is not None and losses is not None:
            self._register_grpo_metrics(output.metrics, kwargs, losses)

    def _register_extra_metrics(
        self,
        logits: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict | None,
        split_index: int,
    ) -> None:
        metrics = compute_grpo_metrics(
            logits,
            self._get_labels(kwargs, split_index),
            self._prepare_target(kwargs[LanguageModelLossKwargs.advantages], split_index),
            self._prepare_target(kwargs[LanguageModelLossKwargs.old_log_probabilities], split_index),
            self._prepare_target(kwargs[LanguageModelLossKwargs.label_counts], split_index),
            self._config.epsilon_low,
            self._config.epsilon_high,
            self._logits_scale_factor,
            group=self._parallel_dim.group if self._vocab_parallel else None,
            compute_entropy=self._config.metrics == GRPOMetricsLevel.with_entropy,
        )
        self._register_grpo_metrics(metrics, kwargs, losses)

    def _register_grpo_metrics(self, metrics: GRPOMetrics, kwargs: dict[str, typing.Any], losses: dict | None) -> None:
        num_documents = kwargs[LanguageModelKwargs.num_documents_in_batch]

        for attr in (
            "old_logprobs",
            "ratio_new_old",
            "kl_new_old",
            "clipped_ratio_fraction",
            "advantage",
        ):
            self._register_loss(f"{self._name}_{attr}", getattr(metrics, attr) / num_documents, losses)

        for attr in (
            "ratio_new_old_sum",
            "ratio_new_old_squared_sum",
            "num_tokens",
        ):
            self._register_loss(f"{self._name}_{attr}", getattr(metrics, attr), losses)

        self._register_loss(
            f"{self._name}_max_advantage",
            metrics.max_advantage,
            losses,
            reduce_op=torch.distributed.ReduceOp.MAX,
        )
        self._register_loss(
            f"{self._name}_min_advantage",
            metrics.min_advantage,
            losses,
            reduce_op=torch.distributed.ReduceOp.MIN,
        )

        if metrics.entropy is not None:
            self._register_loss(f"{self._name}_entropy", metrics.entropy / num_documents, losses)

    def get_loss_definitions(self) -> list[LossDef]:
        defs = super().get_loss_definitions()
        if self._config.metrics != GRPOMetricsLevel.none:
            defs.extend(
                [
                    LossDef(f"{self._name}_old_logprobs"),
                    LossDef(f"{self._name}_ratio_new_old"),
                    LossDef(f"{self._name}_ratio_new_old_sum"),
                    LossDef(f"{self._name}_ratio_new_old_squared_sum"),
                    LossDef(f"{self._name}_kl_new_old"),
                    LossDef(f"{self._name}_clipped_ratio_fraction"),
                    LossDef(f"{self._name}_advantage"),
                    LossDef(f"{self._name}_max_advantage", reduction=ReductionType.maximum),
                    LossDef(f"{self._name}_min_advantage", reduction=ReductionType.minimum),
                    LossDef(f"{self._name}_num_tokens"),
                ]
            )
            if self._config.metrics == GRPOMetricsLevel.with_entropy:
                defs.append(LossDef(f"{self._name}_entropy"))
        return defs


class LanguageModelGSPOLoss[ConfigType: LanguageModelGSPOLossConfig](LanguageModelPolicyGradientLoss[ConfigType]):
    """GSPO: sequence-level geometric-mean IS-ratio clipping."""

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
        # `cross_entropy_splits` chunks the sequence across forward calls; per-segment
        # aggregation can't recombine across chunks since each call only sees a slice.
        Assert.eq(self._num_splits, 1)

    def _forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        document_index_zero_based = (
            self._prepare_target(
                kwargs[BlockKwargs.global_document_index_q], split_index, split_by_distance=False
            ).long()
            - 1
        )
        # `global_document_index_q` is 1-based per the data preprocessor convention; the kernel takes 0-based.
        # `num_documents_in_sequence` is the doc count for this DP rank's batch — identical across
        # SDP/SP ranks within a DP rank, so per-segment buffers are sized consistently for all-reduce.
        num_segments = kwargs[BlockKwargs.num_documents_in_sequence]

        if TritonConfig.enabled(logits.device, self._config.use_triton):
            from fast_llm.functional.triton.gspo_loss import triton_gspo_loss_forward_backward

            fn = triton_gspo_loss_forward_backward
        else:
            fn = fused_gspo_loss_forward_backward
        loss, grad, new_logprobs_mean = fn(
            logits,
            self._get_labels(kwargs, split_index),
            self._prepare_target(kwargs[LanguageModelLossKwargs.advantages], split_index),
            self._prepare_target(kwargs[LanguageModelLossKwargs.old_log_probabilities], split_index),
            document_index_zero_based,
            num_segments,
            grad_logits=grad_logits,
            grad_output=self._get_grad_output(kwargs),
            group=self._parallel_dim.group if self._vocab_parallel else None,
            sdp_group=self._sequence_data_dim.group if self._sequence_data_active else None,
            # Sequence-parallel shards the sequence across the TP group, so per-segment
            # buffers are partial on each rank and must be reduced over TP as well.
            sp_group=self._parallel_dim.group if self._sequence_parallel else None,
            epsilon_low=self._config.epsilon_low,
            epsilon_high=self._config.epsilon_high,
            logits_scale_factor=self._logits_scale_factor,
            num_labels_in_seq=self._prepare_target(kwargs[LanguageModelLossKwargs.label_counts], split_index),
            divisor=kwargs[LanguageModelKwargs.num_documents_in_batch],
        )

        self._register_new_logprobs(new_logprobs_mean, kwargs, losses)
        return loss, grad

    def get_preprocessing_config(self) -> dict[str, typing.Any]:
        return super().get_preprocessing_config() | {"return_document_index": True}

    def get_monolithic_spec(self, kwargs: dict[str, typing.Any], split_index: int = 0) -> MonolithicLossSpec | None:
        return MonolithicLossSpec(
            kind="gspo",
            name=self.name,
            weight=self._weight,
            logits_scale_factor=self._logits_scale_factor,
            grad_output=self._get_grad_output(kwargs),
            divisor=kwargs[LanguageModelKwargs.num_documents_in_batch],
            target=self._get_labels(kwargs, split_index),
            advantages=self._prepare_target(kwargs[LanguageModelLossKwargs.advantages], split_index),
            old_log_probabilities=self._prepare_target(
                kwargs[LanguageModelLossKwargs.old_log_probabilities], split_index
            ),
            epsilon_low=self._config.epsilon_low,
            epsilon_high=self._config.epsilon_high,
            num_labels_in_seq=self._prepare_target(kwargs[LanguageModelLossKwargs.label_counts], split_index),
            document_index=self._prepare_target(
                kwargs[BlockKwargs.global_document_index_q], split_index, split_by_distance=False
            ).long()
            - 1,
            num_segments=kwargs[BlockKwargs.num_documents_in_sequence],
            sdp_group=self._sequence_data_dim.group if self._sequence_data_active else None,
            sp_group=self._parallel_dim.group if self._sequence_parallel else None,
        )

    def register_monolithic_outputs(
        self, output: MonolithicLossOutput, kwargs: dict[str, typing.Any], losses: dict | None
    ) -> None:
        super().register_monolithic_outputs(output, kwargs, losses)
        self._register_new_logprobs(output.new_logprobs_mean, kwargs, losses)


@torch.compile
def compute_grpo_metrics(
    logits: torch.Tensor,  # (*batch, vocab_local)
    target: torch.Tensor,  # (*batch,)
    advantages: torch.Tensor,  # (*batch,)
    old_log_probabilities: torch.Tensor,  # (*batch,)
    label_counts: torch.Tensor,  # (*batch,) — global per-sequence count broadcast per token
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    logits_scale_factor: float = 1.0,
    group: torch.distributed.ProcessGroup | None = None,
    compute_entropy: bool = False,
) -> GRPOMetrics:
    loss_mask = target >= 0
    logits_norm, exp_logits, sum_exp_logits, _ = fused_softmax_base(logits, logits_scale_factor, group)
    predicted_logits, _, _ = fused_predicted_logits_from_labels(logits_norm, target, loss_mask, group)
    new_log_probs = predicted_logits - sum_exp_logits.log()
    return grpo_metrics_core(
        logits_norm,
        exp_logits,
        sum_exp_logits,
        new_log_probs,
        advantages,
        old_log_probabilities,
        loss_mask,
        label_counts,
        epsilon_low,
        epsilon_high,
        group,
        compute_entropy,
    )


@torch.compile
def fused_grpo_loss_forward_backward(
    logits: torch.Tensor,  # (*batch, vocab)
    target: torch.Tensor,  # (*batch,)
    advantages: torch.Tensor,  # (*batch,)
    old_log_probabilities: torch.Tensor,  # (*batch,)
    grad_logits: torch.Tensor | None = None,
    grad_output: float | None = None,
    group: torch.distributed.ProcessGroup | None = None,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    logits_scale_factor: float = 1.0,
    num_labels_in_seq: (
        torch.Tensor | None
    ) = None,  # (*batch,) — response-span length broadcast per token, 0 for non-response
    divisor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if divisor is None:
        divisor = logits.shape[:-1].numel()
    grad_output = None if grad_output is None else grad_output / divisor * logits_scale_factor
    loss_mask = target >= 0

    logits_norm, exp_logits, sum_exp_logits, _ = fused_softmax_base(logits, logits_scale_factor, group)
    predicted_logits, target_masked, target_mask = fused_predicted_logits_from_labels(
        logits_norm, target, loss_mask, group
    )
    new_log_probs = predicted_logits - sum_exp_logits.log()
    probability_ratio = (new_log_probs - old_log_probabilities).exp()

    losses = -torch.min(
        probability_ratio * advantages,
        torch.clamp(probability_ratio, 1 - epsilon_low, 1 + epsilon_high) * advantages,
    )
    loss = reduce_losses(losses, divisor, loss_mask)

    # Sum of per-sequence mean log-probs, matching pipelinerl's new_logprobs metric:
    #   sum_sum(new_logprobs / num_labels_in_seq, masks_shifted, segments)
    # Dividing by num_labels_in_seq (span length broadcast per token) and summing over masked
    # tokens gives mean logprob per sequence; summing those across sequences matches the deepspeed
    # convention exactly (segments are redundant once num_labels_in_seq is correct).
    # Clamp to avoid 0/0=nan when num_labels_in_seq=0 (padded tokens or fully masked documents)
    # — those positions also have loss_mask=0 so they correctly contribute 0 to the sum.
    new_logprobs_mean = (
        None if num_labels_in_seq is None else (new_log_probs * loss_mask / num_labels_in_seq.clamp(min=1)).sum()
    )

    if grad_output is not None:
        # loss[a>=0] = -a * min(x, 1 + epsilon_high)  =>  grad[a>=0] = -a * (x <= 1 + epsilon_high)
        # loss[a<=0] = a * max(x, 1 - epsilon_low)  =>  grad[a<=0] = a * (x >= 1 - epsilon_low)
        probability_ratio_grad = (
            grad_output
            * (
                torch.clamp_min(advantages, 0) * (probability_ratio <= 1 + epsilon_high)
                + torch.clamp_max(advantages, 0) * (probability_ratio >= 1 - epsilon_low)
            )
            * loss_mask
        )

        # d(probability_ratio)/d(logits) = - probability_ratio * (predicted_probabilities - target_probabilities)
        # (Sign absorbed in probability_ratio_grad)
        predicted_probabilities = exp_logits / sum_exp_logits.unsqueeze_(-1)
        grad = (probability_ratio_grad * probability_ratio).unsqueeze(-1) * predicted_probabilities.scatter_add(
            -1,
            target_masked.unsqueeze(-1),
            -(loss_mask if target_mask is None else target_mask).unsqueeze(-1).to(torch.float32),
        )
        grad = grad.to(logits.dtype)

        if grad_logits is None:
            grad_logits = grad
        else:
            grad_logits.add_(grad)

    return loss, grad_logits, new_logprobs_mean


@torch.compile
def _gspo_forward_core(
    logits: torch.Tensor,  # (*batch, vocab)
    target: torch.Tensor,  # (*batch,)
    loss_mask: torch.Tensor,  # (*batch,), == target >= 0
    logits_scale_factor: float,
    group: torch.distributed.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """GSPO compiled forward: one softmax + predicted-logit lookup, producing per-token new log-probs.
    The softmax tensors are handed to the eager segment seam and the compiled backward."""
    logits_norm, exp_logits, sum_exp_logits, _ = softmax_base(logits, logits_scale_factor, group)
    predicted_logits, target_masked, target_mask = predicted_logits_from_labels(logits_norm, target, loss_mask, group)
    new_log_probs = predicted_logits - sum_exp_logits.log()
    return new_log_probs, exp_logits, sum_exp_logits, target_masked, target_mask


# Orchestrator only: the eager `index_add_` segment seam (with the Python-int `num_segments`) sits
# between the compiled forward and backward cores, so it stays out of every compiled boundary.
def fused_gspo_loss_forward_backward(
    logits: torch.Tensor,  # (*batch, vocab)
    target: torch.Tensor,  # (*batch,)
    advantages: torch.Tensor,  # (*batch,)
    old_log_probabilities: torch.Tensor,  # (*batch,)
    document_index_zero_based: torch.Tensor,  # (*batch,) int — segment ID per token, 0-based
    num_segments: int,  # buffer size, ≥ document_index.max() + 1
    divisor: float,
    num_labels_in_seq: torch.Tensor,  # (*batch,) — per-document labeled-token count broadcast per token
    grad_logits: torch.Tensor | None = None,
    grad_output: float | None = None,
    group: torch.distributed.ProcessGroup | None = None,  # TP vocab group
    sdp_group: torch.distributed.ProcessGroup | None = None,  # SDP group for cross-rank segment aggregation
    sp_group: torch.distributed.ProcessGroup | None = None,  # TP group when SP is sharding the sequence
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    logits_scale_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """GSPO loss: sequence-level geometric-mean IS-ratio clipping.

    Per-segment ratio R_s = exp(mean_t(log p_new_t / p_old_t)), clipped per segment.
    Per-segment loss = -min(R_s * A_s, clip(R_s) * A_s), multiplied by the segment
    token count, summed over segments, and divided by `divisor`.
    Computed as an equivalent per-token sum so the gradient chain mirrors GRPO.

    `num_labels_in_seq[t]` is the labeled-token count for the document containing token `t`
    (broadcast per token by the data preprocessor); it is the geometric-mean denominator.
    Using it directly — rather than aggregating token counts inside the kernel — is what
    makes the loss correct when a document spans SDP/SP ranks (numerator
    `log_ratio_sum` is all-reduced; denominator is constant and available locally).

    Constraint: each document must be visible to a single kernel call (modulo SDP/SP, where
    the kernel all-reduces). Splitting a document across separate kernel calls (e.g.
    `schedule.micro_batch_splits > 1`) produces partial per-fragment geometric means that
    cannot be combined linearly into the whole-document `exp(mean)`.

    With `sdp_group` and/or `sp_group`, segment sums are all-reduced over those groups so each rank
    computes the same global R_s/A_s. Token-level contributions remain per-rank, so summing the
    kernel loss across SDP/SP via SUM reduction matches the canonical single-rank result.
    """
    loss_mask = target >= 0
    new_log_probs, exp_logits, sum_exp_logits, target_masked, target_mask = _gspo_forward_core(
        logits, target, loss_mask, logits_scale_factor, group
    )
    loss, new_logprobs_mean, effective_grad = gspo_segment_seam(
        new_log_probs,
        loss_mask,
        advantages,
        old_log_probabilities,
        document_index_zero_based,
        num_segments,
        num_labels_in_seq,
        divisor,
        grad_output,
        sdp_group,
        sp_group,
        epsilon_low,
        epsilon_high,
        logits_scale_factor,
    )
    if effective_grad is not None:
        grad_logits = gspo_backward_core(
            exp_logits,
            sum_exp_logits,
            target_masked,
            target_mask,
            loss_mask,
            effective_grad,
            logits.dtype,
            grad_logits,
        )
    return loss, grad_logits, new_logprobs_mean
