import functools
import typing

import torch

from fast_llm.core.distributed import ReduceOp, all_reduce
from fast_llm.engine.base_model.config import LossDef, ReductionType
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.functional.config import TritonConfig
from fast_llm.functional.entropy_loss import fused_predicted_logits_from_labels, fused_softmax_base
from fast_llm.functional.utils import reduce_losses
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.language_model.loss.config import (
    GRPOMetricsLevel,
    LanguageModelGSPOLossConfig,
    LanguageModelLossKwargs,
    LanguageModelPolicyGradientLossConfig,
)
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss
from fast_llm.utils import Assert


class GRPOMetrics(typing.NamedTuple):
    old_logprobs: torch.Tensor
    ratio_new_old: torch.Tensor
    ratio_new_old_sum: torch.Tensor
    ratio_new_old_squared_sum: torch.Tensor
    kl_new_old: torch.Tensor
    clipped_ratio_fraction: torch.Tensor
    advantage: torch.Tensor
    max_advantage: torch.Tensor
    min_advantage: torch.Tensor
    num_tokens: torch.Tensor
    entropy: torch.Tensor | None


class LanguageModelPolicyGradientLoss[ConfigType: LanguageModelPolicyGradientLossConfig](
    LanguageModelLoss[ConfigType]
):
    """Policy-gradient loss for both GRPO (per-token IS-ratio clipping) and GSPO (sequence-level
    geometric-mean IS-ratio clipping). The kernel choice is dispatched at __init__ via
    `self._call_kernel`, following the same pattern as `Normalization._forward`.
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
        Assert.custom(
            lambda metrics, pipeline_parallel: metrics == GRPOMetricsLevel.none or pipeline_parallel == 1,
            config.metrics,
            distributed_config.pipeline_parallel,
        )
        self._call_kernel = (
            self._call_gspo_kernel if isinstance(config, LanguageModelGSPOLossConfig) else self._call_grpo_kernel
        )

    def _forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._config.normalize_by_documents:
            # Match DeepSpeed exactly. DS has TWO 1/batch_size factors with different sources:
            #   - Loss reported uses /batch_size (via tokens_weights = 1/batch_size, see
            #     pipelinerl/finetune/rl/__init__.py:246).
            #   - Gradient buffer uses an ADDITIONAL /(gas × world_size) factor that comes from
            #     `scale_wrt_gas=True` in engine.backward() (deepspeed/runtime/engine.py:1995-1996)
            #     and `tensor.div_(world_sz)` in reduce_scatter_coalesced
            #     (deepspeed/runtime/comm/coalesced_collectives.py:124).
            # For DS with samples_per_microbatch=1 (PipelineRL standard), gas × world_size = batch_size,
            # so the gradient buffer effectively has factor 1/batch_size² while the loss metric has 1/batch_size.
            # Fast-LLM cancels DS's /(gas × world_size) factor via `grad_output = data_parallel × grad_scale`
            # (runner.py:318) interacting with FSDP's RS-AVG over data_parallel ranks (fsdp.py:396).
            # So we need to apply the second 1/batch_size factor explicitly only to the gradient,
            # keeping the loss metric matched to DS:
            #   loss divisor      = num_documents (matches DS rl/loss)
            #   gradient divisor  = num_documents²  (matches DS grad_norm)
            # Both are independent of TP/PP/SDP/DP parallelism and microbatching schedule.
            num_documents = kwargs[LanguageModelKwargs.num_documents_in_batch]
            divisor = num_documents
            grad_divisor = num_documents * num_documents
        else:
            divisor = self._get_label_count(kwargs)
            grad_divisor = None  # use divisor (default behavior)
        loss, grad, new_logprobs_mean = self._call_kernel(
            logits=logits,
            target=self._get_labels(kwargs, split_index),
            advantages=self._prepare_target(kwargs[LanguageModelLossKwargs.advantages], split_index),
            old_log_probabilities=self._prepare_target(
                kwargs[LanguageModelLossKwargs.old_log_probabilities], split_index
            ),
            kwargs=kwargs,
            split_index=split_index,
            grad_logits=grad_logits,
            grad_output=self._get_grad_output(kwargs),
            divisor=divisor,
            grad_divisor=grad_divisor,
            num_labels_in_seq=(
                None
                if losses is None
                else self._prepare_target(kwargs[LanguageModelLossKwargs.label_counts], split_index)
            ),
        )

        if new_logprobs_mean is not None:
            new_logprobs_mean = new_logprobs_mean / kwargs[LanguageModelKwargs.num_documents_in_batch]
        self._register_loss(
            self._logprob_metric_name, new_logprobs_mean, losses, reduce_op=torch.distributed.ReduceOp.SUM
        )

        # Skip the extra softmax pass when there is nothing to register.
        if losses is not None and self._config.metrics != GRPOMetricsLevel.none:
            self._register_extra_metrics(logits, kwargs, losses, split_index)

        return loss, grad

    def _call_grpo_kernel(
        self,
        *,
        logits: torch.Tensor,
        target: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        kwargs: dict[str, typing.Any],
        split_index: int,
        grad_logits: torch.Tensor | None,
        grad_output: float | None,
        divisor: float,
        grad_divisor: float | None,
        num_labels_in_seq: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if TritonConfig.enabled(logits.device, self._config.use_triton):
            from fast_llm.functional.triton.grpo_loss import triton_grpo_loss_forward_backward

            fn = triton_grpo_loss_forward_backward
        else:
            fn = fused_grpo_loss_forward_backward
        return fn(
            logits,
            target,
            advantages,
            old_log_probabilities,
            grad_logits=grad_logits,
            grad_output=grad_output,
            group=self._parallel_dim.group if self._vocab_parallel else None,
            epsilon_low=self._config.epsilon_low,
            epsilon_high=self._config.epsilon_high,
            logits_scale_factor=self._effective_logits_scale,
            num_labels_in_seq=num_labels_in_seq,
            divisor=divisor,
            grad_divisor=grad_divisor,
        )

    def _call_gspo_kernel(
        self,
        *,
        logits: torch.Tensor,
        target: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        kwargs: dict[str, typing.Any],
        split_index: int,
        grad_logits: torch.Tensor | None,
        grad_output: float | None,
        divisor: float,
        grad_divisor: float | None,
        num_labels_in_seq: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        return fused_gspo_loss_forward_backward(
            logits,
            target,
            advantages,
            old_log_probabilities,
            self._prepare_target(kwargs[LanguageModelKwargs.document_index], split_index),
            grad_logits=grad_logits,
            grad_output=grad_output,
            group=self._parallel_dim.group if self._vocab_parallel else None,
            epsilon_low=self._config.epsilon_low,
            epsilon_high=self._config.epsilon_high,
            logits_scale_factor=self._effective_logits_scale,
            num_labels_in_seq=num_labels_in_seq,
            divisor=divisor,
            grad_divisor=grad_divisor,
            sdp_group=self._sdp_dim.group if self._sdp_active else None,
        )

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
            self._effective_logits_scale,
            group=self._parallel_dim.group if self._vocab_parallel else None,
            compute_entropy=self._config.metrics == GRPOMetricsLevel.with_entropy,
        )

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
        defs.append(LossDef(self._logprob_metric_name))
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

    def get_preprocessing_config(self) -> dict[str, typing.Any]:
        config = {"use_grpo_data": True, "return_label_counts": True, "return_document_count": True}
        if isinstance(self._config, LanguageModelGSPOLossConfig):
            config["return_document_index"] = True
        return config

    @functools.cached_property
    def _effective_logits_scale(self) -> float:
        return self._logits_scale_factor / self._config.temperature

    @functools.cached_property
    def _logprob_metric_name(self) -> str:
        return f"{self._name}_new_logprobs"


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
    mask = loss_mask.float()
    masked = mask / label_counts.float().clamp(min=1)

    logits_norm, exp_logits, sum_exp_logits, _ = fused_softmax_base(logits, logits_scale_factor, group)
    predicted_logits, _, _ = fused_predicted_logits_from_labels(logits_norm, target, loss_mask, group)
    new_log_probs = predicted_logits - sum_exp_logits.log()

    log_ratio = new_log_probs - old_log_probabilities
    ratio = log_ratio.exp()
    clipped = (ratio < 1.0 - epsilon_low) | (ratio > 1.0 + epsilon_high)
    kl = ratio - log_ratio - 1.0

    neg_inf = advantages.new_full((), float("-inf"))
    pos_inf = advantages.new_full((), float("inf"))

    entropy: torch.Tensor | None = None
    if compute_entropy:
        # exp_logits and logits_norm are local vocab slices — sum over the local slice, then all-reduce
        # across the tensor-parallel group to recover the global E_p[logit_norm] before dividing by the
        # already-global sum_exp_logits.
        weighted_logits_sum = (exp_logits * logits_norm).sum(-1)
        if group is not None:
            all_reduce(weighted_logits_sum, op=ReduceOp.SUM, group=group)
        entropy_per_token = sum_exp_logits.log() - weighted_logits_sum / sum_exp_logits
        entropy = (entropy_per_token * masked).sum()

    return GRPOMetrics(
        old_logprobs=(old_log_probabilities * masked).sum(),
        ratio_new_old=(ratio * masked).sum(),
        ratio_new_old_sum=(ratio * mask).sum(),
        ratio_new_old_squared_sum=(ratio * ratio * mask).sum(),
        kl_new_old=(kl * masked).sum(),
        clipped_ratio_fraction=(clipped.float() * masked).sum(),
        advantage=(advantages * masked).sum(),
        max_advantage=torch.where(loss_mask, advantages, neg_inf).max(),
        min_advantage=torch.where(loss_mask, advantages, pos_inf).min(),
        num_tokens=mask.sum(),
        entropy=entropy,
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
    grad_divisor: float | None = None,  # Optional separate divisor for the gradient (defaults to divisor)
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if divisor is None:
        divisor = logits.shape[:-1].numel()
    if grad_divisor is None:
        grad_divisor = divisor
    grad_output = None if grad_output is None else grad_output / grad_divisor * logits_scale_factor
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


def fused_gspo_loss_forward_backward(
    logits: torch.Tensor,  # (n_tokens, vocab_local)
    target: torch.Tensor,  # (n_tokens,)
    advantages: torch.Tensor,  # (n_tokens,)
    old_log_probabilities: torch.Tensor,  # (n_tokens,)
    document_index: torch.Tensor,  # (n_tokens,) int64 — segment ID per token
    grad_logits: torch.Tensor | None = None,
    grad_output: float | None = None,
    group: torch.distributed.ProcessGroup | None = None,  # TP vocab group
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    logits_scale_factor: float = 1.0,
    num_labels_in_seq: torch.Tensor | None = None,  # for new_logprobs_mean metric
    divisor: float | None = None,
    grad_divisor: float | None = None,  # Optional separate divisor for the gradient (defaults to divisor)
    sdp_group: torch.distributed.ProcessGroup | None = None,  # SDP group for cross-rank segment aggregation
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """GSPO loss: sequence-level geometric-mean IS ratio clipping.

    Each segment s gets ratio R_s = exp(mean_t(log(p_new_t/p_old_t))), clipped as a unit.
    Loss = -sum_s tok_count_s * min(R_s*A_s, clip(R_s)*A_s) / divisor.
    Gradient: tok_count_s cancels, so each token in segment s gets the same gradient factor R_s.

    SDP correctness: scatter_add sums are all-reduced across sdp_group before computing R_s and A_s,
    ensuring correct segment-level ratios when tokens are split across ranks.

    The optional `grad_divisor` allows the gradient to use a different divisor than the loss
    (e.g., to match DeepSpeed's metric where loss has /batch_size and gradient has /batch_size²).
    """
    if divisor is None:
        divisor = float(logits.shape[0]) if logits.shape[0] > 0 else 1.0
    if grad_divisor is None:
        grad_divisor = divisor
    grad_output_scaled = None if grad_output is None else grad_output / grad_divisor * logits_scale_factor

    loss_mask = target >= 0
    mask_float = loss_mask.float()

    # Step 1: Softmax + log probs (same as GRPO)
    logits_norm, exp_logits, sum_exp_logits, _ = fused_softmax_base(logits, logits_scale_factor, group)
    predicted_logits, target_masked, target_mask = fused_predicted_logits_from_labels(
        logits_norm, target, loss_mask, group
    )
    new_log_probs = predicted_logits - sum_exp_logits.log()
    log_ratio = (new_log_probs - old_log_probabilities).float()

    # new_logprobs_mean: local partial sum (aggregated across SDP via LossDef.reduce, same as GRPO)
    new_logprobs_mean = (
        None if num_labels_in_seq is None else (new_log_probs * mask_float / num_labels_in_seq.clamp(min=1)).sum()
    )

    # Step 2: Determine global n_segs (max doc index + 1, all-reduced across SDP)
    n_segs_local = int(document_index.max().item()) + 1 if document_index.numel() > 0 else 0
    if sdp_group is not None:
        n_segs_t = torch.tensor(n_segs_local, device=logits.device, dtype=torch.int64)
        torch.distributed.all_reduce(n_segs_t, op=torch.distributed.ReduceOp.MAX, group=sdp_group)
        n_segs = int(n_segs_t.item())
    else:
        n_segs = n_segs_local

    # Step 3: Per-segment scatter_add (local contributions only)
    lrn_sum = log_ratio.new_zeros(n_segs)  # sum of log-ratios per segment
    adv_sum = advantages.new_zeros(n_segs).float()  # sum of advantages per segment

    if loss_mask.any() and n_segs > 0:
        masked_doc_ids = document_index[loss_mask].long()
        lrn_sum.index_add_(0, masked_doc_ids, log_ratio[loss_mask])
        adv_sum.index_add_(0, masked_doc_ids, advantages[loss_mask].float())
        tok_sum = torch.bincount(masked_doc_ids, minlength=n_segs).to(log_ratio.dtype)
    else:
        tok_sum = log_ratio.new_zeros(n_segs)  # token count per segment

    # Step 4: SDP all-reduce so every rank has global per-segment sums
    if sdp_group is not None and n_segs > 0:
        torch.distributed.all_reduce(lrn_sum, op=torch.distributed.ReduceOp.SUM, group=sdp_group)
        torch.distributed.all_reduce(adv_sum, op=torch.distributed.ReduceOp.SUM, group=sdp_group)
        torch.distributed.all_reduce(tok_sum, op=torch.distributed.ReduceOp.SUM, group=sdp_group)

    # Step 5: Segment-level ratio R_s and advantage A_s
    valid = tok_sum > 0
    seg_denom = tok_sum.clamp(min=1e-6)
    R = (lrn_sum / seg_denom).exp()  # geometric mean IS ratio per segment
    A = (adv_sum / seg_denom).detach()  # mean advantage per segment (no gradient through A)

    # Step 6: GSPO loss — length-proportional weight tok_sum cancels with 1/N in gradient
    surr1 = R * A
    surr2 = R.clamp(1.0 - epsilon_low, 1.0 + epsilon_high) * A
    loss_per_seg = -torch.minimum(surr1, surr2) * tok_sum * valid.float()
    loss = loss_per_seg.sum() / divisor
    # SDP correction: after SDP allreduce of lrn/adv/tok, both SDP ranks compute the IDENTICAL
    # per-segment loss, so when LossDef.reduce sums across data_group (which includes SDP), the
    # metric is double-counted by sdp_size. Divide here so each SDP rank reports loss/sdp_size,
    # making the SUM-reduction match a non-SDP run. Gradient is unaffected (each SDP rank
    # contributes gradient from its own LOCAL tokens, no double-counting in the gradient buffer).
    if sdp_group is not None:
        loss = loss / torch.distributed.get_world_size(sdp_group)

    # Step 7: Gradient — broadcast segment-level factors to token level
    if grad_output_scaled is not None and n_segs > 0:
        # d(loss)/d(log_ratio_t) = -R_s * clip_factor_s / divisor (tok_sum cancels)
        # clip_factor_s = clamp_min(A_s,0)*(R_s <= 1+eps_h) + clamp_max(A_s,0)*(R_s >= 1-eps_l)
        clip_up = (R <= 1.0 + epsilon_high).float()
        clip_dn = (R >= 1.0 - epsilon_low).float()
        seg_grad = R * (A.clamp(min=0) * clip_up + A.clamp(max=0) * clip_dn) * valid.float()

        # Broadcast: each token gets its segment's gradient factor
        token_grad = seg_grad[document_index]  # (n_tokens,)

        # d(new_log_prob)/d(logits_k) = delta(k==target) - softmax_k  (same chain rule as GRPO)
        probability_ratio_grad = grad_output_scaled * token_grad * mask_float

        predicted_probabilities = exp_logits / sum_exp_logits.unsqueeze(-1)
        grad = probability_ratio_grad.unsqueeze(-1) * predicted_probabilities.scatter_add(
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
