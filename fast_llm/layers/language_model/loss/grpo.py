import functools
import typing

import torch

from fast_llm.engine.base_model.config import LossDef, ReductionType
from fast_llm.functional.config import TritonConfig
from fast_llm.functional.entropy_loss import fused_predicted_logits_from_labels, fused_softmax_base
from fast_llm.functional.utils import reduce_losses
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.language_model.loss.config import LanguageModelGRPOLossConfig, LanguageModelLossKwargs
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss


class LanguageModelGRPOLoss[ConfigType: LanguageModelGRPOLossConfig](LanguageModelLoss[ConfigType]):
    def _forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._config.policy_loss == "gspo":
            loss, grad, new_logprobs_mean = fused_gspo_loss_forward_backward(
                logits,
                self._get_labels(kwargs, split_index),
                self._prepare_target(kwargs[LanguageModelLossKwargs.advantages], split_index),
                self._prepare_target(kwargs[LanguageModelLossKwargs.old_log_probabilities], split_index),
                self._prepare_target(kwargs[LanguageModelKwargs.document_index], split_index),
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
                sdp_group=self._sdp_dim.group if self._sdp_active else None,
            )
        else:
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

        if new_logprobs_mean is not None:
            new_logprobs_mean = new_logprobs_mean / kwargs[LanguageModelKwargs.num_documents_in_batch]
        self._register_loss(
            self._logprob_metric_name, new_logprobs_mean, losses, reduce_op=torch.distributed.ReduceOp.SUM
        )

        if losses is not None and (self._config.compute_extra_metrics or self._config.compute_entropy_metric):
            self._register_pg_metrics(logits, kwargs, losses, split_index)

        return loss, grad

    def _register_pg_metrics(
        self,
        logits: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict,
        split_index: int,
    ) -> None:
        from fast_llm.layers.language_model.loss.pg_metrics import compute_policy_gradient_metrics

        metrics = compute_policy_gradient_metrics(
            logits,
            self._get_labels(kwargs, split_index),
            self._prepare_target(kwargs[LanguageModelLossKwargs.old_log_probabilities], split_index),
            self._prepare_target(kwargs[LanguageModelLossKwargs.advantages], split_index),
            self._prepare_target(kwargs[LanguageModelLossKwargs.label_counts], split_index),
            self._config.epsilon_low,
            self._config.epsilon_high,
            self._logits_scale_factor,
            vocab_parallel_group=self._parallel_dim.group if self._vocab_parallel else None,
            compute_entropy=self._config.compute_entropy_metric,
            entropy_chunk_size=self._config.entropy_chunk_size,
        )

        num_docs = kwargs[LanguageModelKwargs.num_documents_in_batch]
        name = self._name

        # Per-token mean metrics: divide by num_docs to match new_logprobs_mean normalization.
        for attr in (
            "old_logprobs",
            "ratio_new_old",
            "kl_new_old",
            "clamp_log_ratio_new_old_indicator",
            "advantage",
        ):
            self._register_loss(f"{name}_{attr}", getattr(metrics, attr) / num_docs, losses)

        # Raw sum metrics (no per-doc normalization).
        for attr in (
            "ratio_new_old_sum",
            "ratio_new_old_squared_sum",
            "num_tokens",
        ):
            self._register_loss(f"{name}_{attr}", getattr(metrics, attr), losses)

        # MAX/MIN metrics: pass correct reduce_op for sequence-parallel mode.
        self._register_loss(
            f"{name}_max_advantage",
            metrics.max_advantage,
            losses,
            reduce_op=torch.distributed.ReduceOp.MAX,
        )
        self._register_loss(
            f"{name}_min_advantage",
            metrics.min_advantage,
            losses,
            reduce_op=torch.distributed.ReduceOp.MIN,
        )

        if metrics.entropy is not None:
            self._register_loss(f"{name}_entropy", metrics.entropy / num_docs, losses)

    def get_loss_definitions(self) -> list[LossDef]:
        defs = super().get_loss_definitions() + [LossDef(self._logprob_metric_name)]
        if self._config.compute_extra_metrics or self._config.compute_entropy_metric:
            name = self._name
            defs += [
                LossDef(f"{name}_old_logprobs"),
                LossDef(f"{name}_ratio_new_old"),
                LossDef(f"{name}_ratio_new_old_sum"),
                LossDef(f"{name}_ratio_new_old_squared_sum"),
                LossDef(f"{name}_kl_new_old"),
                LossDef(f"{name}_clamp_log_ratio_new_old_indicator"),
                LossDef(f"{name}_advantage"),
                LossDef(f"{name}_max_advantage", reduction=ReductionType.maximum),
                LossDef(f"{name}_min_advantage", reduction=ReductionType.minimum),
                LossDef(f"{name}_num_tokens"),
            ]
            if self._config.compute_entropy_metric:
                defs.append(LossDef(f"{name}_entropy"))
        return defs

    def get_preprocessing_config(
        self,
    ) -> dict[str, typing.Any]:
        config = {"use_grpo_data": True, "return_label_counts": True, "return_document_count": True}
        if self._config.policy_loss == "gspo":
            config["return_document_index"] = True
        return config

    @functools.cached_property
    def _logprob_metric_name(self) -> str:
        return f"{self._name}_new_logprobs"


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
    sdp_group: torch.distributed.ProcessGroup | None = None,  # SDP group for cross-rank segment aggregation
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """GSPO loss: sequence-level geometric-mean IS ratio clipping.

    Each segment s gets ratio R_s = exp(mean_t(log(p_new_t/p_old_t))), clipped as a unit.
    Loss = -sum_s tok_count_s * min(R_s*A_s, clip(R_s)*A_s) / divisor.
    Gradient: tok_count_s cancels, so each token in segment s gets the same gradient factor R_s.

    SDP correctness: scatter_add sums are all-reduced across sdp_group before computing R_s and A_s,
    ensuring correct segment-level ratios when tokens are split across ranks.
    """
    if divisor is None:
        divisor = float(logits.shape[0]) if logits.shape[0] > 0 else 1.0
    grad_output_scaled = None if grad_output is None else grad_output / divisor * logits_scale_factor

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
    tok_sum = log_ratio.new_zeros(n_segs)  # token count per segment

    if loss_mask.any() and n_segs > 0:
        masked_doc_ids = document_index[loss_mask].long()
        lrn_sum.index_add_(0, masked_doc_ids, log_ratio[loss_mask])
        adv_sum.index_add_(0, masked_doc_ids, advantages[loss_mask].float())
        tok_sum.index_add_(0, masked_doc_ids, torch.ones(masked_doc_ids.numel(), device=logits.device))

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
