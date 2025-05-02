import torch


def _compute_logprobs_for_preference_spans(
    logits: torch.Tensor, targets: torch.Tensor, chosen_spans: torch.Tensor, rejected_spans: torch.Tensor
):
    assert torch.all(targets < logits.size(-1)), "Target out of vocab range"

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # gather log probabilities corresponding to the target tokens
    selected_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # apply chosen mask
    chosen_logp = 0
    for idx, span in enumerate(chosen_spans):
        chosen_logp += selected_log_probs[idx][span[0].item() : span[1].item() + 1].sum()

    # apply rejected mask
    rejected_logp = 0
    for idx, span in enumerate(rejected_spans):
        rejected_logp += selected_log_probs[idx][span[0].item() : span[1].item() + 1].sum()

    return chosen_logp, rejected_logp, selected_log_probs


def _compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    diff_logratios = pi_logratios - ref_logratios

    losses = -torch.nn.functional.logsigmoid(beta * diff_logratios)
    return losses


def compute_dpo_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reference_model_logits: torch.Tensor,
    chosen_spans: torch.Tensor,
    rejected_spans: torch.Tensor,
    beta: float,
    grad_output: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        logits_ = logits.float().detach().requires_grad_()
        reference_model_logits_ = reference_model_logits.float().detach()

        policy_chosen_logps, policy_rejected_logps, _ = _compute_logprobs_for_preference_spans(
            logits_, targets, chosen_spans, rejected_spans
        )

        reference_chosen_logps, reference_rejected_logps, _ = _compute_logprobs_for_preference_spans(
            reference_model_logits_, targets, chosen_spans, rejected_spans
        )

        losses = _compute_dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            beta=beta,
        )

        if grad_output is None:
            loss = None
        else:
            loss = losses.mean()
            loss.backward(torch.full_like(loss, grad_output))
            loss.detach()
    return loss.detach(), logits_.grad.detach().to(logits.dtype)
