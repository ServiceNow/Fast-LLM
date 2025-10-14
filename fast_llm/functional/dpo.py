import torch


def _get_logratios(
    logits: torch.Tensor, targets: torch.Tensor, chosen_spans: torch.Tensor, rejected_spans: torch.Tensor
):
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

    return chosen_logp - rejected_logp


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

        pi_logratios = _get_logratios(logits_, targets, chosen_spans, rejected_spans)
        ref_logratios = _get_logratios(reference_model_logits_, targets, chosen_spans, rejected_spans)
        losses = -torch.nn.functional.logsigmoid(beta * (pi_logratios - ref_logratios))

        if grad_output is None:
            loss = None
        else:
            loss = losses.mean()
            loss.backward(torch.full_like(loss, grad_output))
            loss.detach()
    return loss.detach(), logits_.grad.detach().to(logits.dtype)
