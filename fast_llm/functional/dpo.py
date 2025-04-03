import torch
import torch.nn.functional as F
from typing import Tuple


def compute_logps_for_spans(
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        chosen_span: torch.Tensor, 
        rejected_span: torch.Tensor
    ):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # gather log probabilities corresponding to the target tokens
    # selected_log_probs = log_probs[torch.arange(logits.shape[0] - 1), targets]
    selected_log_probs = log_probs[:-1].gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # apply chosen mask
    chosen_mask = torch.zeros_like(selected_log_probs, dtype=torch.bool)
    chosen_mask[chosen_span[:, 0]: chosen_span[:, 1] + 1] = 1
    chosen_logp = (selected_log_probs * chosen_mask).sum()

    # apply rejected mask
    rejected_mask = torch.zeros_like(selected_log_probs, dtype=torch.bool)
    rejected_mask[rejected_span[:, 0]: rejected_span[:, 1] + 1] = 1
    rejected_logp = (selected_log_probs * rejected_mask).sum()

    # chosen_logp = selected_log_probs[chosen_span[:, 0]: chosen_span[:, 1] + 1].sum()
    # rejected_logp = selected_log_probs[rejected_span[:, 0]: rejected_span[:, 1] + 1].sum()
    
    return chosen_logp, rejected_logp

def compute_simplified_dpo_loss(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    chosen_span: torch.Tensor, 
    rejected_span: torch.Tensor,
    beta: float,
    grad_output: float | None
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        logits_ = logits.float().detach().requires_grad_()

        policy_chosen_logps, policy_rejected_logps = compute_logps_for_spans(logits_, targets, chosen_span, rejected_span)

        pi_logratios = policy_chosen_logps - policy_rejected_logps

        losses = -F.logsigmoid(beta * pi_logratios)
        if grad_output is None:
            loss = None
        else:
            loss = losses.mean()
            loss.backward(torch.full_like(loss, grad_output))
            loss.detach()
    return loss.detach(), logits_.grad.detach().to(logits.dtype)
