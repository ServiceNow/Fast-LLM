import torch


def _get_target_log_probabilities(logits: torch.Tensor, targets: torch.Tensor):
    # Gather log probabilities corresponding to the target tokens
    return torch.nn.functional.log_softmax(logits, dim=-1).gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def _get_target_log_probability_for_spans(log_probabilities: torch.Tensor, spans: list[list[tuple[int, int]]]):
    return sum(
        log_probabilities[sample_index, begin:end].sum()
        for sample_index, sample_spans in enumerate(spans)
        for begin, end in sample_spans
    )


def compute_dpo_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reference_model_logits: torch.Tensor,
    chosen_spans: list[list[tuple[int, int]]],
    rejected_spans: list[list[tuple[int, int]]],
    beta: float,
    grad_output: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        logits_ = logits.float().detach().requires_grad_()
        reference_model_logits_ = reference_model_logits.float().detach()

        policy_log_probabilities = _get_target_log_probabilities(logits_, targets)
        policy_log_ratios = _get_target_log_probability_for_spans(
            policy_log_probabilities, chosen_spans
        ) - _get_target_log_probability_for_spans(policy_log_probabilities, rejected_spans)

        reference_log_probabilities = _get_target_log_probabilities(reference_model_logits_, targets)
        reference_log_ratios = _get_target_log_probability_for_spans(
            reference_log_probabilities, chosen_spans
        ) - _get_target_log_probability_for_spans(reference_log_probabilities, rejected_spans)

        # TODO: ====== Shouldn't the sigmoid be computed independently for each document?
        losses = -torch.nn.functional.logsigmoid(beta * (policy_log_ratios - reference_log_ratios))

        if grad_output is None:
            loss = None
        else:
            loss = losses.mean()
            loss.backward(torch.full_like(loss, grad_output))
            loss.detach()
    return loss.detach(), logits_.grad.detach().to(logits.dtype)
