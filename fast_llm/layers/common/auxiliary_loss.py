import torch


class AuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, aux_loss: torch.Tensor, grad: float) -> torch.Tensor:  # noqa
        ctx.grad = torch.full_like(aux_loss, grad)
        return scores

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # noqa
        return grad_output, ctx.grad, None


@torch.compile
def calculate_z_loss(logits: torch.Tensor, logits_scale_factor: float = 1.0) -> torch.Tensor:
    if logits_scale_factor != 1.0:
        logits *= logits_scale_factor
    return torch.mean(torch.logsumexp(logits, dim=-1) ** 2)


def z_loss(
    logits: torch.Tensor,
    grad_scale: float | None = None,
    logits_scale_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compute z-loss and its gradient.

    Z-loss = mean(logsumexp(logits, dim=-1) ** 2)

    Returns:
        loss: The z-loss value (unscaled)
        grad: The gradient w.r.t. logits (scaled by grad_scale), or None if grad_scale is None
    """
    if logits_scale_factor != 1.0:
        scaled_logits = logits * logits_scale_factor
    else:
        scaled_logits = logits

    # Forward: z_loss = mean(logsumexp^2)
    lse = torch.logsumexp(scaled_logits, dim=-1)  # (N,)
    loss = torch.mean(lse**2)

    # Backward: grad = (2/N) * lse * softmax(scaled_logits)
    grad = None
    if grad_scale is not None:
        N = scaled_logits.shape[0]
        softmax_logits = torch.softmax(scaled_logits, dim=-1)
        grad = (2.0 / N) * lse.unsqueeze(-1) * softmax_logits * grad_scale
        if logits_scale_factor != 1.0:
            grad = grad * logits_scale_factor  # Chain rule for logits_scale_factor

    return loss, grad
