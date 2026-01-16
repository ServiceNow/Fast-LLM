import torch


class AuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, aux_loss: torch.Tensor, grad: float) -> torch.Tensor:  # noqa
        ctx.grad = torch.full_like(aux_loss, grad)
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # noqa
        return grad_output, ctx.grad, None


@torch.compile
def z_loss(
    logits: torch.Tensor, logits_scale_factor: float = 1.0, loss_mask: "torch.Tensor | None" = None
) -> torch.Tensor:
    out = torch.logsumexp(logits if logits_scale_factor == 1.0 else logits * logits_scale_factor, dim=-1) ** 2
    if loss_mask is not None:
        out = out * loss_mask
    return torch.mean(out)


def auxiliary_z_loss(
    logits: torch.Tensor,
    z_loss_factor: float,
    training: bool,
    grad_scale: float | None = None,
    losses: dict | None = None,
    loss_name: str | None = None,
    logits_scale_factor: float = 1.0,
    loss_mask: "torch.Tensor | None" = None,
) -> torch.Tensor:
    if losses is not None or (training and grad_scale is not None):
        loss = z_loss(logits, logits_scale_factor, loss_mask)
        if losses is not None and loss_name is not None:
            losses[loss_name].append(loss.detach())
        if training and grad_scale is not None:
            logits = AuxiliaryLoss.apply(logits, loss, z_loss_factor * grad_scale)

    return logits


def z_loss_forward_backward(
    logits: torch.Tensor,
    grad_output: float | None = None,
    loss_mask: "torch.Tensor | None" = None,
    logits_scale_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compute z-loss and its gradient.

    Z-loss = mean(logsumexp(logits, dim=-1) ** 2)

    Returns:
        loss: The z-loss value (unscaled)
        grad: The gradient w.r.t. logits (scaled by grad_scale), or None if grad_scale is None
    """

    with torch.set_grad_enabled(grad_output is not None):
        logits_ = logits.detach().requires_grad_(grad_output is not None)
        loss = z_loss(logits_, logits_scale_factor, loss_mask)
        if grad_output is None:
            grad = None
        else:
            loss.backward(torch.full_like(loss, grad_output))
            grad = logits_.grad.detach().to(logits.dtype)

    return loss, grad
