import torch


class AuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, aux_loss: torch.Tensor, grad: float) -> torch.Tensor:  # noqa
        ctx.grad = torch.full_like(aux_loss, grad)
        return scores

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa
        return grad_output, ctx.grad, None


@torch.compile
def calculate_z_loss(logits: torch.Tensor, logits_scale_factor: float = 1.0) -> torch.Tensor:
    if logits_scale_factor != 1.0:
        logits *= logits_scale_factor
    return torch.mean(torch.square(torch.logsumexp(logits, dim=-1)))


def z_loss(
    logits: torch.Tensor,
    z_loss_factor: float,
    training: bool,
    grad_scale: float | None = None,
    losses: dict | None = None,
    loss_name: str | None = None,
    logits_scale_factor: float = 1.0,
) -> torch.Tensor:
    if losses is not None or (training and grad_scale is not None):
        loss = calculate_z_loss(logits, logit_scale_factor=logits_scale_factor)
        if losses is not None and loss_name is not None:
            losses[loss_name].append(loss.detach())
        if training and grad_scale is not None:
            logits = AuxiliaryLoss.apply(logits, loss, z_loss_factor * grad_scale)

    return logits
