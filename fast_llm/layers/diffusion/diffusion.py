import math
import torch
import torch.nn.functional as F

def get_noise_schedule(schedule_type: str, num_timesteps: int) -> torch.Tensor:
    """
    Get noise schedule for diffusion process.
    Args:
        schedule_type: Type of noise schedule ('linear', 'cosine', or 'sqrt')
        num_timesteps: Number of diffusion timesteps
    Returns:
        Tensor of shape [num_timesteps] containing noise schedule values
    """
    if schedule_type == "linear":
        return torch.linspace(0.0001, 0.02, num_timesteps)
    elif schedule_type == "cosine":
        steps = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = torch.clamp(1 - alpha_bar[1:] / alpha_bar[:-1], 0.0001, 0.9999)
        return betas
    elif schedule_type == "sqrt":
        return torch.linspace(0.0001 ** 0.5, 0.02 ** 0.5, num_timesteps) ** 2
    else:
        raise ValueError(f"Unknown noise schedule type: {schedule_type}")

def get_alphas(betas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha values from beta schedule.
    Returns:
        Tuple of (alphas, alpha_bars, alpha_bars_prev)
    """
    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
    return alphas, alpha_bars, alpha_bars_prev

def apply_forward_diffusion(
    x: torch.Tensor,
    t: torch.Tensor,
    alpha_bars: torch.Tensor,
    vocab_size: int,
    mask_token_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply forward diffusion process to input tokens.
    Args:
        x: Input token ids [batch_size, seq_len]
        t: Timesteps [batch_size]
        alpha_bars: Cumulative products of (1 - beta) for noise schedule
        vocab_size: Size of vocabulary
        mask_token_id: Optional mask token ID for MLM-style training
    Returns:
        Tuple of (noisy_tokens, noise_mask) where noise_mask indicates which tokens were corrupted
    """
    x_one_hot = F.one_hot(x, num_classes=vocab_size).float()
    noise_scale = (1. - alpha_bars[t]).view(-1, 1, 1)
    noise = torch.randn_like(x_one_hot)
    noisy_logits = x_one_hot + noise_scale.sqrt() * noise

    if mask_token_id is not None:
        # MLM-style: replace noisy tokens with mask token
        noise_mask = torch.bernoulli(noise_scale.squeeze(-1)).bool()
        noisy_tokens = torch.where(noise_mask, mask_token_id, x)
    else:
        # Continuous diffusion: sample from categorical
        noisy_tokens = torch.multinomial(F.softmax(noisy_logits, dim=-1), num_samples=1).squeeze(-1)
        noise_mask = torch.bernoulli(noise_scale.squeeze(-1)).bool()
    
    return noisy_tokens, noise_mask

def get_loss_weights(
    loss_type: str,
    timesteps: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """
    Get loss weights for different timesteps.
    Args:
        loss_type: Type of loss ('mlm' or 'l2')
        timesteps: Timestep indices
        alpha_bars: Cumulative products of (1 - beta)
    Returns:
        Weights tensor of shape [batch_size]
    """
    if loss_type == "mlm":
        return torch.ones_like(timesteps, dtype=torch.float32)
    elif loss_type == "l2":
        snr = alpha_bars[timesteps] / (1 - alpha_bars[timesteps])
        return snr / snr.mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}") 