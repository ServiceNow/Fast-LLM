"""GRPO (Group Relative Policy Optimization) loss: clipped policy ratio with
fused softmax + gather + clipped advantage in a single kernel."""

import dataclasses

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.grpo_loss import triton_grpo_loss_forward_backward
from tools.benchmark.triton_kernels.runner import DtypedCase, Inputs, Variant
from tools.benchmark.triton_kernels.utils import dtype_short, standard_pytorch_variants

_SHAPES = [
    (4096, 32768),
    (4096, 65536),
    (4096, 131072),
]
_EPSILON_LOW = 0.2
_EPSILON_HIGH = 0.2


@dataclasses.dataclass
class GrpoLossCase(DtypedCase):
    tokens: int
    vocab: int
    dtype: torch.dtype

    @property
    def name(self) -> str:
        return f"({self.tokens}, {self.vocab}) {dtype_short(self.dtype)}"

    @property
    def expected_bytes(self) -> int:
        # 3× logits traffic (read fwd, read+write bwd) + per-token scalars:
        # labels (int64 = 8B), advantages (fp32 = 4B), old_log_probs (fp32 = 4B).
        return 3 * self.tokens * self.vocab * self.dtype.itemsize + self.tokens * 16

    @property
    def expected_flops(self) -> int:
        # softmax (fwd) + grad (bwd) ≈ 14 FLOPs/element.
        return 14 * self.tokens * self.vocab

    def make_inputs(self, device: torch.device) -> Inputs:
        return {
            "logits": torch.randn(self.tokens, self.vocab, dtype=self.dtype, device=device, requires_grad=True),
            "labels": torch.randint(0, self.vocab, (self.tokens,), dtype=torch.long, device=device),
            "advantages": torch.randn(self.tokens, dtype=torch.float32, device=device),
            "old_log_probs": torch.randn(self.tokens, dtype=torch.float32, device=device) - 5.0,
        }


def _grpo_eager(
    logits: torch.Tensor, labels: torch.Tensor, advantages: torch.Tensor, old_log_probs: torch.Tensor
) -> torch.Tensor:
    log_probs = logits.float().log_softmax(-1)
    # clamp + labels>=0 guards mirror production code that handles ignore_index=-100;
    # labels here are always non-negative (randint), so the masks are dead in this benchmark.
    new_log_probs = log_probs.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    new_log_probs = torch.where(labels >= 0, new_log_probs, torch.zeros_like(new_log_probs))
    ratio = (new_log_probs - old_log_probs).exp()
    clipped_ratio = ratio.clamp(1.0 - _EPSILON_LOW, 1.0 + _EPSILON_HIGH)
    per_token_loss = torch.where(
        labels >= 0,
        -torch.minimum(ratio * advantages, clipped_ratio * advantages),
        torch.zeros_like(ratio),
    )
    return per_token_loss.mean()


def _triton_fwd(inputs: Inputs) -> dict:
    loss, _, _ = triton_grpo_loss_forward_backward(
        inputs["logits"],
        inputs["labels"],
        inputs["advantages"],
        inputs["old_log_probs"],
        grad_output=None,
        epsilon_low=_EPSILON_LOW,
        epsilon_high=_EPSILON_HIGH,
    )
    return {"loss": loss}


def _triton_fwd_bwd(inputs: Inputs) -> dict:
    loss, grad_logits, _ = triton_grpo_loss_forward_backward(
        inputs["logits"],
        inputs["labels"],
        inputs["advantages"],
        inputs["old_log_probs"],
        grad_output=1.0,
        epsilon_low=_EPSILON_LOW,
        epsilon_high=_EPSILON_HIGH,
    )
    return {"loss": loss, "grad_logits": grad_logits}


def benchmarks(
    dtypes: tuple[torch.dtype, ...],
    shapes: list[tuple[int, int]] | None = None,
) -> list[tuple[str, list, list]]:
    shapes = shapes if shapes is not None else _SHAPES
    variants = standard_pytorch_variants(
        _grpo_eager,
        input_keys=("logits", "labels", "advantages", "old_log_probs"),
        grad_input_keys=("logits",),
        output_key="loss",
    )
    if TritonConfig.enabled():
        variants.append(Variant(name="fast_llm_triton", fwd=_triton_fwd, fwd_bwd=_triton_fwd_bwd))
    return [
        (
            "grpo_loss",
            [GrpoLossCase(tokens=t, vocab=v, dtype=d) for d in dtypes for (t, v) in shapes],
            variants,
        )
    ]
