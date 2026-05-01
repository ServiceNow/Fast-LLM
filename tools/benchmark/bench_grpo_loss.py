"""
Benchmark the fused GRPO loss kernel.

GRPO (Group Relative Policy Optimization) loss computes a clipped importance-weighted
policy gradient per token: loss = -min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv),
where ratio = exp(log_prob_new - log_prob_old).

The Triton kernel fuses softmax, log-prob extraction, ratio computation, clipping, and
the backward gradient into a single pass over logits — same structure as the cross_entropy
kernel.

Comparisons:
- fp32_reference: PyTorch GRPO in fp32
- pytorch_eager: PyTorch GRPO in compute dtype
- pytorch_compiled / pytorch_compiled_max: torch.compile of the above
- fast_llm_triton: triton_grpo_loss_forward_backward

Shapes match bench_entropy_loss: tokens=4096, vocab swept over 32K/64K/128K.
"""

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.grpo_loss import triton_grpo_loss_forward_backward
from tools.benchmark.runner import Variant
from tools.benchmark.utils import bench_main, device, make_cases, standard_fwd_bwd_pytorch_variants

_SHAPES = [
    (4096, 32768),
    (4096, 65536),
    (4096, 131072),
]
_DEFAULT_DTYPES = (torch.bfloat16,)
_EPSILON_LOW = 0.2
_EPSILON_HIGH = 0.2


def _make_grpo_inputs(tokens: int, vocab: int, dtype: torch.dtype) -> dict:
    return {
        "logits": torch.randn(tokens, vocab, dtype=dtype, device=device(), requires_grad=True),
        "labels": torch.randint(0, vocab, (tokens,), dtype=torch.long, device=device()),
        "advantages": torch.randn(tokens, dtype=torch.float32, device=device()),
        "old_log_probs": torch.randn(tokens, dtype=torch.float32, device=device()) - 5.0,
    }


def _grpo_eager(logits: torch.Tensor, labels: torch.Tensor, advantages: torch.Tensor, old_log_probs: torch.Tensor):
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


def _reset_logits_grad(inputs: dict) -> None:
    inputs["logits"].grad = None


def _triton_fwd(inputs: dict) -> dict:
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


def _triton_fwd_bwd(inputs: dict) -> dict:
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


def _grpo_variants() -> list[Variant]:
    variants = standard_fwd_bwd_pytorch_variants(
        _grpo_eager,
        input_keys=("logits", "labels", "advantages", "old_log_probs"),
        grad_input_keys=("logits",),
        output_key="loss",
        reset_inputs=_reset_logits_grad,
    )
    if TritonConfig.enabled():
        variants.append(Variant(name="fast_llm_triton", fwd=_triton_fwd, fwd_bwd=_triton_fwd_bwd))
    return variants


def _grpo_bytes(tokens: int, vocab: int, dtype: torch.dtype) -> int:
    # fwd: read logits + bwd: read logits + write grad_logits
    logit_traffic = 3 * tokens * vocab * dtype.itemsize
    # labels (int64), advantages (fp32), old_log_probs (fp32)
    scalar_traffic = tokens * (8 + 4 + 4)
    return logit_traffic + scalar_traffic


def _grpo_flops(tokens: int, vocab: int) -> int:
    # Similar to cross_entropy labels: softmax (fwd) + grad (bwd) ≈ 14 FLOPs/element
    return 14 * tokens * vocab


def benchmarks(
    dtypes: tuple[torch.dtype, ...] | None = None,
    shapes: list[tuple[int, int]] | None = None,
) -> list[tuple[str, list, list]]:
    dtypes = tuple(dtypes) if dtypes else _DEFAULT_DTYPES
    shapes = shapes if shapes is not None else _SHAPES
    return [
        (
            "grpo_loss",
            make_cases("grpo_loss", dtypes, shapes, _make_grpo_inputs, _grpo_bytes, _grpo_flops),
            _grpo_variants(),
        )
    ]


run = bench_main(benchmarks)


if __name__ == "__main__":
    run()
