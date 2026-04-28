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
from tools.benchmark.runner import Case, Variant, run_benchmark
from tools.benchmark.utils import case_name, device

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


_grpo_compiled_default = torch.compile(_grpo_eager, mode="default", dynamic=False)
_grpo_compiled_max = torch.compile(_grpo_eager, mode="max-autotune-no-cudagraphs", dynamic=False)


def _run_fwd(inp: dict, fn) -> dict:
    return {"loss": fn(inp["logits"], inp["labels"], inp["advantages"], inp["old_log_probs"])}


def _run_fwd_fp32(inp: dict) -> dict:
    return {
        "loss": _grpo_eager(
            inp["logits"].float().detach().requires_grad_(),
            inp["labels"],
            inp["advantages"],
            inp["old_log_probs"],
        )
    }


def _run_fwd_bwd(inp: dict, fn) -> dict:
    loss = fn(inp["logits"], inp["labels"], inp["advantages"], inp["old_log_probs"])
    loss.backward()
    return {"loss": loss.detach(), "grad_logits": inp["logits"].grad}


def _run_fwd_bwd_fp32(inp: dict) -> dict:
    logits_fp32 = inp["logits"].float().detach().requires_grad_()
    loss = _grpo_eager(logits_fp32, inp["labels"], inp["advantages"], inp["old_log_probs"])
    loss.backward()
    return {"loss": loss.detach(), "grad_logits": logits_fp32.grad}


def _run_fwd_triton(inp: dict) -> dict:
    loss, _, _ = triton_grpo_loss_forward_backward(
        inp["logits"],
        inp["labels"],
        inp["advantages"],
        inp["old_log_probs"],
        grad_output=None,
        epsilon_low=_EPSILON_LOW,
        epsilon_high=_EPSILON_HIGH,
    )
    return {"loss": loss}


def _run_fwd_bwd_triton(inp: dict) -> dict:
    loss, grad_logits, _ = triton_grpo_loss_forward_backward(
        inp["logits"],
        inp["labels"],
        inp["advantages"],
        inp["old_log_probs"],
        grad_output=1.0,
        epsilon_low=_EPSILON_LOW,
        epsilon_high=_EPSILON_HIGH,
    )
    return {"loss": loss, "grad_logits": grad_logits}


def _grpo_variants() -> list[Variant]:
    variants = [
        Variant(
            name="fp32_reference",
            fwd=_run_fwd_fp32,
            fwd_bwd=_run_fwd_bwd_fp32,
            is_reference=True,
        ),
        Variant(
            name="pytorch_eager",
            fwd=lambda inp: _run_fwd(inp, _grpo_eager),
            fwd_bwd=lambda inp: _run_fwd_bwd(inp, _grpo_eager),
        ),
        Variant(
            name="pytorch_compiled",
            fwd=lambda inp: _run_fwd(inp, _grpo_compiled_default),
            fwd_bwd=lambda inp: _run_fwd_bwd(inp, _grpo_compiled_default),
        ),
        Variant(
            name="pytorch_compiled_max",
            fwd=lambda inp: _run_fwd(inp, _grpo_compiled_max),
            fwd_bwd=lambda inp: _run_fwd_bwd(inp, _grpo_compiled_max),
        ),
    ]
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_run_fwd_triton,
                fwd_bwd=_run_fwd_bwd_triton,
            )
        )
    return variants


def _bytes_per_elem(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _grpo_bytes(tokens: int, vocab: int, dtype: torch.dtype) -> int:
    elem = _bytes_per_elem(dtype)
    # fwd: read logits + bwd: read logits + write grad_logits
    logit_traffic = 3 * tokens * vocab * elem
    # labels (int64), advantages (fp32), old_log_probs (fp32)
    scalar_traffic = tokens * (8 + 4 + 4)
    return logit_traffic + scalar_traffic


def _grpo_flops(tokens: int, vocab: int) -> int:
    # Similar to cross_entropy labels: softmax (fwd) + grad (bwd) ≈ 14 FLOPs/element
    return 14 * tokens * vocab


def _grpo_cases(dtypes: tuple[torch.dtype, ...]) -> list[Case]:
    return [
        Case(
            name=case_name("grpo_loss", (tokens, vocab), dtype),
            make_inputs=lambda t=tokens, v=vocab, d=dtype: _make_grpo_inputs(t, v, d),
            expected_bytes=_grpo_bytes(tokens, vocab, dtype),
            expected_flops=_grpo_flops(tokens, vocab),
            compute_dtype=dtype,
        )
        for dtype in dtypes
        for tokens, vocab in _SHAPES
    ]


def run(verbose: bool = False, dtypes: tuple[torch.dtype, ...] | None = None) -> None:
    dtypes = tuple(dtypes) if dtypes else _DEFAULT_DTYPES
    run_benchmark("grpo_loss", _grpo_cases(dtypes), _grpo_variants(), verbose=verbose)


if __name__ == "__main__":
    run()
