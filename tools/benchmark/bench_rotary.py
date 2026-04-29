"""
Benchmark rotary position embeddings.

The Triton kernel (`triton_rotary_`) operates in-place on (tokens, num_heads,
head_size) tensors, loading pre-computed (cos, sin) frequencies from
(tokens, 2*rotary_dim).  The backward is an identical rotation call with
conjugated frequencies — same cost — so only fwd is benchmarked.

Shapes sweep (tokens, num_heads, head_size) across typical attention configs:
- 32 heads × 128 → 7B/13B models
- 64 heads × 128 → 70B / MoE models
- 8 heads × 128 → GQA key-value heads (Llama 3)
"""

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.rotary import triton_rotary_
from tools.benchmark.runner import Case, Variant, run_benchmark
from tools.benchmark.utils import case_name, device

# (tokens, num_heads, head_size) — tokens = batch * seq_len
_SHAPES = [
    (4096, 32, 128),  # 7B/13B, 4K context
    (8192, 32, 128),  # 7B/13B, 8K context
    (4096, 64, 128),  # 70B / MoE, 4K context
    (4096, 8, 128),  # GQA key-value heads, 4K context
]
_DEFAULT_DTYPES = (torch.bfloat16,)


def _make_rotary_inputs(tokens: int, num_heads: int, head_size: int, dtype: torch.dtype) -> dict:
    rotary_dim = head_size // 2
    return {
        "input_": torch.randn(tokens, num_heads, head_size, dtype=dtype, device=device()),
        "frequencies": torch.randn(tokens, 2 * rotary_dim, dtype=torch.float32, device=device()),
    }


def _rotary_eager(input_: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """Non-in-place full rotary (rotary_dim = head_size / 2)."""
    rotary_dim = frequencies.shape[-1] // 2
    freq_re = frequencies[:, :rotary_dim].unsqueeze(1)  # (tokens, 1, rotary_dim)
    freq_im = frequencies[:, rotary_dim:].unsqueeze(1)
    x_re, x_im = input_.chunk(2, dim=-1)
    out_re = x_re * freq_re - x_im * freq_im
    out_im = x_im * freq_re + x_re * freq_im
    return torch.cat([out_re, out_im], dim=-1)


_rotary_compiled_default = torch.compile(_rotary_eager, mode="default", dynamic=False)
_rotary_compiled_max = torch.compile(_rotary_eager, mode="max-autotune-no-cudagraphs", dynamic=False)


def _rotary_bytes(tokens: int, num_heads: int, head_size: int, dtype: torch.dtype) -> int:
    elem = torch.tensor([], dtype=dtype).element_size()
    # Read + write input tensor; frequencies are float32.
    return 2 * tokens * num_heads * head_size * elem + tokens * head_size * 4


def _rotary_flops(tokens: int, num_heads: int, head_size: int) -> int:
    # 6 FLOPs per (re, im) element pair: 4 muls + 2 add/sub.
    return 6 * tokens * num_heads * (head_size // 2)


def _rotary_variants() -> list[Variant]:
    variants = [
        Variant(
            name="fp32_reference",
            fwd=lambda inp: {"output": _rotary_eager(inp["input_"].float(), inp["frequencies"])},
            is_reference=True,
        ),
        Variant(
            name="pytorch_eager",
            fwd=lambda inp: {"output": _rotary_eager(inp["input_"], inp["frequencies"])},
        ),
        Variant(
            name="pytorch_compiled",
            fwd=lambda inp: {"output": _rotary_compiled_default(inp["input_"], inp["frequencies"])},
        ),
        Variant(
            name="pytorch_compiled_max",
            fwd=lambda inp: {"output": _rotary_compiled_max(inp["input_"], inp["frequencies"])},
        ),
    ]
    if TritonConfig.enabled():
        # triton_rotary_ is in-place; clone so the benchmark input stays intact.
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=lambda inp: {"output": triton_rotary_(inp["input_"].clone(), inp["frequencies"])},
            )
        )
    return variants


def _rotary_cases(dtypes: tuple[torch.dtype, ...]) -> list[Case]:
    return [
        Case(
            name=case_name("rotary", (tokens, num_heads, head_size), dtype),
            make_inputs=(lambda t=tokens, h=num_heads, s=head_size, d=dtype: _make_rotary_inputs(t, h, s, d)),
            expected_bytes=_rotary_bytes(tokens, num_heads, head_size, dtype),
            expected_flops=_rotary_flops(tokens, num_heads, head_size),
            compute_dtype=dtype,
        )
        for dtype in dtypes
        for tokens, num_heads, head_size in _SHAPES
    ]


def run(verbose: bool = False, dtypes: tuple[torch.dtype, ...] | None = None) -> None:
    dtypes = tuple(dtypes) if dtypes else _DEFAULT_DTYPES
    run_benchmark("rotary", _rotary_cases(dtypes), _rotary_variants(), verbose=verbose)


if __name__ == "__main__":
    run()
