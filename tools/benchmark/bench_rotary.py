"""Rotary position embeddings. The Triton kernel is in-place; backward is an
identical rotation with conjugated frequencies, so only fwd is benchmarked."""

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.rotary import triton_rotary_
from tools.benchmark.runner import Variant
from tools.benchmark.utils import bench_main, device, make_cases

# (tokens, num_heads, head_size) — tokens = batch * seq_len
_SHAPES = [
    (4096, 32, 128),  # 7B/13B, 4K context
    (8192, 32, 128),  # 7B/13B, 8K context
    (4096, 64, 128),  # 70B / MoE, 4K context
    (4096, 8, 128),  # GQA key-value heads, 4K context
]


def _make_rotary_inputs(tokens: int, num_heads: int, head_size: int, dtype: torch.dtype) -> dict:
    rotary_dim = head_size // 2
    input_ = torch.randn(tokens, num_heads, head_size, dtype=dtype, device=device())
    return {
        "input_": input_,
        "work": input_.clone(),
        "frequencies": torch.randn(tokens, 2 * rotary_dim, dtype=torch.float32, device=device()),
    }


def _rotary_eager(input_: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    rotary_dim = frequencies.shape[-1] // 2
    freq_re = frequencies[:, :rotary_dim].unsqueeze(1)
    freq_im = frequencies[:, rotary_dim:].unsqueeze(1)
    x_re, x_im = input_.chunk(2, dim=-1)
    out_re = x_re * freq_re - x_im * freq_im
    out_im = x_im * freq_re + x_re * freq_im
    return torch.cat([out_re, out_im], dim=-1)


_rotary_compiled_default = torch.compile(_rotary_eager, mode="default", dynamic=False)
_rotary_compiled_max = torch.compile(_rotary_eager, mode="max-autotune-no-cudagraphs", dynamic=False)


def _rotary_bytes(tokens: int, num_heads: int, head_size: int, dtype: torch.dtype) -> int:
    # frequencies are float32, hence the extra 4 bytes per token×head_size.
    return 2 * tokens * num_heads * head_size * dtype.itemsize + tokens * head_size * 4


def _rotary_flops(tokens: int, num_heads: int, head_size: int) -> int:
    # 6 FLOPs per (re, im) element pair: 4 muls + 2 add/sub.
    return 6 * tokens * num_heads * (head_size // 2)


def _rotary_variants() -> list[Variant]:
    variants = [
        Variant(
            name="fp32_reference",
            fwd=lambda inputs: {"output": _rotary_eager(inputs["input_"].float(), inputs["frequencies"])},
            is_reference=True,
        ),
        Variant(
            name="pytorch_eager",
            fwd=lambda inputs: {"output": _rotary_eager(inputs["input_"], inputs["frequencies"])},
        ),
        Variant(
            name="pytorch_compiled",
            fwd=lambda inputs: {"output": _rotary_compiled_default(inputs["input_"], inputs["frequencies"])},
        ),
        Variant(
            name="pytorch_compiled_max",
            fwd=lambda inputs: {"output": _rotary_compiled_max(inputs["input_"], inputs["frequencies"])},
        ),
    ]
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=lambda inputs: {"output": triton_rotary_(inputs["work"], inputs["frequencies"])},
                reset_inputs=lambda inputs: inputs["work"].copy_(inputs["input_"]),
            )
        )
    return variants


def benchmarks(
    dtypes: tuple[torch.dtype, ...],
    shapes: list[tuple[int, int, int]] | None = None,
) -> list[tuple[str, list, list]]:
    shapes = shapes if shapes is not None else _SHAPES
    return [
        (
            "rotary",
            make_cases("rotary", dtypes, shapes, _make_rotary_inputs, _rotary_bytes, _rotary_flops),
            _rotary_variants(),
        )
    ]


run = bench_main(benchmarks)
