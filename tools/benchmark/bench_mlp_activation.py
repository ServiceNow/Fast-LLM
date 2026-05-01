"""
Benchmark the fused MLP activation kernel.

The Triton kernel (`triton_mlp_activation_autograd`) fuses the element-wise
activation and (for gated models) the gated multiply into a single pass.  For
gated SiLU the fwd input is (tokens, 2*ffn_dim) — [gate_proj, up_proj]
concatenated — and the output is (tokens, ffn_dim).

Comparisons:
- fp32_reference: torch_mlp_activation in fp32 with autograd
- pytorch_eager: torch_mlp_activation in compute dtype
- pytorch_compiled / pytorch_compiled_max: torch.compile of the above
- fast_llm_triton: triton_mlp_activation_autograd

Shapes fix tokens=8192 and sweep ffn_dim across typical MLP widths.
"""

import torch

from fast_llm.functional.config import ActivationType, TritonConfig
from fast_llm.functional.triton.mlp import (
    torch_mlp_activation,
    triton_mlp_activation_autograd,
    triton_mlp_activation_forward,
)
from tools.benchmark.runner import Variant
from tools.benchmark.utils import bench_main, device, make_cases, standard_fwd_bwd_pytorch_variants

# (tokens, ffn_dim) — input tensor has shape (tokens, 2*ffn_dim) for gated.
_SHAPES = [
    (8192, 4096),  # 7B/13B models
    (8192, 8192),  # large
    (8192, 14336),  # 70B models
    (4096, 28672),  # MoE up-projection
]
_ACTIVATION = ActivationType.silu  # standard for Llama-style gated models
_DEFAULT_DTYPES = (torch.bfloat16,)


def _make_mlp_inputs(tokens: int, ffn_dim: int, dtype: torch.dtype) -> dict:
    return {
        "input": torch.randn(tokens, 2 * ffn_dim, dtype=dtype, device=device(), requires_grad=True),
        "grad_output": torch.randn(tokens, ffn_dim, dtype=dtype, device=device()),
        "gated": True,
        "activation_type": _ACTIVATION,
    }


def _triton_fwd(inputs: dict) -> dict:
    output, _ = triton_mlp_activation_forward(inputs["input"], inputs["gated"], inputs["activation_type"])
    return {"output": output}


def _triton_fwd_bwd(inputs: dict) -> dict:
    output = triton_mlp_activation_autograd(inputs["input"], inputs["gated"], inputs["activation_type"])
    output.backward(inputs["grad_output"])
    return {"output": output.detach(), "grad_input": inputs["input"].grad}


def _mlp_activation_variants() -> list[Variant]:
    variants = standard_fwd_bwd_pytorch_variants(
        torch_mlp_activation,
        input_keys=("input", "gated", "activation_type"),
        grad_input_keys=("input",),
        grad_output_key="grad_output",
    )
    if TritonConfig.enabled():
        variants.append(Variant(name="fast_llm_triton", fwd=_triton_fwd, fwd_bwd=_triton_fwd_bwd))
    return variants


def _mlp_activation_bytes(tokens: int, ffn_dim: int, dtype: torch.dtype) -> int:
    """fwd: read input (2*ffn_dim) + write output (ffn_dim).
    bwd: read grad_output (ffn_dim) + read input (2*ffn_dim) + write grad_input (2*ffn_dim).
    Total: 8 × tokens × ffn_dim × elem_size."""
    return 8 * tokens * ffn_dim * dtype.itemsize


def _mlp_activation_flops(tokens: int, ffn_dim: int) -> int:
    # gated silu: fwd ≈ 6 FLOPs/element, bwd ≈ 8 FLOPs/element, total ≈ 14 per output element.
    return 14 * tokens * ffn_dim


def benchmarks(
    dtypes: tuple[torch.dtype, ...] | None = None,
    shapes: list[tuple[int, int]] | None = None,
) -> list[tuple[str, list, list]]:
    dtypes = tuple(dtypes) if dtypes else _DEFAULT_DTYPES
    shapes = shapes if shapes is not None else _SHAPES
    return [
        (
            "mlp_activation (gated silu)",
            make_cases(
                "mlp_activation", dtypes, shapes, _make_mlp_inputs, _mlp_activation_bytes, _mlp_activation_flops
            ),
            _mlp_activation_variants(),
        )
    ]


run = bench_main(benchmarks)


if __name__ == "__main__":
    run()
