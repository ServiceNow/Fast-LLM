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
from tools.benchmark.runner import Case, Variant, run_benchmark
from tools.benchmark.utils import case_name, device

# (tokens, ffn_dim) — input tensor has shape (tokens, 2*ffn_dim) for gated.
_SHAPES = [
    (8192, 4096),  # 7B/13B models
    (8192, 8192),  # large
    (8192, 14336),  # 70B models
    (4096, 28672),  # MoE up-projection
]
_ACTIVATION = ActivationType.silu  # standard for Llama-style gated models
_DEFAULT_DTYPES = (torch.bfloat16,)


# --------------------------------------------------------------------------- inputs


def _make_mlp_inputs(tokens: int, ffn_dim: int, dtype: torch.dtype) -> dict:
    return {
        "input_": torch.randn(tokens, 2 * ffn_dim, dtype=dtype, device=device(), requires_grad=True),
        "grad_output": torch.randn(tokens, ffn_dim, dtype=dtype, device=device()),
        "gated": True,
        "activation_type": _ACTIVATION,
    }


# --------------------------------------------------------------------------- forward wrappers


def _pytorch_fwd(input_: torch.Tensor, gated: bool, activation_type: ActivationType) -> torch.Tensor:
    return torch_mlp_activation(input_, gated, activation_type)


_pytorch_compiled_default = torch.compile(_pytorch_fwd, mode="default", dynamic=False)
_pytorch_compiled_max = torch.compile(_pytorch_fwd, mode="max-autotune-no-cudagraphs", dynamic=False)


def _run_fwd(inp: dict, fn) -> dict:
    return {"output": fn(inp["input_"], inp["gated"], inp["activation_type"])}


def _run_fwd_fp32(inp: dict) -> dict:
    return {"output": _pytorch_fwd(inp["input_"].float(), inp["gated"], inp["activation_type"])}


def _run_fwd_triton(inp: dict) -> dict:
    output, _ = triton_mlp_activation_forward(inp["input_"], inp["gated"], inp["activation_type"])
    return {"output": output}


# --------------------------------------------------------------------------- fwd+bwd wrappers


def _run_fwd_bwd(inp: dict, fn) -> dict:
    output = fn(inp["input_"], inp["gated"], inp["activation_type"])
    output.backward(inp["grad_output"])
    return {"output": output.detach(), "grad_input": inp["input_"].grad}


def _run_fwd_bwd_fp32(inp: dict) -> dict:
    input_fp32 = inp["input_"].float().detach().requires_grad_(True)
    output = _pytorch_fwd(input_fp32, inp["gated"], inp["activation_type"])
    output.backward(inp["grad_output"].float())
    return {"output": output.detach(), "grad_input": input_fp32.grad}


def _run_fwd_bwd_triton(inp: dict) -> dict:
    output = triton_mlp_activation_autograd(inp["input_"], inp["gated"], inp["activation_type"])
    output.backward(inp["grad_output"])
    return {"output": output.detach(), "grad_input": inp["input_"].grad}


# --------------------------------------------------------------------------- variants


def _mlp_activation_variants() -> list[Variant]:
    variants = [
        Variant(
            name="fp32_reference",
            fwd=_run_fwd_fp32,
            fwd_bwd=_run_fwd_bwd_fp32,
            is_reference=True,
        ),
        Variant(
            name="pytorch_eager",
            fwd=lambda inp: _run_fwd(inp, _pytorch_fwd),
            fwd_bwd=lambda inp: _run_fwd_bwd(inp, _pytorch_fwd),
        ),
        Variant(
            name="pytorch_compiled",
            fwd=lambda inp: _run_fwd(inp, _pytorch_compiled_default),
            fwd_bwd=lambda inp: _run_fwd_bwd(inp, _pytorch_compiled_default),
        ),
        Variant(
            name="pytorch_compiled_max",
            fwd=lambda inp: _run_fwd(inp, _pytorch_compiled_max),
            fwd_bwd=lambda inp: _run_fwd_bwd(inp, _pytorch_compiled_max),
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


# --------------------------------------------------------------------------- cases


def _bytes_per_elem(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _mlp_activation_bytes(tokens: int, ffn_dim: int, dtype: torch.dtype) -> int:
    """fwd: read input (2*ffn_dim) + write output (ffn_dim).
    bwd: read grad_output (ffn_dim) + read input (2*ffn_dim) + write grad_input (2*ffn_dim).
    Total: 8 × tokens × ffn_dim × elem_size."""
    return 8 * tokens * ffn_dim * _bytes_per_elem(dtype)


def _mlp_activation_flops(tokens: int, ffn_dim: int) -> int:
    # gated silu: fwd ≈ 6 FLOPs/elem, bwd ≈ 8 FLOPs/elem, total ≈ 14 per output element.
    return 14 * tokens * ffn_dim


def _mlp_activation_cases(dtypes: tuple[torch.dtype, ...]) -> list[Case]:
    return [
        Case(
            name=case_name("mlp_activation", (tokens, ffn_dim), dtype),
            make_inputs=(lambda t=tokens, f=ffn_dim, d=dtype: _make_mlp_inputs(t, f, d)),
            expected_bytes=_mlp_activation_bytes(tokens, ffn_dim, dtype),
            expected_flops=_mlp_activation_flops(tokens, ffn_dim),
            compute_dtype=dtype,
        )
        for dtype in dtypes
        for tokens, ffn_dim in _SHAPES
    ]


# --------------------------------------------------------------------------- entry point


def run(verbose: bool = False, dtypes: tuple[torch.dtype, ...] | None = None) -> None:
    dtypes = tuple(dtypes) if dtypes else _DEFAULT_DTYPES
    run_benchmark(
        "mlp_activation (gated silu)", _mlp_activation_cases(dtypes), _mlp_activation_variants(), verbose=verbose
    )


if __name__ == "__main__":
    run()
