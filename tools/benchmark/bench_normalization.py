"""LayerNorm and RMSNorm. The Triton implementation handles both via the
`bias` argument (LayerNorm when given, RMSNorm when None) and writes parameter
gradients to Fast-LLM's `grad_buffer` attribute rather than autograd's `.grad`."""

import torch

from fast_llm.functional.triton.normalization import triton_normalization_autograd
from fast_llm.layers.common.normalization.normalization import (
    FastLayerNorm,
    FusedLayerNorm,
    FusedRMSNorm,
    fast_normalization_available,
    fused_normalization_available,
)
from tools.benchmark.runner import Variant
from tools.benchmark.utils import bench_main, device, make_cases, standard_fwd_bwd_pytorch_variants

# (batch*seq, hidden). Numel fixed at 32M to mimic a constant training memory
# budget across model widths; hidden swept from 1K to 16K.
_SHAPES = [
    (32768, 1024),
    (16384, 2048),
    (8192, 4096),
    (4096, 8192),
    (2048, 16384),
]
_EPS = 1e-5


def _setup_param(tensor: torch.Tensor) -> torch.Tensor:
    """Triton's normalization backward writes weight/bias gradients to a
    `grad_buffer` attribute (Fast-LLM convention) instead of autograd's `.grad`."""
    tensor.grad_buffer = torch.zeros_like(tensor)
    tensor.param_grad_is_zero = True
    return tensor


def _make_layer_norm_inputs(rows: int, cols: int, dtype: torch.dtype) -> dict:
    return {
        "input": torch.randn(rows, cols, dtype=dtype, device=device(), requires_grad=True),
        "weight": _setup_param(torch.randn(cols, dtype=dtype, device=device(), requires_grad=True)),
        "bias": _setup_param(torch.zeros(cols, dtype=dtype, device=device(), requires_grad=True)),
        "grad_output": torch.randn(rows, cols, dtype=dtype, device=device()),
    }


def _make_rms_norm_inputs(rows: int, cols: int, dtype: torch.dtype) -> dict:
    return {
        "input": torch.randn(rows, cols, dtype=dtype, device=device(), requires_grad=True),
        "weight": _setup_param(torch.randn(cols, dtype=dtype, device=device(), requires_grad=True)),
        "grad_output": torch.randn(rows, cols, dtype=dtype, device=device()),
    }


def _layer_norm_eager(input_, weight, bias):
    return torch.layer_norm(input_, weight.shape, weight, bias, _EPS)


def _rms_norm_eager(input_, weight):
    return torch.rms_norm(input_, weight.shape, weight, _EPS)


def _layer_norm_apex_fused(input_, weight, bias):
    return FusedLayerNorm.apply(input_, weight.shape, weight, bias, _EPS)


def _layer_norm_apex_fast(input_, weight, bias):
    return FastLayerNorm.apply(input_, weight.shape, weight, bias, _EPS)


def _rms_norm_apex_fused(input_, weight):
    return FusedRMSNorm.apply(input_, weight.shape, weight, _EPS)


def _param_grad(param: torch.Tensor) -> torch.Tensor:
    """Triton writes to `grad_buffer`; autograd writes to `.grad`."""
    return param.grad if param.grad is not None else param.grad_buffer


def _layer_norm_triton_fwd(inputs: dict) -> dict:
    return {
        "output": triton_normalization_autograd(inputs["input"], inputs["weight"], inputs["bias"], _EPS, True, False)
    }


def _layer_norm_triton_fwd_bwd(inputs: dict) -> dict:
    output = triton_normalization_autograd(inputs["input"], inputs["weight"], inputs["bias"], _EPS, True, False)
    output.backward(inputs["grad_output"])
    return {
        "output": output.detach(),
        "grad_input": inputs["input"].grad,
        "grad_weight": _param_grad(inputs["weight"]),
        "grad_bias": _param_grad(inputs["bias"]),
    }


def _rms_norm_triton_fwd(inputs: dict) -> dict:
    return {"output": triton_normalization_autograd(inputs["input"], inputs["weight"], None, _EPS, True, False)}


def _rms_norm_triton_fwd_bwd(inputs: dict) -> dict:
    output = triton_normalization_autograd(inputs["input"], inputs["weight"], None, _EPS, True, False)
    output.backward(inputs["grad_output"])
    return {
        "output": output.detach(),
        "grad_input": inputs["input"].grad,
        "grad_weight": _param_grad(inputs["weight"]),
    }


def _layer_norm_variants() -> list[Variant]:
    extras: dict = {}
    if fused_normalization_available:
        extras["apex_fused"] = _layer_norm_apex_fused
    if fast_normalization_available:
        extras["apex_fast"] = _layer_norm_apex_fast
    return standard_fwd_bwd_pytorch_variants(
        _layer_norm_eager,
        input_keys=("input", "weight", "bias"),
        grad_input_keys=("input", "weight", "bias"),
        grad_output_key="grad_output",
        extra_functions=extras,
        triton_fwd=_layer_norm_triton_fwd,
        triton_fwd_bwd=_layer_norm_triton_fwd_bwd,
    )


def _rms_norm_variants() -> list[Variant]:
    extras: dict = {}
    if fused_normalization_available:
        extras["apex_fused"] = _rms_norm_apex_fused
    return standard_fwd_bwd_pytorch_variants(
        _rms_norm_eager,
        input_keys=("input", "weight"),
        grad_input_keys=("input", "weight"),
        grad_output_key="grad_output",
        extra_functions=extras,
        triton_fwd=_rms_norm_triton_fwd,
        triton_fwd_bwd=_rms_norm_triton_fwd_bwd,
    )


def _layer_norm_bytes(rows: int, cols: int, dtype: torch.dtype) -> int:
    activations = 4 * rows * cols * dtype.itemsize  # fwd in/out + bwd grad_in/out
    parameters = 6 * cols * dtype.itemsize  # weight, bias × (read + grad write) twice
    return activations + parameters


def _rms_norm_bytes(rows: int, cols: int, dtype: torch.dtype) -> int:
    activations = 4 * rows * cols * dtype.itemsize
    parameters = 3 * cols * dtype.itemsize
    return activations + parameters


def _layer_norm_flops(rows: int, cols: int) -> int:
    # fwd ≈ 7 per element (mean, variance, normalize, scale+shift); bwd ≈ 2× fwd.
    return 21 * rows * cols


def _rms_norm_flops(rows: int, cols: int) -> int:
    return 15 * rows * cols


def benchmarks(
    dtypes: tuple[torch.dtype, ...],
    shapes: list[tuple[int, int]] | None = None,
) -> list[tuple[str, list, list]]:
    shapes = shapes if shapes is not None else _SHAPES
    return [
        (
            "normalization: layer_norm",
            make_cases("layer_norm", dtypes, shapes, _make_layer_norm_inputs, _layer_norm_bytes, _layer_norm_flops),
            _layer_norm_variants(),
        ),
        (
            "normalization: rms_norm",
            make_cases("rms_norm", dtypes, shapes, _make_rms_norm_inputs, _rms_norm_bytes, _rms_norm_flops),
            _rms_norm_variants(),
        ),
    ]


run = bench_main(benchmarks)
