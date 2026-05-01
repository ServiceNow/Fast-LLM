"""
Benchmark normalization kernels: LayerNorm and RMSNorm.

Both are fwd+bwd kernels. The Triton implementation in
`fast_llm/functional/triton/normalization.py` handles both flavors via the
`bias` argument (LayerNorm when given, RMSNorm when None) and writes parameter
gradients to Fast-LLM's `grad_buffer` attribute rather than autograd's `.grad`.

Comparisons:
- fp32_reference: torch.{layer,rms}_norm in fp32 (eager)
- pytorch_eager: torch.{layer,rms}_norm in the case dtype
- pytorch_compiled / pytorch_compiled_max: torch.compile of the above
- apex_fused: Apex fused_layer_norm_cuda (all widths, layer+rms norm)
- apex_fast: Apex fast_layer_norm contrib (layer norm only, restricted widths)
- fast_llm_triton: triton_normalization_autograd
"""

import torch

from fast_llm.functional.config import TritonConfig
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

# Activation shape (batch*seq, hidden). Numel fixed at 32M to mimic a constant
# training memory budget across model widths; hidden swept from 1K to 16K covers
# small models through Llama-405B / wide-MoE territory.
_SHAPES = [
    (32768, 1024),
    (16384, 2048),
    (8192, 4096),
    (4096, 8192),
    (2048, 16384),
]
_DEFAULT_DTYPES = (torch.bfloat16,)
_EPS = 1e-5


def _setup_param(tensor: torch.Tensor) -> torch.Tensor:
    """Triton's normalization backward writes weight/bias gradients to a
    `grad_buffer` attribute (Fast-LLM convention) instead of autograd's `.grad`.
    Wire up the buffer + zero-flag the kernel expects."""
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
    """Pull the parameter gradient from wherever the kernel wrote it.
    Triton writes to `grad_buffer`; autograd writes to `.grad`."""
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
        # apex_fast only supports widths in _PERSIST_LN_SIZES; all shapes in _SHAPES qualify.
        extras["apex_fast"] = _layer_norm_apex_fast
    variants = standard_fwd_bwd_pytorch_variants(
        _layer_norm_eager,
        input_keys=("input", "weight", "bias"),
        grad_input_keys=("input", "weight", "bias"),
        grad_output_key="grad_output",
        extra_functions=extras,
    )
    if TritonConfig.enabled():
        variants.append(
            Variant(name="fast_llm_triton", fwd=_layer_norm_triton_fwd, fwd_bwd=_layer_norm_triton_fwd_bwd)
        )
    return variants


def _rms_norm_variants() -> list[Variant]:
    extras: dict = {}
    if fused_normalization_available:
        extras["apex_fused"] = _rms_norm_apex_fused
    variants = standard_fwd_bwd_pytorch_variants(
        _rms_norm_eager,
        input_keys=("input", "weight"),
        grad_input_keys=("input", "weight"),
        grad_output_key="grad_output",
        extra_functions=extras,
    )
    if TritonConfig.enabled():
        variants.append(Variant(name="fast_llm_triton", fwd=_rms_norm_triton_fwd, fwd_bwd=_rms_norm_triton_fwd_bwd))
    return variants


def _layer_norm_bytes(rows: int, cols: int, dtype: torch.dtype) -> int:
    """Approximate fwd+bwd memory traffic for LayerNorm.
    fwd reads input + weight + bias and writes output (also stores inv_var).
    bwd reads grad_output, output, weight, bias, inv_var; writes grad_input,
    grad_weight, grad_bias. Activation tensors dominate."""
    activations = 4 * rows * cols * dtype.itemsize  # fwd in/out + bwd grad_in/out
    parameters = 6 * cols * dtype.itemsize  # weight, bias × (read + grad write) twice
    return activations + parameters


def _rms_norm_bytes(rows: int, cols: int, dtype: torch.dtype) -> int:
    activations = 4 * rows * cols * dtype.itemsize
    parameters = 3 * cols * dtype.itemsize
    return activations + parameters


def _layer_norm_flops(rows: int, cols: int) -> int:
    """Approximate fwd+bwd FLOPs for LayerNorm.
    fwd: mean (1), variance (2), normalize (2), scale+shift (2) ≈ 7 per element.
    bwd: ~2x fwd."""
    return 21 * rows * cols


def _rms_norm_flops(rows: int, cols: int) -> int:
    """Same idea as LayerNorm but no mean subtraction or bias."""
    return 15 * rows * cols


def benchmarks(
    dtypes: tuple[torch.dtype, ...] | None = None,
    shapes: list[tuple[int, int]] | None = None,
) -> list[tuple[str, list, list]]:
    dtypes = tuple(dtypes) if dtypes else _DEFAULT_DTYPES
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


if __name__ == "__main__":
    run()
