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
    _fast_normalization_available,
    _fused_normalization_available,
)
from tools.benchmark.runner import Case, Variant, run_benchmark
from tools.benchmark.utils import case_name, device

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


# --------------------------------------------------------------------------- input setup


def _setup_param(tensor: torch.Tensor) -> torch.Tensor:
    """Triton's normalization backward writes weight/bias gradients to a
    `grad_buffer` attribute (Fast-LLM convention) instead of autograd's `.grad`.
    Wire up the buffer + zero-flag the kernel expects."""
    tensor.grad_buffer = torch.zeros_like(tensor)
    tensor.param_grad_is_zero = True
    return tensor


def _to_fp32_input(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().detach().requires_grad_()


def _to_fp32_param(tensor: torch.Tensor) -> torch.Tensor:
    return _setup_param(tensor.float().detach().requires_grad_())


def _make_layer_norm_inputs(rows: int, cols: int, dtype: torch.dtype) -> dict:
    return {
        "input_": torch.randn(rows, cols, dtype=dtype, device=device(), requires_grad=True),
        "weight": _setup_param(torch.randn(cols, dtype=dtype, device=device(), requires_grad=True)),
        "bias": _setup_param(torch.zeros(cols, dtype=dtype, device=device(), requires_grad=True)),
        "grad_output": torch.randn(rows, cols, dtype=dtype, device=device()),
    }


def _make_rms_norm_inputs(rows: int, cols: int, dtype: torch.dtype) -> dict:
    return {
        "input_": torch.randn(rows, cols, dtype=dtype, device=device(), requires_grad=True),
        "weight": _setup_param(torch.randn(cols, dtype=dtype, device=device(), requires_grad=True)),
        "grad_output": torch.randn(rows, cols, dtype=dtype, device=device()),
    }


def _layer_norm_inputs_fp32(inp: dict) -> dict:
    return {
        "input_": _to_fp32_input(inp["input_"]),
        "weight": _to_fp32_param(inp["weight"]),
        "bias": _to_fp32_param(inp["bias"]),
        "grad_output": inp["grad_output"].float(),
    }


def _rms_norm_inputs_fp32(inp: dict) -> dict:
    return {
        "input_": _to_fp32_input(inp["input_"]),
        "weight": _to_fp32_param(inp["weight"]),
        "grad_output": inp["grad_output"].float(),
    }


# --------------------------------------------------------------------------- forward functions


def _layer_norm_eager(input_, weight, bias):
    return torch.layer_norm(input_, weight.shape, weight, bias, _EPS)


def _rms_norm_eager(input_, weight):
    return torch.rms_norm(input_, weight.shape, weight, _EPS)


def _layer_norm_triton(input_, weight, bias):
    return triton_normalization_autograd(input_, weight, bias, _EPS, True, False)


def _rms_norm_triton(input_, weight):
    return triton_normalization_autograd(input_, weight, None, _EPS, True, False)


def _layer_norm_apex_fused(input_, weight, bias):
    return FusedLayerNorm.apply(input_, weight.shape, weight, bias, _EPS)


def _layer_norm_apex_fast(input_, weight, bias):
    return FastLayerNorm.apply(input_, weight.shape, weight, bias, _EPS)


def _rms_norm_apex_fused(input_, weight):
    return FusedRMSNorm.apply(input_, weight.shape, weight, _EPS)


_layer_compiled_default = torch.compile(_layer_norm_eager, mode="default", dynamic=False)
_layer_compiled_max = torch.compile(_layer_norm_eager, mode="max-autotune-no-cudagraphs", dynamic=False)
_rms_compiled_default = torch.compile(_rms_norm_eager, mode="default", dynamic=False)
_rms_compiled_max = torch.compile(_rms_norm_eager, mode="max-autotune-no-cudagraphs", dynamic=False)


# --------------------------------------------------------------------------- variant wrappers


def _param_grad(param: torch.Tensor) -> torch.Tensor:
    """Pull the parameter gradient from wherever the kernel wrote it.
    Triton writes to `grad_buffer`; autograd writes to `.grad`."""
    return param.grad if param.grad is not None else param.grad_buffer


def _run_layer_fwd(inp: dict, fn) -> dict:
    return {"output": fn(inp["input_"], inp["weight"], inp["bias"])}


def _run_layer_fwd_bwd(inp: dict, fn) -> dict:
    output = fn(inp["input_"], inp["weight"], inp["bias"])
    output.backward(inp["grad_output"])
    return {
        "grad_input": inp["input_"].grad,
        "grad_weight": _param_grad(inp["weight"]),
        "grad_bias": _param_grad(inp["bias"]),
    }


def _run_rms_fwd(inp: dict, fn) -> dict:
    return {"output": fn(inp["input_"], inp["weight"])}


def _run_rms_fwd_bwd(inp: dict, fn) -> dict:
    output = fn(inp["input_"], inp["weight"])
    output.backward(inp["grad_output"])
    return {
        "grad_input": inp["input_"].grad,
        "grad_weight": _param_grad(inp["weight"]),
    }


# --------------------------------------------------------------------------- variants


def _layer_norm_variants() -> list[Variant]:
    variants = [
        Variant(
            name="fp32_reference",
            fwd=lambda inp: _run_layer_fwd(_layer_norm_inputs_fp32(inp), _layer_norm_eager),
            fwd_bwd=lambda inp: _run_layer_fwd_bwd(_layer_norm_inputs_fp32(inp), _layer_norm_eager),
            is_reference=True,
        ),
        Variant(
            name="pytorch_eager",
            fwd=lambda inp: _run_layer_fwd(inp, _layer_norm_eager),
            fwd_bwd=lambda inp: _run_layer_fwd_bwd(inp, _layer_norm_eager),
        ),
        Variant(
            name="pytorch_compiled",
            fwd=lambda inp: _run_layer_fwd(inp, _layer_compiled_default),
            fwd_bwd=lambda inp: _run_layer_fwd_bwd(inp, _layer_compiled_default),
        ),
        Variant(
            name="pytorch_compiled_max",
            fwd=lambda inp: _run_layer_fwd(inp, _layer_compiled_max),
            fwd_bwd=lambda inp: _run_layer_fwd_bwd(inp, _layer_compiled_max),
        ),
    ]
    if _fused_normalization_available:
        variants.append(
            Variant(
                name="apex_fused",
                fwd=lambda inp: _run_layer_fwd(inp, _layer_norm_apex_fused),
                fwd_bwd=lambda inp: _run_layer_fwd_bwd(inp, _layer_norm_apex_fused),
            )
        )
    if _fast_normalization_available:
        # apex_fast only supports widths in _PERSIST_LN_SIZES; all shapes in _SHAPES qualify.
        variants.append(
            Variant(
                name="apex_fast",
                fwd=lambda inp: _run_layer_fwd(inp, _layer_norm_apex_fast),
                fwd_bwd=lambda inp: _run_layer_fwd_bwd(inp, _layer_norm_apex_fast),
            )
        )
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=lambda inp: _run_layer_fwd(inp, _layer_norm_triton),
                fwd_bwd=lambda inp: _run_layer_fwd_bwd(inp, _layer_norm_triton),
            )
        )
    return variants


def _rms_norm_variants() -> list[Variant]:
    variants = [
        Variant(
            name="fp32_reference",
            fwd=lambda inp: _run_rms_fwd(_rms_norm_inputs_fp32(inp), _rms_norm_eager),
            fwd_bwd=lambda inp: _run_rms_fwd_bwd(_rms_norm_inputs_fp32(inp), _rms_norm_eager),
            is_reference=True,
        ),
        Variant(
            name="pytorch_eager",
            fwd=lambda inp: _run_rms_fwd(inp, _rms_norm_eager),
            fwd_bwd=lambda inp: _run_rms_fwd_bwd(inp, _rms_norm_eager),
        ),
        Variant(
            name="pytorch_compiled",
            fwd=lambda inp: _run_rms_fwd(inp, _rms_compiled_default),
            fwd_bwd=lambda inp: _run_rms_fwd_bwd(inp, _rms_compiled_default),
        ),
        Variant(
            name="pytorch_compiled_max",
            fwd=lambda inp: _run_rms_fwd(inp, _rms_compiled_max),
            fwd_bwd=lambda inp: _run_rms_fwd_bwd(inp, _rms_compiled_max),
        ),
    ]
    if _fused_normalization_available:
        variants.append(
            Variant(
                name="apex_fused",
                fwd=lambda inp: _run_rms_fwd(inp, _rms_norm_apex_fused),
                fwd_bwd=lambda inp: _run_rms_fwd_bwd(inp, _rms_norm_apex_fused),
            )
        )
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=lambda inp: _run_rms_fwd(inp, _rms_norm_triton),
                fwd_bwd=lambda inp: _run_rms_fwd_bwd(inp, _rms_norm_triton),
            )
        )
    return variants


# --------------------------------------------------------------------------- cases


def _bytes_per_elem(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _layer_norm_bytes(rows: int, cols: int, dtype: torch.dtype) -> int:
    """Approximate fwd+bwd memory traffic for LayerNorm.
    fwd reads input + weight + bias and writes output (also stores inv_var).
    bwd reads grad_output, output, weight, bias, inv_var; writes grad_input,
    grad_weight, grad_bias. Activation tensors dominate."""
    elem = _bytes_per_elem(dtype)
    activations = 4 * rows * cols * elem  # fwd in/out + bwd grad_in/out
    parameters = 6 * cols * elem  # weight, bias × (read + grad write) twice
    return activations + parameters


def _rms_norm_bytes(rows: int, cols: int, dtype: torch.dtype) -> int:
    elem = _bytes_per_elem(dtype)
    activations = 4 * rows * cols * elem
    parameters = 3 * cols * elem
    return activations + parameters


def _layer_norm_flops(rows: int, cols: int) -> int:
    """Approximate fwd+bwd FLOPs for LayerNorm.
    fwd: mean (1), variance (2), normalize (2), scale+shift (2) ≈ 7 per element.
    bwd: ~2x fwd."""
    return 21 * rows * cols


def _rms_norm_flops(rows: int, cols: int) -> int:
    """Same idea as LayerNorm but no mean subtraction or bias."""
    return 15 * rows * cols


def _layer_norm_cases(dtypes: tuple[torch.dtype, ...]) -> list[Case]:
    return [
        Case(
            name=case_name("layer_norm", shape, dtype),
            make_inputs=(lambda s=shape, d=dtype: _make_layer_norm_inputs(s[0], s[1], d)),
            expected_bytes=_layer_norm_bytes(shape[0], shape[1], dtype),
            expected_flops=_layer_norm_flops(shape[0], shape[1]),
            compute_dtype=dtype,
        )
        for dtype in dtypes
        for shape in _SHAPES
    ]


def _rms_norm_cases(dtypes: tuple[torch.dtype, ...]) -> list[Case]:
    return [
        Case(
            name=case_name("rms_norm", shape, dtype),
            make_inputs=(lambda s=shape, d=dtype: _make_rms_norm_inputs(s[0], s[1], d)),
            expected_bytes=_rms_norm_bytes(shape[0], shape[1], dtype),
            expected_flops=_rms_norm_flops(shape[0], shape[1]),
            compute_dtype=dtype,
        )
        for dtype in dtypes
        for shape in _SHAPES
    ]


# --------------------------------------------------------------------------- entry point


def run(verbose: bool = False, dtypes: tuple[torch.dtype, ...] | None = None) -> None:
    dtypes = tuple(dtypes) if dtypes else _DEFAULT_DTYPES
    run_benchmark("normalization: layer_norm", _layer_norm_cases(dtypes), _layer_norm_variants(), verbose=verbose)
    run_benchmark("normalization: rms_norm", _rms_norm_cases(dtypes), _rms_norm_variants(), verbose=verbose)


if __name__ == "__main__":
    run()
