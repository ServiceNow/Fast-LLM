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

from functools import partial

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


def _layer_norm_inputs_fp32(inputs: dict) -> dict:
    return {
        "input_": _to_fp32_input(inputs["input_"]),
        "weight": _to_fp32_param(inputs["weight"]),
        "bias": _to_fp32_param(inputs["bias"]),
        "grad_output": inputs["grad_output"].float(),
    }


def _rms_norm_inputs_fp32(inputs: dict) -> dict:
    return {
        "input_": _to_fp32_input(inputs["input_"]),
        "weight": _to_fp32_param(inputs["weight"]),
        "grad_output": inputs["grad_output"].float(),
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


def _run_layer_fwd(inputs: dict, fn) -> dict:
    return {"output": fn(inputs["input_"], inputs["weight"], inputs["bias"])}


def _run_layer_fwd_bwd(inputs: dict, fn) -> dict:
    output = fn(inputs["input_"], inputs["weight"], inputs["bias"])
    output.backward(inputs["grad_output"])
    return {
        "grad_input": inputs["input_"].grad,
        "grad_weight": _param_grad(inputs["weight"]),
        "grad_bias": _param_grad(inputs["bias"]),
    }


def _run_rms_fwd(inputs: dict, fn) -> dict:
    return {"output": fn(inputs["input_"], inputs["weight"])}


def _run_rms_fwd_bwd(inputs: dict, fn) -> dict:
    output = fn(inputs["input_"], inputs["weight"])
    output.backward(inputs["grad_output"])
    return {
        "grad_input": inputs["input_"].grad,
        "grad_weight": _param_grad(inputs["weight"]),
    }


# --------------------------------------------------------------------------- variants


def _layer_norm_variants() -> list[Variant]:
    variants = [
        Variant(
            name="fp32_reference",
            fwd=lambda inputs: _run_layer_fwd(_layer_norm_inputs_fp32(inputs), _layer_norm_eager),
            fwd_bwd=lambda inputs: _run_layer_fwd_bwd(_layer_norm_inputs_fp32(inputs), _layer_norm_eager),
            is_reference=True,
        ),
        Variant(
            name="pytorch_eager",
            fwd=lambda inputs: _run_layer_fwd(inputs, _layer_norm_eager),
            fwd_bwd=lambda inputs: _run_layer_fwd_bwd(inputs, _layer_norm_eager),
        ),
        Variant(
            name="pytorch_compiled",
            fwd=lambda inputs: _run_layer_fwd(inputs, _layer_compiled_default),
            fwd_bwd=lambda inputs: _run_layer_fwd_bwd(inputs, _layer_compiled_default),
        ),
        Variant(
            name="pytorch_compiled_max",
            fwd=lambda inputs: _run_layer_fwd(inputs, _layer_compiled_max),
            fwd_bwd=lambda inputs: _run_layer_fwd_bwd(inputs, _layer_compiled_max),
        ),
    ]
    if fused_normalization_available:
        variants.append(
            Variant(
                name="apex_fused",
                fwd=lambda inputs: _run_layer_fwd(inputs, _layer_norm_apex_fused),
                fwd_bwd=lambda inputs: _run_layer_fwd_bwd(inputs, _layer_norm_apex_fused),
            )
        )
    if fast_normalization_available:
        # apex_fast only supports widths in _PERSIST_LN_SIZES; all shapes in _SHAPES qualify.
        variants.append(
            Variant(
                name="apex_fast",
                fwd=lambda inputs: _run_layer_fwd(inputs, _layer_norm_apex_fast),
                fwd_bwd=lambda inputs: _run_layer_fwd_bwd(inputs, _layer_norm_apex_fast),
            )
        )
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=lambda inputs: _run_layer_fwd(inputs, _layer_norm_triton),
                fwd_bwd=lambda inputs: _run_layer_fwd_bwd(inputs, _layer_norm_triton),
            )
        )
    return variants


def _rms_norm_variants() -> list[Variant]:
    variants = [
        Variant(
            name="fp32_reference",
            fwd=lambda inputs: _run_rms_fwd(_rms_norm_inputs_fp32(inputs), _rms_norm_eager),
            fwd_bwd=lambda inputs: _run_rms_fwd_bwd(_rms_norm_inputs_fp32(inputs), _rms_norm_eager),
            is_reference=True,
        ),
        Variant(
            name="pytorch_eager",
            fwd=lambda inputs: _run_rms_fwd(inputs, _rms_norm_eager),
            fwd_bwd=lambda inputs: _run_rms_fwd_bwd(inputs, _rms_norm_eager),
        ),
        Variant(
            name="pytorch_compiled",
            fwd=lambda inputs: _run_rms_fwd(inputs, _rms_compiled_default),
            fwd_bwd=lambda inputs: _run_rms_fwd_bwd(inputs, _rms_compiled_default),
        ),
        Variant(
            name="pytorch_compiled_max",
            fwd=lambda inputs: _run_rms_fwd(inputs, _rms_compiled_max),
            fwd_bwd=lambda inputs: _run_rms_fwd_bwd(inputs, _rms_compiled_max),
        ),
    ]
    if fused_normalization_available:
        variants.append(
            Variant(
                name="apex_fused",
                fwd=lambda inputs: _run_rms_fwd(inputs, _rms_norm_apex_fused),
                fwd_bwd=lambda inputs: _run_rms_fwd_bwd(inputs, _rms_norm_apex_fused),
            )
        )
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=lambda inputs: _run_rms_fwd(inputs, _rms_norm_triton),
                fwd_bwd=lambda inputs: _run_rms_fwd_bwd(inputs, _rms_norm_triton),
            )
        )
    return variants


# --------------------------------------------------------------------------- cases


def _bytes_per_element(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _layer_norm_bytes(rows: int, cols: int, dtype: torch.dtype) -> int:
    """Approximate fwd+bwd memory traffic for LayerNorm.
    fwd reads input + weight + bias and writes output (also stores inv_var).
    bwd reads grad_output, output, weight, bias, inv_var; writes grad_input,
    grad_weight, grad_bias. Activation tensors dominate."""
    element_size = _bytes_per_element(dtype)
    activations = 4 * rows * cols * element_size  # fwd in/out + bwd grad_in/out
    parameters = 6 * cols * element_size  # weight, bias × (read + grad write) twice
    return activations + parameters


def _rms_norm_bytes(rows: int, cols: int, dtype: torch.dtype) -> int:
    element_size = _bytes_per_element(dtype)
    activations = 4 * rows * cols * element_size
    parameters = 3 * cols * element_size
    return activations + parameters


def _layer_norm_flops(rows: int, cols: int) -> int:
    """Approximate fwd+bwd FLOPs for LayerNorm.
    fwd: mean (1), variance (2), normalize (2), scale+shift (2) ≈ 7 per element.
    bwd: ~2x fwd."""
    return 21 * rows * cols


def _rms_norm_flops(rows: int, cols: int) -> int:
    """Same idea as LayerNorm but no mean subtraction or bias."""
    return 15 * rows * cols


def _layer_norm_cases(dtypes: tuple[torch.dtype, ...], shapes: list[tuple[int, int]] | None = None) -> list[Case]:
    shapes = shapes if shapes is not None else _SHAPES
    return [
        Case(
            name=case_name("layer_norm", shape, dtype),
            make_inputs=partial(_make_layer_norm_inputs, shape[0], shape[1], dtype),
            expected_bytes=_layer_norm_bytes(shape[0], shape[1], dtype),
            expected_flops=_layer_norm_flops(shape[0], shape[1]),
            compute_dtype=dtype,
        )
        for dtype in dtypes
        for shape in shapes
    ]


def _rms_norm_cases(dtypes: tuple[torch.dtype, ...], shapes: list[tuple[int, int]] | None = None) -> list[Case]:
    shapes = shapes if shapes is not None else _SHAPES
    return [
        Case(
            name=case_name("rms_norm", shape, dtype),
            make_inputs=partial(_make_rms_norm_inputs, shape[0], shape[1], dtype),
            expected_bytes=_rms_norm_bytes(shape[0], shape[1], dtype),
            expected_flops=_rms_norm_flops(shape[0], shape[1]),
            compute_dtype=dtype,
        )
        for dtype in dtypes
        for shape in shapes
    ]


# --------------------------------------------------------------------------- entry point


def benchmarks(
    dtypes: tuple[torch.dtype, ...] | None = None,
    shapes: list[tuple[int, int]] | None = None,
) -> list[tuple[str, list, list]]:
    dtypes = tuple(dtypes) if dtypes else _DEFAULT_DTYPES
    return [
        ("normalization: layer_norm", _layer_norm_cases(dtypes, shapes), _layer_norm_variants()),
        ("normalization: rms_norm", _rms_norm_cases(dtypes, shapes), _rms_norm_variants()),
    ]


def run(
    verbose: bool = False,
    dtypes: tuple[torch.dtype, ...] | None = None,
    shapes: list[tuple[int, int]] | None = None,
    warmup_ms: float = 25.0,
    rep_ms: float = 100.0,
    min_reps: int = 5,
) -> None:
    for name, cases, variants in benchmarks(dtypes, shapes):
        run_benchmark(name, cases, variants, verbose=verbose, warmup_ms=warmup_ms, rep_ms=rep_ms, min_reps=min_reps)


if __name__ == "__main__":
    run()
