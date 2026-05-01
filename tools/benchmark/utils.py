"""Helpers shared by the `bench_*.py` modules — case construction, variant
builders for the canonical fp32_reference + pytorch_eager + pytorch_compiled
chunk, and the `run = bench_main(benchmarks)` glue."""

from collections.abc import Callable
from functools import partial

import torch

from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.functional.config import TritonConfig
from tools.benchmark.runner import Case, Inputs, Variant, VariantFn, run_benchmark

DEFAULT_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16,)


def format_size(n: int) -> str:
    for unit, factor in (("Gi", 1 << 30), ("Mi", 1 << 20), ("Ki", 1 << 10)):
        if n >= factor and n % factor == 0:
            return f"{n // factor} {unit}"
    return str(n)


def format_shape(shape: tuple[int, ...]) -> str:
    joined = ", ".join(format_size(n) for n in shape)
    return f"({joined},)" if len(shape) == 1 else f"({joined})"


def case_name(kernel: str, shape: tuple[int, ...], dtype: torch.dtype) -> str:
    return f"[{kernel}] {format_shape(shape)} {DataType.from_torch(dtype).short}"


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_cases(
    kernel_name: str,
    dtypes: tuple[torch.dtype, ...],
    shapes: list,
    make_inputs: Callable,
    bytes_fn: Callable | None = None,
    flops_fn: Callable | None = None,
) -> list[Case]:
    """Cross-product of `dtypes × shapes`. Each shape may be a tuple or scalar;
    tuples are unpacked positionally into `make_inputs`, `bytes_fn`, `flops_fn`."""
    cases = []
    for dtype in dtypes:
        for shape in shapes:
            shape_tuple = shape if isinstance(shape, tuple) else (shape,)
            cases.append(
                Case(
                    name=case_name(kernel_name, shape_tuple, dtype),
                    make_inputs=partial(make_inputs, *shape_tuple, dtype),
                    expected_bytes=bytes_fn(*shape_tuple, dtype) if bytes_fn else None,
                    expected_flops=flops_fn(*shape_tuple) if flops_fn else None,
                    compute_dtype=dtype,
                )
            )
    return cases


def bench_main(benchmarks_fn: Callable) -> Callable:
    """Build the `run()` callable each `bench_*.py` exports for `__main__.py` to dispatch to."""

    def run(
        verbose: bool = False,
        dtypes: tuple[torch.dtype, ...] | None = None,
        shapes: list | None = None,
        warmup_ms: float = 25.0,
        rep_ms: float = 100.0,
        min_reps: int = 5,
    ) -> None:
        for name, cases, variants in benchmarks_fn(dtypes or DEFAULT_DTYPES, shapes):
            run_benchmark(
                name, cases, variants, verbose=verbose, warmup_ms=warmup_ms, rep_ms=rep_ms, min_reps=min_reps
            )

    return run


def standard_fwd_variants(
    eager_function: Callable,
    triton_function: Callable | None,
    unpack: Callable[[Inputs], tuple],
) -> list[Variant]:
    """Forward-only variant set: fp32_reference, pytorch_eager, pytorch_compiled,
    pytorch_compiled_max, and (if `TritonConfig.enabled()`) fast_llm_triton."""

    def fp32_unpack(inputs: Inputs) -> tuple:
        return tuple(
            arg.float() if isinstance(arg, torch.Tensor) and arg.is_floating_point() else arg for arg in unpack(inputs)
        )

    compiled_default = torch.compile(eager_function, mode="default", dynamic=False)
    compiled_max = torch.compile(eager_function, mode="max-autotune-no-cudagraphs", dynamic=False)

    variants = [
        Variant(
            name="fp32_reference",
            fwd=lambda inputs: eager_function(*fp32_unpack(inputs)),
            is_reference=True,
        ),
        Variant(name="pytorch_eager", fwd=lambda inputs: eager_function(*unpack(inputs))),
        Variant(name="pytorch_compiled", fwd=lambda inputs: compiled_default(*unpack(inputs))),
        Variant(name="pytorch_compiled_max", fwd=lambda inputs: compiled_max(*unpack(inputs))),
    ]
    if triton_function is not None and TritonConfig.enabled():
        variants.append(
            Variant(name="fast_llm_triton", fwd=lambda inputs: triton_function(*unpack(inputs), use_triton=True))
        )
    return variants


def _run_pytorch_fwd(
    inputs: Inputs,
    function: Callable,
    input_keys: tuple[str, ...],
    output_key: str,
) -> dict:
    return {output_key: function(*(inputs[key] for key in input_keys))}


def _run_pytorch_fwd_bwd(
    inputs: Inputs,
    function: Callable,
    input_keys: tuple[str, ...],
    grad_input_keys: tuple[str, ...],
    grad_output_key: str | None,
    output_key: str,
) -> dict:
    output = function(*(inputs[key] for key in input_keys))
    if grad_output_key is None:
        output.backward()
    else:
        output.backward(inputs[grad_output_key])
    result = {output_key: output.detach()}
    for key in grad_input_keys:
        result[f"grad_{key}"] = inputs[key].grad
    return result


def _to_fp32_inputs(inputs: Inputs, grad_input_keys: tuple[str, ...]) -> Inputs:
    result = dict(inputs)
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            float_value = value.float().detach()
            result[key] = float_value.requires_grad_(True) if key in grad_input_keys else float_value
    return result


def standard_fwd_bwd_pytorch_variants(
    eager_function: Callable,
    input_keys: tuple[str, ...],
    grad_input_keys: tuple[str, ...],
    *,
    grad_output_key: str | None = None,
    output_key: str = "output",
    reset_inputs: Callable[[Inputs], None] | None = None,
    extra_functions: dict[str, Callable] | None = None,
    triton_fwd: VariantFn | None = None,
    triton_fwd_bwd: VariantFn | None = None,
    triton_name: str = "fast_llm_triton",
    triton_output_postprocess: Callable[[dict, Inputs], dict] | None = None,
    eager_name: str = "pytorch_eager",
    enable_max_autotune: bool = True,
) -> list[Variant]:
    """Forward+backward variant set for kernels driven by a dict-style input.

    Generates fp32_reference, <eager_name>, pytorch_compiled, [pytorch_compiled_max,]
    plus any callables in `extra_functions` (variant name = dict key). When
    `triton_fwd`/`triton_fwd_bwd` are given and `TritonConfig.enabled()`, a
    `<triton_name>` variant is appended; the callables receive the raw inputs dict.

    `eager_function(*[inputs[key] for key in input_keys])` computes the forward output.
    `grad_input_keys` lists input keys whose `.grad` is collected as `grad_<key>`.
    `grad_output_key=None` triggers a scalar-loss `output.backward()`.
    The fp32 reference upcasts every floating-point tensor in the input dict, re-attaching
    `requires_grad=True` on `grad_input_keys`.
    """
    fwd_kwargs = {"input_keys": input_keys, "output_key": output_key}
    fwd_bwd_kwargs = {
        "input_keys": input_keys,
        "grad_input_keys": grad_input_keys,
        "grad_output_key": grad_output_key,
        "output_key": output_key,
    }

    def variant(name: str, function: Callable) -> Variant:
        return Variant(
            name=name,
            fwd=partial(_run_pytorch_fwd, function=function, **fwd_kwargs),
            fwd_bwd=partial(_run_pytorch_fwd_bwd, function=function, **fwd_bwd_kwargs),
            reset_inputs=reset_inputs,
        )

    compiled_default = torch.compile(eager_function, mode="default", dynamic=False)
    variants = [
        Variant(
            name="fp32_reference",
            fwd=lambda inputs: _run_pytorch_fwd(
                _to_fp32_inputs(inputs, grad_input_keys), eager_function, **fwd_kwargs
            ),
            fwd_bwd=lambda inputs: _run_pytorch_fwd_bwd(
                _to_fp32_inputs(inputs, grad_input_keys), eager_function, **fwd_bwd_kwargs
            ),
            is_reference=True,
        ),
        variant(eager_name, eager_function),
        variant("pytorch_compiled", compiled_default),
    ]
    if enable_max_autotune:
        compiled_max = torch.compile(eager_function, mode="max-autotune-no-cudagraphs", dynamic=False)
        variants.append(variant("pytorch_compiled_max", compiled_max))
    for name, function in (extra_functions or {}).items():
        variants.append(variant(name, function))
    if (triton_fwd is not None or triton_fwd_bwd is not None) and TritonConfig.enabled():
        variants.append(
            Variant(
                name=triton_name,
                fwd=triton_fwd,
                fwd_bwd=triton_fwd_bwd,
                reset_inputs=reset_inputs,
                output_postprocess=triton_output_postprocess,
            )
        )
    return variants
