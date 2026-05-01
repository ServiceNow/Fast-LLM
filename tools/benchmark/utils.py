import dataclasses
from collections.abc import Callable
from typing import Any

import torch

from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.functional.config import TritonConfig
from tools.benchmark.runner import Inputs, Variant, run_benchmark

DEFAULT_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16,)


def dtype_short(dtype: torch.dtype) -> str:
    return DataType.from_torch(dtype).short


def bench_main(benchmarks_fn: Callable) -> Callable:
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


@dataclasses.dataclass(kw_only=True)
class PytorchVariant(Variant):
    """Variant that calls a pytorch function on inputs picked by key. Used for
    eager, torch.compile, and apex variants — each instance differs in `function`
    while sharing the dispatch logic."""

    function: Callable
    input_keys: tuple[str, ...]
    grad_input_keys: tuple[str, ...] = ()
    grad_output_key: str | None = None
    output_key: str = "output"

    def __post_init__(self) -> None:
        # Wire the inherited `fwd`/`fwd_bwd` callable fields to bound methods
        # so subclasses can override the methods without touching the fields.
        self.fwd = self._fwd
        self.fwd_bwd = self._fwd_bwd

    def _fwd(self, inputs: Inputs) -> dict:
        return {self.output_key: self.function(*(inputs[k] for k in self.input_keys))}

    def _fwd_bwd(self, inputs: Inputs) -> dict:
        output = self.function(*(inputs[k] for k in self.input_keys))
        if self.grad_output_key is None:
            output.backward()
        else:
            output.backward(inputs[self.grad_output_key])
        result = {self.output_key: output.detach()}
        for key in self.grad_input_keys:
            result[f"grad_{key}"] = inputs[key].grad
        return result


@dataclasses.dataclass(kw_only=True)
class Fp32ReferenceVariant(PytorchVariant):
    """Reference variant: upcasts every floating-point input to fp32 before
    running the eager pytorch function. Re-attaches `requires_grad=True` on
    `grad_input_keys` so backward sees a leaf tensor."""

    name: str = "fp32_reference"
    is_reference: bool = True

    def _fwd(self, inputs: Inputs) -> dict:
        return super()._fwd(self._to_fp32(inputs))

    def _fwd_bwd(self, inputs: Inputs) -> dict:
        return super()._fwd_bwd(self._to_fp32(inputs))

    def _to_fp32(self, inputs: Inputs) -> Inputs:
        result = dict(inputs)
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                float_value = value.float().detach()
                result[key] = float_value.requires_grad_(True) if key in self.grad_input_keys else float_value
        return result


@dataclasses.dataclass(kw_only=True)
class FwdOnlyPytorchVariant(Variant):
    """Forward-only variant: calls a pytorch function with positional args
    extracted via `unpack`. Used by bench_pointwise where there's no backward."""

    function: Callable
    unpack: Callable[[Inputs], tuple]

    def __post_init__(self) -> None:
        self.fwd = self._fwd

    def _fwd(self, inputs: Inputs) -> Any:
        return self.function(*self.unpack(inputs))


@dataclasses.dataclass(kw_only=True)
class Fp32FwdOnlyReferenceVariant(FwdOnlyPytorchVariant):
    name: str = "fp32_reference"
    is_reference: bool = True

    def _fwd(self, inputs: Inputs) -> Any:
        args = tuple(
            arg.float() if isinstance(arg, torch.Tensor) and arg.is_floating_point() else arg
            for arg in self.unpack(inputs)
        )
        return self.function(*args)


def standard_fwd_variants(
    eager_function: Callable,
    triton_function: Callable | None,
    unpack: Callable[[Inputs], tuple],
) -> list[Variant]:
    """fp32_reference, pytorch_eager, pytorch_compiled, pytorch_compiled_max,
    and (if `TritonConfig.enabled()`) fast_llm_triton."""
    variants: list[Variant] = [
        Fp32FwdOnlyReferenceVariant(function=eager_function, unpack=unpack),
        FwdOnlyPytorchVariant(name="pytorch_eager", function=eager_function, unpack=unpack),
        FwdOnlyPytorchVariant(
            name="pytorch_compiled",
            function=torch.compile(eager_function, mode="default", dynamic=False),
            unpack=unpack,
        ),
        FwdOnlyPytorchVariant(
            name="pytorch_compiled_max",
            function=torch.compile(eager_function, mode="max-autotune-no-cudagraphs", dynamic=False),
            unpack=unpack,
        ),
    ]
    if triton_function is not None and TritonConfig.enabled():
        variants.append(
            FwdOnlyPytorchVariant(
                name="fast_llm_triton",
                function=lambda *args: triton_function(*args, use_triton=True),
                unpack=unpack,
            )
        )
    return variants


def make_grad_reset(keys: tuple[str, ...]) -> Callable[[Inputs], None]:
    """Reset autograd `.grad` to None for the given input keys between reps.
    `.backward()` accumulates into `.grad` on rep 2+, biasing fwd_bwd timing
    via an extra read+write of the full grad tensor. Also resets
    `param_grad_is_zero=True` on tensors with a `grad_buffer` (Fast-LLM
    convention) so the next backward writes fresh instead of accumulating."""

    def reset(inputs: Inputs) -> None:
        for key in keys:
            tensor = inputs[key]
            tensor.grad = None
            if hasattr(tensor, "grad_buffer"):
                tensor.param_grad_is_zero = True

    return reset


def standard_fwd_bwd_pytorch_variants(
    eager_function: Callable,
    input_keys: tuple[str, ...],
    grad_input_keys: tuple[str, ...],
    *,
    grad_output_key: str | None = None,
    output_key: str = "output",
    reset_inputs: Callable[[Inputs], None] | None = None,
    extra_functions: dict[str, Callable] | None = None,
    eager_name: str = "pytorch_eager",
    enable_max_autotune: bool = True,
) -> list[Variant]:
    """fp32_reference + <eager_name> + pytorch_compiled + [pytorch_compiled_max]
    + `extra_functions`. Triton variants are appended by the caller (so each
    bench file owns its triton wiring explicitly). `reset_inputs` defaults to
    clearing `.grad` on `grad_input_keys` between reps."""
    if reset_inputs is None:
        reset_inputs = make_grad_reset(grad_input_keys)
    common = {
        "input_keys": input_keys,
        "grad_input_keys": grad_input_keys,
        "grad_output_key": grad_output_key,
        "output_key": output_key,
        "reset_inputs": reset_inputs,
    }
    variants: list[Variant] = [
        Fp32ReferenceVariant(function=eager_function, **common),
        PytorchVariant(name=eager_name, function=eager_function, **common),
        PytorchVariant(
            name="pytorch_compiled",
            function=torch.compile(eager_function, mode="default", dynamic=False),
            **common,
        ),
    ]
    if enable_max_autotune:
        variants.append(
            PytorchVariant(
                name="pytorch_compiled_max",
                function=torch.compile(eager_function, mode="max-autotune-no-cudagraphs", dynamic=False),
                **common,
            )
        )
    for name, function in (extra_functions or {}).items():
        variants.append(PytorchVariant(name=name, function=function, **common))
    return variants
