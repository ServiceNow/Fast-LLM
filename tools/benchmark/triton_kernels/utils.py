"""Variant builders shared across bench files: pytorch eager / compiled /
fp32-reference factories and a `make_grad_reset` helper for fwd_bwd resets."""

import typing

import torch

from fast_llm.engine.config_utils.data_type import DataType
from tools.benchmark.triton_kernels.runner import Inputs, Variant


def dtype_short(dtype: torch.dtype) -> str:
    return DataType.from_torch(dtype).short


def make_grad_reset(keys: tuple[str, ...]) -> typing.Callable[[Inputs], None]:
    """Reset autograd `.grad` to None for the given input keys between reps.
    `.backward()` accumulates into `.grad` on rep 2+, biasing fwd_bwd timing
    via an extra read+write of the full grad tensor. For tensors with a
    `grad_buffer` (Fast-LLM convention), set `param_grad_is_zero=True` instead
    of zeroing the buffer — Triton kernels that respect the flag will overwrite
    it on the next backward, so an explicit zero would be wasted bandwidth."""

    def reset(inputs: Inputs) -> None:
        for key in keys:
            tensor = inputs[key]
            tensor.grad = None
            if hasattr(tensor, "grad_buffer"):
                tensor.param_grad_is_zero = True

    return reset


def _to_fp32(inputs: Inputs, grad_input_keys: tuple[str, ...]) -> Inputs:
    """Upcast every floating-point input to fp32. Re-attach `requires_grad=True`
    on `grad_input_keys` so backward sees a leaf tensor."""
    result = dict(inputs)
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            float_value = value.float().detach()
            result[key] = float_value.requires_grad_(True) if key in grad_input_keys else float_value
    return result


def pytorch_variant(
    name: str,
    function: typing.Callable,
    input_keys: tuple[str, ...],
    *,
    grad_input_keys: tuple[str, ...] = (),
    grad_output_key: str | None = None,
    output_key: str = "output",
    is_reference: bool = False,
    convert_to_fp32: bool = False,
    reset_inputs: typing.Callable[[Inputs], typing.Any] | None = None,
) -> Variant:
    """Build a Variant that calls `function(*[inputs[k] for k in input_keys])`.
    Backward is wired up when `grad_input_keys` is non-empty; if `convert_to_fp32`
    is set, all floating-point inputs are upcast to fp32 first (used by the
    reference variant)."""

    def _prepare(inputs: Inputs) -> Inputs:
        return _to_fp32(inputs, grad_input_keys) if convert_to_fp32 else inputs

    def fwd(inputs: Inputs) -> dict:
        prepared = _prepare(inputs)
        return {output_key: function(*(prepared[k] for k in input_keys))}

    def fwd_bwd(inputs: Inputs) -> dict:
        prepared = _prepare(inputs)
        output = function(*(prepared[k] for k in input_keys))
        if grad_output_key is None:
            output.backward()
        else:
            output.backward(prepared[grad_output_key])
        result = {output_key: output.detach()}
        for key in grad_input_keys:
            result[f"grad_{key}"] = prepared[key].grad
        return result

    return Variant(
        name=name,
        fwd=fwd,
        fwd_bwd=fwd_bwd if grad_input_keys else None,
        is_reference=is_reference,
        reset_inputs=reset_inputs,
    )


def standard_pytorch_variants(
    eager_function: typing.Callable,
    input_keys: tuple[str, ...],
    *,
    grad_input_keys: tuple[str, ...] = (),
    grad_output_key: str | None = None,
    output_key: str = "output",
    reset_inputs: typing.Callable[[Inputs], None] | None = None,
    extra_functions: dict[str, typing.Callable] | None = None,
    eager_name: str = "pytorch_eager",
    enable_max_autotune: bool = True,
) -> list[Variant]:
    """fp32_reference + <eager_name> + pytorch_compiled + [pytorch_compiled_max]
    + `extra_functions`. When `grad_input_keys` is empty, only fwd is wired up.
    Triton variants are appended by the caller (so each bench file owns its
    triton wiring explicitly). For fwd_bwd cases, `reset_inputs` defaults to
    clearing `.grad` on `grad_input_keys` between reps."""
    if grad_input_keys and reset_inputs is None:
        reset_inputs = make_grad_reset(grad_input_keys)
    common = {
        "input_keys": input_keys,
        "grad_input_keys": grad_input_keys,
        "grad_output_key": grad_output_key,
        "output_key": output_key,
        "reset_inputs": reset_inputs,
    }
    variants: list[Variant] = [
        pytorch_variant("fp32_reference", eager_function, is_reference=True, convert_to_fp32=True, **common),
        pytorch_variant(eager_name, eager_function, **common),
        pytorch_variant("pytorch_compiled", torch.compile(eager_function, mode="default", dynamic=False), **common),
    ]
    if enable_max_autotune:
        variants.append(
            pytorch_variant(
                "pytorch_compiled_max",
                torch.compile(eager_function, mode="max-autotune-no-cudagraphs", dynamic=False),
                **common,
            )
        )
    for name, function in (extra_functions or {}).items():
        variants.append(pytorch_variant(name, function, **common))
    return variants
