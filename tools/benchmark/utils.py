"""
Convenience helpers for writing kernel benchmark files. Reduces the boilerplate
of building cases and variants so each `bench_*.py` can stay focused on
kernel-specific logic (input construction, expected_bytes/flops, special variants).
"""

from collections.abc import Callable

import torch

from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.functional.config import TritonConfig
from tools.benchmark.runner import Inputs, Variant

# --------------------------------------------------------------------------- formatting


def format_size(n: int) -> str:
    """Format an int with the largest binary prefix that divides it exactly: 1048576 → '1 Mi'."""
    for unit, factor in (("Gi", 1 << 30), ("Mi", 1 << 20), ("Ki", 1 << 10)):
        if n >= factor and n % factor == 0:
            return f"{n // factor} {unit}"
    return str(n)


def format_shape(shape: tuple[int, ...]) -> str:
    """Format a shape tuple with human-readable sizes per dim: (16777216,) → '(16 Mi,)'."""
    joined = ", ".join(format_size(n) for n in shape)
    return f"({joined},)" if len(shape) == 1 else f"({joined})"


def case_name(kernel: str, shape: tuple[int, ...], dtype: torch.dtype) -> str:
    """Build the standard case header: `[copy] (16 Mi,) bf16`."""
    return f"[{kernel}] {format_shape(shape)} {DataType.from_torch(dtype).short}"


def device() -> str:
    """The device benchmarks should target. Falls back to CPU when CUDA is missing
    so non-Triton variants can still run for local smoke testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- variant builders


def standard_fwd_variants(
    eager_fn: Callable,
    triton_fn: Callable | None,
    unpack: Callable[[Inputs], tuple],
) -> list[Variant]:
    """Build the canonical 5-variant set for a forward-only kernel.

    Generates: fp32_reference, pytorch_eager, pytorch_compiled, pytorch_compiled_max,
    and (if `TritonConfig.enabled()`) fast_llm_triton.

    `eager_fn` is the plain PyTorch implementation taking positional tensor args.
    `triton_fn` is the Fast-LLM Triton wrapper; pass `None` if the kernel has no
    Triton variant. Both are invoked with `unpack(inputs)` unpacked positionally;
    `triton_fn` is called with an extra `use_triton=True` kwarg.

    The fp32 reference upcasts every floating-point tensor in the unpacked
    arguments to fp32 (non-tensor / non-float arguments are passed through).
    """

    def _fp32_unpack(inputs: Inputs) -> tuple:
        return tuple(
            arg.float() if isinstance(arg, torch.Tensor) and arg.is_floating_point() else arg for arg in unpack(inputs)
        )

    compiled_default = torch.compile(eager_fn, mode="default", dynamic=False)
    compiled_max = torch.compile(eager_fn, mode="max-autotune-no-cudagraphs", dynamic=False)

    variants = [
        Variant(
            name="fp32_reference",
            fwd=lambda inp: eager_fn(*_fp32_unpack(inp)),
            is_reference=True,
        ),
        Variant(name="pytorch_eager", fwd=lambda inp: eager_fn(*unpack(inp))),
        Variant(name="pytorch_compiled", fwd=lambda inp: compiled_default(*unpack(inp))),
        Variant(name="pytorch_compiled_max", fwd=lambda inp: compiled_max(*unpack(inp))),
    ]
    if triton_fn is not None and TritonConfig.enabled():
        variants.append(Variant(name="fast_llm_triton", fwd=lambda inp: triton_fn(*unpack(inp), use_triton=True)))
    return variants
