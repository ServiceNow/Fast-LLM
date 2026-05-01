"""
Benchmark pointwise kernels: copy, fill, add.

These kernels are pure bandwidth-bound: runtime is dominated by reading inputs
and writing outputs, so GB/s and %-of-peak-BW are the headline metrics. The
Triton kernels live in `fast_llm/functional/triton/pointwise.py` and are
documented as being ~2x faster than the PyTorch equivalent on A100.
"""

import torch

from fast_llm.functional.triton.pointwise import triton_add, triton_copy, triton_fill
from tools.benchmark.utils import bench_main, device, make_cases, standard_fwd_variants

# Sizes span from L2-resident to comfortably HBM-bound, in 4× steps so the
# regime transitions (L2 → HBM, mid-HBM → saturated-HBM) are visible.
_SIZES_NUMEL = [
    1 << 20,  # 1M     — 2 MiB bf16 (L2-resident on most GPUs)
    1 << 22,  # 4M     — 8 MiB bf16 (L2 boundary)
    1 << 24,  # 16M    — 32 MiB bf16 (HBM)
    1 << 26,  # 64M    — 128 MiB bf16 (HBM)
    1 << 28,  # 256M   — 512 MiB bf16 (large HBM, near-saturated)
]
_DEFAULT_DTYPES = (torch.bfloat16,)


# --------------------------------------------------------------------------- copy


def _copy_eager(input_: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return out.copy_(input_)


def _make_copy_inputs(numel: int, dtype: torch.dtype) -> dict:
    input_ = torch.randn(numel, dtype=dtype, device=device())
    return {"input_": input_, "out": torch.empty_like(input_)}


def _copy_bytes(numel: int, dtype: torch.dtype) -> int:
    # Read input + write output.
    return 2 * numel * dtype.itemsize


_COPY_VARIANTS = standard_fwd_variants(
    eager_function=_copy_eager,
    triton_function=triton_copy,
    unpack=lambda inputs: (inputs["input_"], inputs["out"]),
)


# --------------------------------------------------------------------------- fill


def _fill_eager(input_: torch.Tensor, value: float) -> torch.Tensor:
    return input_.fill_(value)


def _make_fill_inputs(numel: int, dtype: torch.dtype) -> dict:
    return {"input_": torch.empty(numel, dtype=dtype, device=device()), "value": 1.5}


def _fill_bytes(numel: int, dtype: torch.dtype) -> int:
    # Write only.
    return numel * dtype.itemsize


_FILL_VARIANTS = standard_fwd_variants(
    eager_function=_fill_eager,
    triton_function=triton_fill,
    unpack=lambda inputs: (inputs["input_"], inputs["value"]),
)


# --------------------------------------------------------------------------- add


def _add_eager(input_: torch.Tensor, other: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return torch.add(input_, other, out=out)


def _make_add_inputs(numel: int, dtype: torch.dtype) -> dict:
    return {
        "input_": torch.randn(numel, dtype=dtype, device=device()),
        "other": torch.randn(numel, dtype=dtype, device=device()),
        "out": torch.empty(numel, dtype=dtype, device=device()),
    }


def _add_bytes(numel: int, dtype: torch.dtype) -> int:
    # Read 2 inputs + write 1 output.
    return 3 * numel * dtype.itemsize


def _add_flops(numel: int) -> int:
    # One fp add per element.
    return numel


_ADD_VARIANTS = standard_fwd_variants(
    eager_function=_add_eager,
    triton_function=triton_add,
    unpack=lambda inputs: (inputs["input_"], inputs["other"], inputs["out"]),
)


# --------------------------------------------------------------------------- entry point


def benchmarks(
    dtypes: tuple[torch.dtype, ...] | None = None,
    shapes: list[int] | None = None,
) -> list[tuple[str, list, list]]:
    dtypes = tuple(dtypes) if dtypes else _DEFAULT_DTYPES
    shapes = shapes if shapes is not None else _SIZES_NUMEL
    return [
        ("pointwise: copy", make_cases("copy", dtypes, shapes, _make_copy_inputs, _copy_bytes), _COPY_VARIANTS),
        ("pointwise: fill", make_cases("fill", dtypes, shapes, _make_fill_inputs, _fill_bytes), _FILL_VARIANTS),
        (
            "pointwise: add",
            make_cases("add", dtypes, shapes, _make_add_inputs, _add_bytes, _add_flops),
            _ADD_VARIANTS,
        ),
    ]


run = bench_main(benchmarks)


if __name__ == "__main__":
    run()
