"""
Benchmark pointwise kernels: copy, fill, add.

These kernels are pure bandwidth-bound: runtime is dominated by reading inputs
and writing outputs, so GB/s and %-of-peak-BW are the headline metrics. The
Triton kernels live in `fast_llm/functional/triton/pointwise.py` and are
documented as being ~2x faster than the PyTorch equivalent on A100.
"""

import torch

from fast_llm.functional.triton.pointwise import triton_add, triton_copy, triton_fill
from tools.benchmark.runner import Case, run_benchmark
from tools.benchmark.utils import case_name, device, standard_fwd_variants

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
    out = torch.empty_like(input_)
    return {"input_": input_, "out": out}


def _copy_cases(dtypes: tuple[torch.dtype, ...]) -> list[Case]:
    return [
        Case(
            name=case_name("copy", (numel,), dtype),
            make_inputs=(lambda n=numel, d=dtype: _make_copy_inputs(n, d)),
            # Read input + write output.
            expected_bytes=2 * numel * torch.tensor([], dtype=dtype).element_size(),
        )
        for dtype in dtypes
        for numel in _SIZES_NUMEL
    ]


_COPY_VARIANTS = standard_fwd_variants(
    eager_fn=_copy_eager,
    triton_fn=triton_copy,
    unpack=lambda inp: (inp["input_"], inp["out"]),
)


# --------------------------------------------------------------------------- fill


def _fill_eager(input_: torch.Tensor, value: float) -> torch.Tensor:
    return input_.fill_(value)


def _make_fill_inputs(numel: int, dtype: torch.dtype) -> dict:
    return {"input_": torch.empty(numel, dtype=dtype, device=device()), "value": 1.5}


def _fill_cases(dtypes: tuple[torch.dtype, ...]) -> list[Case]:
    return [
        Case(
            name=case_name("fill", (numel,), dtype),
            make_inputs=(lambda n=numel, d=dtype: _make_fill_inputs(n, d)),
            # Write only.
            expected_bytes=numel * torch.tensor([], dtype=dtype).element_size(),
        )
        for dtype in dtypes
        for numel in _SIZES_NUMEL
    ]


_FILL_VARIANTS = standard_fwd_variants(
    eager_fn=_fill_eager,
    triton_fn=triton_fill,
    unpack=lambda inp: (inp["input_"], inp["value"]),
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


def _add_cases(dtypes: tuple[torch.dtype, ...]) -> list[Case]:
    return [
        Case(
            name=case_name("add", (numel,), dtype),
            make_inputs=(lambda n=numel, d=dtype: _make_add_inputs(n, d)),
            # Read 2 inputs + write 1 output.
            expected_bytes=3 * numel * torch.tensor([], dtype=dtype).element_size(),
            # One fp add per element.
            expected_flops=numel,
            compute_dtype=dtype,
        )
        for dtype in dtypes
        for numel in _SIZES_NUMEL
    ]


_ADD_VARIANTS = standard_fwd_variants(
    eager_fn=_add_eager,
    triton_fn=triton_add,
    unpack=lambda inp: (inp["input_"], inp["other"], inp["out"]),
)


# --------------------------------------------------------------------------- entry point


def run(verbose: bool = False, dtypes: tuple[torch.dtype, ...] | None = None) -> None:
    dtypes = tuple(dtypes) if dtypes else _DEFAULT_DTYPES
    run_benchmark("pointwise: copy", _copy_cases(dtypes), _COPY_VARIANTS, verbose=verbose)
    run_benchmark("pointwise: fill", _fill_cases(dtypes), _FILL_VARIANTS, verbose=verbose)
    run_benchmark("pointwise: add", _add_cases(dtypes), _ADD_VARIANTS, verbose=verbose)


if __name__ == "__main__":
    run()
