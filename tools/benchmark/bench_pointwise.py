import torch

from fast_llm.functional.triton.pointwise import triton_add, triton_copy, triton_fill
from tools.benchmark.utils import bench_main, device, make_cases, standard_fwd_variants

# 4× steps so L2 → HBM and saturated-HBM regimes are visible.
_SIZES_NUMEL = [
    1 << 20,  # 1M     — 2 MiB bf16 (L2-resident on most GPUs)
    1 << 22,  # 4M     — 8 MiB bf16 (L2 boundary)
    1 << 24,  # 16M    — 32 MiB bf16 (HBM)
    1 << 26,  # 64M    — 128 MiB bf16 (HBM)
    1 << 28,  # 256M   — 512 MiB bf16 (large HBM, near-saturated)
]


def _copy_eager(input_: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return out.copy_(input_)


def _make_copy_inputs(numel: int, dtype: torch.dtype) -> dict:
    input_ = torch.randn(numel, dtype=dtype, device=device())
    return {"input_": input_, "out": torch.empty_like(input_)}


def _copy_bytes(numel: int, dtype: torch.dtype) -> int:
    return 2 * numel * dtype.itemsize


_COPY_VARIANTS = standard_fwd_variants(
    eager_function=_copy_eager,
    triton_function=triton_copy,
    unpack=lambda inputs: (inputs["input_"], inputs["out"]),
)


def _fill_eager(input_: torch.Tensor, value: float) -> torch.Tensor:
    return input_.fill_(value)


def _make_fill_inputs(numel: int, dtype: torch.dtype) -> dict:
    return {"input_": torch.empty(numel, dtype=dtype, device=device()), "value": 1.5}


def _fill_bytes(numel: int, dtype: torch.dtype) -> int:
    return numel * dtype.itemsize


_FILL_VARIANTS = standard_fwd_variants(
    eager_function=_fill_eager,
    triton_function=triton_fill,
    unpack=lambda inputs: (inputs["input_"], inputs["value"]),
)


def _add_eager(input_: torch.Tensor, other: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return torch.add(input_, other, out=out)


def _make_add_inputs(numel: int, dtype: torch.dtype) -> dict:
    return {
        "input_": torch.randn(numel, dtype=dtype, device=device()),
        "other": torch.randn(numel, dtype=dtype, device=device()),
        "out": torch.empty(numel, dtype=dtype, device=device()),
    }


def _add_bytes(numel: int, dtype: torch.dtype) -> int:
    return 3 * numel * dtype.itemsize


def _add_flops(numel: int) -> int:
    return numel


_ADD_VARIANTS = standard_fwd_variants(
    eager_function=_add_eager,
    triton_function=triton_add,
    unpack=lambda inputs: (inputs["input_"], inputs["other"], inputs["out"]),
)


def benchmarks(dtypes: tuple[torch.dtype, ...], shapes: list[int] | None = None) -> list[tuple[str, list, list]]:
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
