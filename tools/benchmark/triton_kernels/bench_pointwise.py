"""Memory-bound pointwise ops (copy, fill, add). Sweeps numel from L2-resident
to HBM-saturated to surface the bandwidth-bound regime where Triton wins or
loses against PyTorch."""

import dataclasses
import typing

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.pointwise import triton_add, triton_copy, triton_fill
from tools.benchmark.triton_kernels.runner import DtypedCase, Inputs, Variant
from tools.benchmark.triton_kernels.utils import dtype_short, standard_pytorch_variants

# 4× steps so L2 → HBM and saturated-HBM regimes are visible.
_SIZES_NUMEL = [
    1 << 20,  # 1M     — 2 MiB bf16 (L2-resident on most GPUs)
    1 << 22,  # 4M     — 8 MiB bf16 (L2 boundary)
    1 << 24,  # 16M    — 32 MiB bf16 (HBM)
    1 << 26,  # 64M    — 128 MiB bf16 (HBM)
    1 << 28,  # 256M   — 512 MiB bf16 (large HBM, near-saturated)
]


@dataclasses.dataclass
class _PointwiseCase(DtypedCase):
    numel: int
    dtype: torch.dtype
    # Bytes traffic = bytes_factor × numel × dtype.itemsize.
    bytes_factor: typing.ClassVar[int]

    @property
    def name(self) -> str:
        return f"({self.numel},) {dtype_short(self.dtype)}"

    @property
    def expected_bytes(self) -> int:
        return self.bytes_factor * self.numel * self.dtype.itemsize


@dataclasses.dataclass
class CopyCase(_PointwiseCase):
    bytes_factor = 2

    def make_inputs(self, device: torch.device) -> Inputs:
        input_ = torch.randn(self.numel, dtype=self.dtype, device=device)
        return {"input_": input_, "out": torch.empty_like(input_)}


@dataclasses.dataclass
class FillCase(_PointwiseCase):
    bytes_factor = 1

    def make_inputs(self, device: torch.device) -> Inputs:
        return {"input_": torch.empty(self.numel, dtype=self.dtype, device=device), "value": 1.5}


@dataclasses.dataclass
class AddCase(_PointwiseCase):
    bytes_factor = 3

    @property
    def expected_flops(self) -> int:
        return self.numel

    def make_inputs(self, device: torch.device) -> Inputs:
        return {
            "input_": torch.randn(self.numel, dtype=self.dtype, device=device),
            "other": torch.randn(self.numel, dtype=self.dtype, device=device),
            "out": torch.empty(self.numel, dtype=self.dtype, device=device),
        }


def _copy_eager(input_: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return out.copy_(input_)


def _fill_eager(input_: torch.Tensor, value: float) -> torch.Tensor:
    return input_.fill_(value)


def _add_eager(input_: torch.Tensor, other: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return torch.add(input_, other, out=out)


def _make_variants(
    eager_function: typing.Callable, triton_function: typing.Callable, input_keys: tuple[str, ...]
) -> list[Variant]:
    variants = standard_pytorch_variants(eager_function, input_keys=input_keys)
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=lambda inputs: {"output": triton_function(*(inputs[k] for k in input_keys), use_triton=True)},
            )
        )
    return variants


def benchmarks(dtypes: tuple[torch.dtype, ...], shapes: list[int] | None = None) -> list[tuple[str, list, list]]:
    shapes = shapes if shapes is not None else _SIZES_NUMEL
    return [
        (
            "pointwise: copy",
            [CopyCase(numel=n, dtype=d) for d in dtypes for n in shapes],
            _make_variants(_copy_eager, triton_copy, ("input_", "out")),
        ),
        (
            "pointwise: fill",
            [FillCase(numel=n, dtype=d) for d in dtypes for n in shapes],
            _make_variants(_fill_eager, triton_fill, ("input_", "value")),
        ),
        (
            "pointwise: add",
            [AddCase(numel=n, dtype=d) for d in dtypes for n in shapes],
            _make_variants(_add_eager, triton_add, ("input_", "other", "out")),
        ),
    ]
