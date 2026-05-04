"""Rotary position embeddings. The Triton kernel is in-place; backward is an
identical rotation with conjugated frequencies, so only fwd is benchmarked."""

import dataclasses

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.rotary import triton_rotary_
from tools.benchmark.triton_kernels.runner import DtypedCase, Inputs, Variant
from tools.benchmark.triton_kernels.utils import dtype_short, standard_pytorch_variants

# (tokens, num_heads, head_size) — tokens = batch * seq_len
_SHAPES = [
    (4096, 32, 128),  # 7B/13B, 4K context
    (8192, 32, 128),  # 7B/13B, 8K context
    (4096, 64, 128),  # 70B / MoE, 4K context
    (4096, 8, 128),  # GQA key-value heads, 4K context
]


@dataclasses.dataclass
class RotaryCase(DtypedCase):
    tokens: int
    num_heads: int
    head_size: int
    dtype: torch.dtype

    @property
    def name(self) -> str:
        return f"({self.tokens}, {self.num_heads}, {self.head_size}) {dtype_short(self.dtype)}"

    @property
    def expected_bytes(self) -> int:
        # frequencies are float32, hence the extra 4 bytes per token×head_size.
        return (
            2 * self.tokens * self.num_heads * self.head_size * self.dtype.itemsize + self.tokens * self.head_size * 4
        )

    @property
    def expected_flops(self) -> int:
        # 6 FLOPs per (re, im) element pair: 4 muls + 2 add/sub.
        return 6 * self.tokens * self.num_heads * (self.head_size // 2)

    def make_inputs(self, device: torch.device) -> Inputs:
        rotary_dim = self.head_size // 2
        input_ = torch.randn(self.tokens, self.num_heads, self.head_size, dtype=self.dtype, device=device)
        return {
            "input_": input_,
            "work": input_.clone(),
            "frequencies": torch.randn(self.tokens, 2 * rotary_dim, dtype=torch.float32, device=device),
        }


def _rotary_eager(input_: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    rotary_dim = frequencies.shape[-1] // 2
    freq_re = frequencies[:, :rotary_dim].unsqueeze(1)
    freq_im = frequencies[:, rotary_dim:].unsqueeze(1)
    x_re, x_im = input_.chunk(2, dim=-1)
    out_re = x_re * freq_re - x_im * freq_im
    out_im = x_im * freq_re + x_re * freq_im
    return torch.cat([out_re, out_im], dim=-1)


def _rotary_variants() -> list[Variant]:
    variants = standard_pytorch_variants(_rotary_eager, input_keys=("input_", "frequencies"))
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=lambda inputs: {"output": triton_rotary_(inputs["work"], inputs["frequencies"])},
                reset_inputs=lambda inputs: inputs["work"].copy_(inputs["input_"]),
            )
        )
    return variants


def benchmarks(
    dtypes: tuple[torch.dtype, ...],
    shapes: list[tuple[int, int, int]] | None = None,
) -> list[tuple[str, list, list]]:
    shapes = shapes if shapes is not None else _SHAPES
    return [
        (
            "rotary",
            [RotaryCase(tokens=t, num_heads=h, head_size=hs, dtype=d) for d in dtypes for (t, h, hs) in shapes],
            _rotary_variants(),
        )
    ]
