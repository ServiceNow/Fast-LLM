import dataclasses

import torch

from fast_llm.functional.config import ActivationType, TritonConfig
from fast_llm.functional.triton.mlp import (
    torch_mlp_activation,
    triton_mlp_activation_autograd,
    triton_mlp_activation_forward,
)
from tools.benchmark.runner import Case, Inputs, Variant
from tools.benchmark.utils import bench_main, dtype_short, standard_fwd_bwd_pytorch_variants

# (tokens, ffn_dim) — input has shape (tokens, 2*ffn_dim) for gated.
_SHAPES = [
    (8192, 4096),  # 7B/13B
    (8192, 8192),
    (8192, 14336),  # 70B
    (4096, 28672),  # MoE up-projection
]
_ACTIVATION = ActivationType.silu


@dataclasses.dataclass
class MlpActivationCase(Case):
    tokens: int
    ffn_dim: int
    dtype: torch.dtype

    @property
    def name(self) -> str:
        return f"({self.tokens}, {self.ffn_dim}) {dtype_short(self.dtype)}"

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.dtype

    @property
    def expected_bytes(self) -> int:
        # fwd: 3*ffn_dim traffic; bwd: 5*ffn_dim. 8 elements/token total.
        return 8 * self.tokens * self.ffn_dim * self.dtype.itemsize

    @property
    def expected_flops(self) -> int:
        # gated silu: fwd ≈ 6 FLOPs/element, bwd ≈ 8 FLOPs/element.
        return 14 * self.tokens * self.ffn_dim

    def make_inputs(self, device: str) -> Inputs:
        return {
            "input": torch.randn(self.tokens, 2 * self.ffn_dim, dtype=self.dtype, device=device, requires_grad=True),
            "grad_output": torch.randn(self.tokens, self.ffn_dim, dtype=self.dtype, device=device),
            "gated": True,
            "activation_type": _ACTIVATION,
        }


def _triton_fwd(inputs: dict) -> dict:
    output, _ = triton_mlp_activation_forward(inputs["input"], inputs["gated"], inputs["activation_type"])
    return {"output": output}


def _triton_fwd_bwd(inputs: dict) -> dict:
    output = triton_mlp_activation_autograd(inputs["input"], inputs["gated"], inputs["activation_type"])
    output.backward(inputs["grad_output"])
    return {"output": output.detach(), "grad_input": inputs["input"].grad}


def benchmarks(
    dtypes: tuple[torch.dtype, ...],
    shapes: list[tuple[int, int]] | None = None,
) -> list[tuple[str, list, list]]:
    shapes = shapes if shapes is not None else _SHAPES
    variants = standard_fwd_bwd_pytorch_variants(
        torch_mlp_activation,
        input_keys=("input", "gated", "activation_type"),
        grad_input_keys=("input",),
        grad_output_key="grad_output",
    )
    if TritonConfig.enabled():
        variants.append(Variant(name="fast_llm_triton", fwd=_triton_fwd, fwd_bwd=_triton_fwd_bwd))
    return [
        (
            "mlp_activation (gated silu)",
            [MlpActivationCase(tokens=t, ffn_dim=f, dtype=d) for d in dtypes for (t, f) in shapes],
            variants,
        )
    ]


run = bench_main(benchmarks)
