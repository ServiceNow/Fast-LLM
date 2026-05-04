"""MoE expert-grouped matmul: layer-1 (output_sparse, up-proj) and layer-2
(input_inner_sparse, down-proj). Compares the Triton sparse kernel against a
per-expert pytorch loop reference."""

import dataclasses

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.sparse_copy import SparseMap, get_sparse_map
from fast_llm.functional.triton.sparse_linear import InputSparseLinear, OutputSparseLinear
from tools.benchmark.triton_kernels.runner import DtypedCase, Inputs, Variant
from tools.benchmark.triton_kernels.utils import (
    dtype_short,
    make_grad_reset,
    standard_pytorch_variants,
)

# (tokens, top_k, num_experts, hidden, ffn_per_expert)
_SHAPES = [
    (4096, 2, 8, 4096, 14336),  # Mixtral-8x7B: 8 experts, ffn=14336
    (4096, 2, 64, 4096, 1792),  # fine-grained MoE: 64 experts, same total capacity
    (4096, 2, 8, 8192, 28672),  # large hidden / wide FFN
]

# Triton autotuning warmup needs to run only once per shape; make_inputs is
# called many times per case (per variant, per fwd/fwd_bwd/memory pass).
# Process-local — a fresh interpreter starts with empty caches, which is
# desired (autotuning state may differ across processes).
_WarmupKey = tuple[int, int, int, int, int, torch.dtype]
_output_sparse_warmed_up: set[_WarmupKey] = set()
_input_inner_sparse_warmed_up: set[_WarmupKey] = set()


def _mask_padded_rows(candidate: dict[str, torch.Tensor], inputs: Inputs) -> dict[str, torch.Tensor]:
    # Two regions in the kernel's forward output and grad_lhs are by-design garbage:
    # per-expert padding [pad_begin, expert_end) and phantom rows past expert_ends[-1].
    # The loop reference produces zeros there; mask the kernel output to match so
    # rel_rms reflects only the real rows. grad_rhs already excludes padding.
    sparse_map = inputs["sparse_map"]
    pad_begins = sparse_map.expert_pad_begins.tolist()
    pad_ends = sparse_map.expert_ends.tolist()
    last_expert_end = pad_ends[-1]
    masked = dict(candidate)
    for key in ("output", "grad_lhs"):
        if key not in masked:
            continue
        clone = masked[key].clone()
        for begin, end in zip(pad_begins, pad_ends, strict=True):
            if end > begin:
                clone[begin:end] = 0
        if clone.shape[0] > last_expert_end:
            clone[last_expert_end:] = 0
        masked[key] = clone
    return masked


@dataclasses.dataclass
class _SparseLinearCase(DtypedCase):
    tokens: int
    top_k: int
    num_experts: int
    hidden: int
    ffn_per_expert: int
    dtype: torch.dtype

    @property
    def name(self) -> str:
        return (
            f"({self.tokens}, {self.top_k}, {self.num_experts}, "
            f"{self.hidden}, {self.ffn_per_expert}) {dtype_short(self.dtype)}"
        )

    @property
    def expected_bytes(self) -> int:
        # Approximation: 3× lhs + 3× rhs + 2× output traffic across fwd+bwd.
        sparse_tokens = self.tokens * self.top_k
        lhs_bytes = sparse_tokens * self.hidden * self.dtype.itemsize
        rhs_bytes = self.hidden * self.ffn_per_expert * self.num_experts * self.dtype.itemsize
        output_bytes = sparse_tokens * self.ffn_per_expert * self.dtype.itemsize
        return 3 * lhs_bytes + 3 * rhs_bytes + 2 * output_bytes

    @property
    def expected_flops(self) -> int:
        # 3 matmuls (fwd: lhs@rhs, bwd_lhs: grad@rhs.T, bwd_rhs: lhs.T@grad).
        return 3 * 2 * self.tokens * self.top_k * self.hidden * self.ffn_per_expert

    def _make_sparse_map(self, device: torch.device) -> SparseMap:
        top_experts = torch.randint(0, self.num_experts, (self.tokens, self.top_k), device=device)
        return get_sparse_map(top_experts, self.num_experts)


@dataclasses.dataclass
class OutputSparseCase(_SparseLinearCase):
    def make_inputs(self, device: torch.device) -> Inputs:
        sparse_map = self._make_sparse_map(device)
        lhs_data = torch.randn(sparse_map.num_rows, self.hidden, dtype=self.dtype, device=device)
        rhs_data = torch.randn(self.hidden, self.ffn_per_expert * self.num_experts, dtype=self.dtype, device=device)
        backward_grad = torch.ones(sparse_map.num_rows, self.ffn_per_expert, dtype=self.dtype, device=device)
        warmup_key = (self.tokens, self.top_k, self.num_experts, self.hidden, self.ffn_per_expert, self.dtype)
        if TritonConfig.enabled() and warmup_key not in _output_sparse_warmed_up:
            warmup_lhs = lhs_data.detach().requires_grad_(True)
            warmup_rhs = rhs_data.detach().requires_grad_(True)
            warmup_out = OutputSparseLinear.apply(warmup_lhs, warmup_rhs, sparse_map)
            warmup_out.backward(backward_grad)
            del warmup_lhs, warmup_rhs, warmup_out
            _output_sparse_warmed_up.add(warmup_key)
        return {
            "lhs": lhs_data.requires_grad_(True),
            "rhs": rhs_data.requires_grad_(True),
            "sparse_map": sparse_map,
            "backward_grad": backward_grad,
        }


@dataclasses.dataclass
class InputInnerSparseCase(_SparseLinearCase):
    def make_inputs(self, device: torch.device) -> Inputs:
        sparse_map = self._make_sparse_map(device)
        lhs_data = torch.randn(sparse_map.num_rows, self.ffn_per_expert, dtype=self.dtype, device=device)
        rhs_data = torch.randn(self.ffn_per_expert * self.num_experts, self.hidden, dtype=self.dtype, device=device)
        backward_grad = torch.ones(sparse_map.num_rows, self.hidden, dtype=self.dtype, device=device)
        warmup_key = (self.tokens, self.top_k, self.num_experts, self.hidden, self.ffn_per_expert, self.dtype)
        if TritonConfig.enabled() and warmup_key not in _input_inner_sparse_warmed_up:
            warmup_lhs = lhs_data.detach().requires_grad_(True)
            warmup_rhs = rhs_data.detach().requires_grad_(True)
            warmup_out = InputSparseLinear.apply(warmup_lhs, warmup_rhs, sparse_map)
            warmup_out.backward(backward_grad)
            del warmup_lhs, warmup_rhs, warmup_out
            _input_inner_sparse_warmed_up.add(warmup_key)
        return {
            "lhs": lhs_data.requires_grad_(True),
            "rhs": rhs_data.requires_grad_(True),
            "sparse_map": sparse_map,
            "backward_grad": backward_grad,
        }


def _output_sparse_loop(lhs: torch.Tensor, rhs: torch.Tensor, sparse_map: SparseMap) -> torch.Tensor:
    ffn_per_expert = rhs.shape[1] // sparse_map.num_experts
    out = lhs.new_zeros(sparse_map.num_rows, ffn_per_expert)
    for expert in range(sparse_map.num_experts):
        row_begin = int(sparse_map.expert_ends[expert - 1]) if expert > 0 else 0
        row_end = int(sparse_map.expert_pad_begins[expert])
        if row_end > row_begin:
            col_begin = expert * ffn_per_expert
            out[row_begin:row_end] = lhs[row_begin:row_end] @ rhs[:, col_begin : col_begin + ffn_per_expert]
    return out


def _output_sparse_triton_fwd(inputs: Inputs) -> dict:
    return {"output": OutputSparseLinear.apply(inputs["lhs"], inputs["rhs"], inputs["sparse_map"])}


def _output_sparse_triton_fwd_bwd(inputs: Inputs) -> dict:
    output = OutputSparseLinear.apply(inputs["lhs"], inputs["rhs"], inputs["sparse_map"])
    output.backward(inputs["backward_grad"])
    return {"output": output.detach(), "grad_lhs": inputs["lhs"].grad, "grad_rhs": inputs["rhs"].grad}


def _input_inner_sparse_loop(lhs: torch.Tensor, rhs: torch.Tensor, sparse_map: SparseMap) -> torch.Tensor:
    ffn_per_expert = rhs.shape[0] // sparse_map.num_experts
    out = lhs.new_zeros(sparse_map.num_rows, rhs.shape[1])
    for expert in range(sparse_map.num_experts):
        row_begin = int(sparse_map.expert_ends[expert - 1]) if expert > 0 else 0
        row_end = int(sparse_map.expert_pad_begins[expert])
        if row_end > row_begin:
            inner_begin = expert * ffn_per_expert
            out[row_begin:row_end] = lhs[row_begin:row_end] @ rhs[inner_begin : inner_begin + ffn_per_expert]
    return out


def _input_inner_sparse_triton_fwd(inputs: Inputs) -> dict:
    return {"output": InputSparseLinear.apply(inputs["lhs"], inputs["rhs"], inputs["sparse_map"])}


def _input_inner_sparse_triton_fwd_bwd(inputs: Inputs) -> dict:
    output = InputSparseLinear.apply(inputs["lhs"], inputs["rhs"], inputs["sparse_map"])
    output.backward(inputs["backward_grad"])
    return {"output": output.detach(), "grad_lhs": inputs["lhs"].grad, "grad_rhs": inputs["rhs"].grad}


def benchmarks(
    dtypes: tuple[torch.dtype, ...],
    shapes: list[tuple[int, int, int, int, int]] | None = None,
) -> list[tuple[str, list, list]]:
    shapes = shapes if shapes is not None else _SHAPES
    output_sparse_variants = standard_pytorch_variants(
        _output_sparse_loop,
        input_keys=("lhs", "rhs", "sparse_map"),
        grad_input_keys=("lhs", "rhs"),
        grad_output_key="backward_grad",
        eager_name="pytorch_loop",
        enable_max_autotune=False,
    )
    if TritonConfig.enabled():
        output_sparse_variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_output_sparse_triton_fwd,
                fwd_bwd=_output_sparse_triton_fwd_bwd,
                output_postprocess=_mask_padded_rows,
                reset_inputs=make_grad_reset(("lhs", "rhs")),
            )
        )
    input_inner_sparse_variants = standard_pytorch_variants(
        _input_inner_sparse_loop,
        input_keys=("lhs", "rhs", "sparse_map"),
        grad_input_keys=("lhs", "rhs"),
        grad_output_key="backward_grad",
        eager_name="pytorch_loop",
        enable_max_autotune=False,
    )
    if TritonConfig.enabled():
        input_inner_sparse_variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_input_inner_sparse_triton_fwd,
                fwd_bwd=_input_inner_sparse_triton_fwd_bwd,
                output_postprocess=_mask_padded_rows,
                reset_inputs=make_grad_reset(("lhs", "rhs")),
            )
        )
    return [
        (
            "sparse_linear: output_sparse (layer 1 / up-proj)",
            [
                OutputSparseCase(tokens=t, top_k=k, num_experts=e, hidden=h, ffn_per_expert=f, dtype=d)
                for d in dtypes
                for (t, k, e, h, f) in shapes
            ],
            output_sparse_variants,
        ),
        (
            "sparse_linear: input_inner_sparse (layer 2 / down-proj)",
            [
                InputInnerSparseCase(tokens=t, top_k=k, num_experts=e, hidden=h, ffn_per_expert=f, dtype=d)
                for d in dtypes
                for (t, k, e, h, f) in shapes
            ],
            input_inner_sparse_variants,
        ),
    ]
