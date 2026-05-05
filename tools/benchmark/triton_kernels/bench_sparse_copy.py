"""MoE dispatch and combine: scatter dense rows into sparse expert-grouped
buffers and gather them back with score weighting."""

import dataclasses

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.sparse_copy import (
    SparseMap,
    copy_dense_to_sparse_autograd,
    copy_sparse_to_dense_autograd,
    get_sparse_map,
)
from tools.benchmark.triton_kernels.runner import DtypedCase, Inputs, Variant
from tools.benchmark.triton_kernels.utils import (
    dtype_short,
    make_grad_reset,
    standard_pytorch_variants,
)

# (tokens, top_k, num_experts, hidden_size)
_SHAPES = [
    (4096, 2, 8, 4096),  # Mixtral-8x7B-like
    (4096, 2, 64, 4096),  # fine-grained MoE
    (4096, 2, 8, 8192),  # wide hidden
]


def _make_phantom_mask(sparse_map: SparseMap, device: torch.device) -> torch.Tensor:
    # True for within-expert padding rows and the static tail past expert_ends[-1];
    # used only in output_postprocess, never in the timed path.
    mask = torch.zeros(sparse_map.num_rows, 1, dtype=torch.bool, device=device)
    for expert in range(sparse_map.num_experts):
        pad_begin = int(sparse_map.expert_pad_begins[expert])
        pad_end = int(sparse_map.expert_ends[expert])
        if pad_end > pad_begin:
            mask[pad_begin:pad_end] = True
    tail_begin = int(sparse_map.expert_ends[-1])
    if sparse_map.num_rows > tail_begin:
        mask[tail_begin:] = True
    return mask


@dataclasses.dataclass
class _SparseCopyCase(DtypedCase):
    tokens: int
    top_k: int
    num_experts: int
    hidden: int
    dtype: torch.dtype

    @property
    def name(self) -> str:
        return f"({self.tokens}, {self.top_k}, {self.num_experts}, {self.hidden}) {dtype_short(self.dtype)}"

    @property
    def expected_bytes(self) -> int:
        # 2× (sparse + dense) hidden traffic; combine adds scores read/write.
        return 2 * (1 + self.top_k) * self.tokens * self.hidden * self.dtype.itemsize


@dataclasses.dataclass
class DispatchCase(_SparseCopyCase):
    def make_inputs(self, device: torch.device) -> Inputs:
        top_experts = torch.randint(0, self.num_experts, (self.tokens, self.top_k), device=device)
        sparse_map = get_sparse_map(top_experts, self.num_experts)
        return {
            "dense": torch.randn(self.tokens, self.hidden, dtype=self.dtype, device=device, requires_grad=True),
            "sparse_map": sparse_map,
            "phantom_mask": _make_phantom_mask(sparse_map, device),
            "backward_grad": torch.ones(sparse_map.num_rows, self.hidden, dtype=self.dtype, device=device),
        }


@dataclasses.dataclass
class CombineCase(_SparseCopyCase):
    @property
    def expected_bytes(self) -> int:
        # Adds scores read/write on top of the dispatch traffic.
        return super().expected_bytes + 4 * self.tokens * self.top_k * self.dtype.itemsize

    def make_inputs(self, device: torch.device) -> Inputs:
        top_experts = torch.randint(0, self.num_experts, (self.tokens, self.top_k), device=device)
        sparse_map = get_sparse_map(top_experts, self.num_experts)
        return {
            "sparse": torch.randn(
                sparse_map.num_rows, self.hidden, dtype=self.dtype, device=device, requires_grad=True
            ),
            "scores": torch.softmax(
                torch.randn(self.tokens, self.top_k, dtype=self.dtype, device=device), dim=-1
            ).requires_grad_(True),
            "sparse_map": sparse_map,
            "phantom_mask": _make_phantom_mask(sparse_map, device),
            "backward_grad": torch.ones(self.tokens, self.hidden, dtype=self.dtype, device=device),
        }


def _dispatch_pytorch(dense_input: torch.Tensor, sparse_map: SparseMap) -> torch.Tensor:
    out = dense_input.new_zeros(sparse_map.num_rows, dense_input.shape[1])
    sparse_rows = sparse_map.sparse_rows.long()
    for k in range(sparse_map.num_experts_per_token):
        out[sparse_rows[:, k]] = dense_input
    return out


def _dispatch_triton_fwd(inputs: Inputs) -> dict:
    return {"output": copy_dense_to_sparse_autograd(inputs["dense"], inputs["sparse_map"])}


def _dispatch_triton_fwd_bwd(inputs: Inputs) -> dict:
    output = copy_dense_to_sparse_autograd(inputs["dense"], inputs["sparse_map"])
    output.backward(inputs["backward_grad"])
    return {"output": output.detach(), "grad_dense": inputs["dense"].grad}


def _dispatch_postprocess(output: dict[str, torch.Tensor], inputs: Inputs) -> dict[str, torch.Tensor]:
    output["output"].masked_fill_(inputs["phantom_mask"], 0)
    return output


def _combine_pytorch(sparse_input: torch.Tensor, scores: torch.Tensor, sparse_map: SparseMap) -> torch.Tensor:
    out = sparse_input.new_zeros(sparse_map.num_rows_dense, sparse_input.shape[1])
    sparse_rows = sparse_map.sparse_rows.long()
    for k in range(sparse_map.num_experts_per_token):
        out = out + sparse_input[sparse_rows[:, k]] * scores[:, k : k + 1]
    return out


def _combine_triton_fwd(inputs: Inputs) -> dict:
    return {"output": copy_sparse_to_dense_autograd(inputs["sparse"], inputs["scores"], inputs["sparse_map"])}


def _combine_triton_fwd_bwd(inputs: Inputs) -> dict:
    output = copy_sparse_to_dense_autograd(inputs["sparse"], inputs["scores"], inputs["sparse_map"])
    output.backward(inputs["backward_grad"])
    return {
        "output": output.detach(),
        "grad_sparse": inputs["sparse"].grad,
        "grad_scores": inputs["scores"].grad,
    }


def _combine_postprocess(output: dict[str, torch.Tensor], inputs: Inputs) -> dict[str, torch.Tensor]:
    if "grad_sparse" in output:
        output["grad_sparse"].masked_fill_(inputs["phantom_mask"], 0)
    return output


def benchmarks(
    dtypes: tuple[torch.dtype, ...],
    shapes: list[tuple[int, int, int, int]] | None = None,
) -> list[tuple[str, list, list]]:
    shapes = shapes if shapes is not None else _SHAPES
    dispatch_variants = standard_pytorch_variants(
        _dispatch_pytorch,
        input_keys=("dense", "sparse_map"),
        grad_input_keys=("dense",),
        grad_output_key="backward_grad",
    )
    if TritonConfig.enabled():
        dispatch_variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_dispatch_triton_fwd,
                fwd_bwd=_dispatch_triton_fwd_bwd,
                output_postprocess=_dispatch_postprocess,
                reset_inputs=make_grad_reset(("dense",)),
            )
        )
    combine_variants = standard_pytorch_variants(
        _combine_pytorch,
        input_keys=("sparse", "scores", "sparse_map"),
        grad_input_keys=("sparse", "scores"),
        grad_output_key="backward_grad",
    )
    if TritonConfig.enabled():
        combine_variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_combine_triton_fwd,
                fwd_bwd=_combine_triton_fwd_bwd,
                output_postprocess=_combine_postprocess,
                reset_inputs=make_grad_reset(("sparse", "scores")),
            )
        )
    return [
        (
            "sparse_copy: dispatch",
            [
                DispatchCase(tokens=t, top_k=k, num_experts=e, hidden=h, dtype=d)
                for d in dtypes
                for (t, k, e, h) in shapes
            ],
            dispatch_variants,
        ),
        (
            "sparse_copy: combine",
            [
                CombineCase(tokens=t, top_k=k, num_experts=e, hidden=h, dtype=d)
                for d in dtypes
                for (t, k, e, h) in shapes
            ],
            combine_variants,
        ),
    ]
