"""
Benchmark MoE token dispatch and combine (sparse copy) kernels.

Two operations are benchmarked separately:

dispatch (dense → sparse):
  Each token is copied to top_k expert slots in the sparse buffer.
  copy_dense_to_sparse_autograd handles fwd+bwd (bwd = sparse-to-dense, no scores).

combine (sparse → dense):
  Expert outputs are gathered and weighted by routing scores back to token space.
  copy_sparse_to_dense_autograd handles fwd+bwd (bwd = dense-to-sparse + score grad).

Comparisons:
- pytorch_eager: index-based scatter/gather in compute dtype
- pytorch_compiled / pytorch_compiled_max: torch.compile of the above
- fast_llm_triton: copy_dense_to_sparse_autograd / copy_sparse_to_dense_autograd

Shapes: (tokens, top_k, num_experts, hidden_size) matching Mixtral-8x7B and fine-grained MoE.
The SparseMap is pre-computed once per case (routing structure, not data).
"""

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.sparse_copy import (
    SparseMap,
    copy_dense_to_sparse_autograd,
    copy_sparse_to_dense_autograd,
    get_sparse_map,
)
from tools.benchmark.runner import Variant
from tools.benchmark.utils import bench_main, device, make_cases, standard_fwd_bwd_pytorch_variants

# (tokens, top_k, num_experts, hidden_size)
_SHAPES = [
    (4096, 2, 8, 4096),  # Mixtral-8x7B-like
    (4096, 2, 64, 4096),  # fine-grained MoE
    (4096, 2, 8, 8192),  # wide hidden
]
_DEFAULT_DTYPES = (torch.bfloat16,)


def _make_sparse_map(tokens: int, top_k: int, num_experts: int) -> SparseMap:
    top_experts = torch.randint(0, num_experts, (tokens, top_k), device=device())
    return get_sparse_map(top_experts, num_experts)


def _make_phantom_mask(sparse_map: SparseMap) -> torch.Tensor:
    # Boolean mask shape (num_rows, 1): True for phantom rows (within-expert padding
    # and the static tail beyond expert_ends[-1]).  Precomputed once per case and
    # used with masked_fill_ in output_postprocess — never inside the timed path.
    mask = torch.zeros(sparse_map.num_rows, 1, dtype=torch.bool, device=device())
    for expert in range(sparse_map.num_experts):
        pad_begin = int(sparse_map.expert_pad_begins[expert])
        pad_end = int(sparse_map.expert_ends[expert])
        if pad_end > pad_begin:
            mask[pad_begin:pad_end] = True
    tail_begin = int(sparse_map.expert_ends[-1])
    if sparse_map.num_rows > tail_begin:
        mask[tail_begin:] = True
    return mask


def _make_dispatch_inputs(tokens: int, top_k: int, num_experts: int, hidden: int, dtype: torch.dtype) -> dict:
    sparse_map = _make_sparse_map(tokens, top_k, num_experts)
    return {
        "dense": torch.randn(tokens, hidden, dtype=dtype, device=device(), requires_grad=True),
        "sparse_map": sparse_map,
        "phantom_mask": _make_phantom_mask(sparse_map),
        "backward_grad": torch.ones(sparse_map.num_rows, hidden, dtype=dtype, device=device()),
    }


def _make_combine_inputs(tokens: int, top_k: int, num_experts: int, hidden: int, dtype: torch.dtype) -> dict:
    sparse_map = _make_sparse_map(tokens, top_k, num_experts)
    return {
        "sparse": torch.randn(sparse_map.num_rows, hidden, dtype=dtype, device=device(), requires_grad=True),
        "scores": torch.softmax(torch.randn(tokens, top_k, dtype=dtype, device=device()), dim=-1).requires_grad_(True),
        "sparse_map": sparse_map,
        "phantom_mask": _make_phantom_mask(sparse_map),
        "backward_grad": torch.ones(tokens, hidden, dtype=dtype, device=device()),
    }


# --------------------------------------------------------------------------- dispatch


def _dispatch_pytorch(dense_input: torch.Tensor, sparse_map: SparseMap) -> torch.Tensor:
    out = dense_input.new_zeros(sparse_map.num_rows, dense_input.shape[1])
    sparse_rows = sparse_map.sparse_rows.long()
    for k in range(sparse_map.num_experts_per_token):
        out[sparse_rows[:, k]] = dense_input
    return out


def _dispatch_triton_fwd(inputs: dict) -> dict:
    return {"output": copy_dense_to_sparse_autograd(inputs["dense"], inputs["sparse_map"])}


def _dispatch_triton_fwd_bwd(inputs: dict) -> dict:
    output = copy_dense_to_sparse_autograd(inputs["dense"], inputs["sparse_map"])
    output.backward(inputs["backward_grad"])
    return {"output": output.detach(), "grad_dense": inputs["dense"].grad}


def _dispatch_postprocess(output: dict[str, torch.Tensor], inputs: dict) -> dict[str, torch.Tensor]:
    output["output"].masked_fill_(inputs["phantom_mask"], 0)
    return output


def _dispatch_variants() -> list[Variant]:
    variants = standard_fwd_bwd_pytorch_variants(
        _dispatch_pytorch,
        input_keys=("dense", "sparse_map"),
        grad_input_keys=("dense",),
        grad_output_key="backward_grad",
    )
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_dispatch_triton_fwd,
                fwd_bwd=_dispatch_triton_fwd_bwd,
                output_postprocess=_dispatch_postprocess,
            )
        )
    return variants


# --------------------------------------------------------------------------- combine


def _combine_pytorch(sparse_input: torch.Tensor, scores: torch.Tensor, sparse_map: SparseMap) -> torch.Tensor:
    out = sparse_input.new_zeros(sparse_map.num_rows_dense, sparse_input.shape[1])
    sparse_rows = sparse_map.sparse_rows.long()
    for k in range(sparse_map.num_experts_per_token):
        out = out + sparse_input[sparse_rows[:, k]] * scores[:, k : k + 1]
    return out


def _combine_triton_fwd(inputs: dict) -> dict:
    return {"output": copy_sparse_to_dense_autograd(inputs["sparse"], inputs["scores"], inputs["sparse_map"])}


def _combine_triton_fwd_bwd(inputs: dict) -> dict:
    output = copy_sparse_to_dense_autograd(inputs["sparse"], inputs["scores"], inputs["sparse_map"])
    output.backward(inputs["backward_grad"])
    return {
        "output": output.detach(),
        "grad_sparse": inputs["sparse"].grad,
        "grad_scores": inputs["scores"].grad,
    }


def _combine_postprocess(output: dict[str, torch.Tensor], inputs: dict) -> dict[str, torch.Tensor]:
    if "grad_sparse" in output:
        output["grad_sparse"].masked_fill_(inputs["phantom_mask"], 0)
    return output


def _combine_variants() -> list[Variant]:
    variants = standard_fwd_bwd_pytorch_variants(
        _combine_pytorch,
        input_keys=("sparse", "scores", "sparse_map"),
        grad_input_keys=("sparse", "scores"),
        grad_output_key="backward_grad",
    )
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_combine_triton_fwd,
                fwd_bwd=_combine_triton_fwd_bwd,
                output_postprocess=_combine_postprocess,
            )
        )
    return variants


# --------------------------------------------------------------------------- bytes


def _dispatch_bytes(tokens: int, top_k: int, num_experts: int, hidden: int, dtype: torch.dtype) -> int:
    # fwd: read dense (tokens×h) + write sparse (top_k×tokens×h)
    # bwd: read sparse grad + write dense grad  → same traffic reversed
    return 2 * (1 + top_k) * tokens * hidden * dtype.itemsize


def _combine_bytes(tokens: int, top_k: int, num_experts: int, hidden: int, dtype: torch.dtype) -> int:
    sparse_rows = top_k * tokens
    # fwd: read sparse (sparse×h) + read scores (tokens×top_k) + write dense (tokens×h)
    # bwd: read dense grad + read scores + write sparse grad + write score grad
    return 2 * (sparse_rows + tokens) * hidden * dtype.itemsize + 4 * tokens * top_k * dtype.itemsize


# --------------------------------------------------------------------------- entry point


def benchmarks(
    dtypes: tuple[torch.dtype, ...] | None = None,
    shapes: list[tuple[int, int, int, int]] | None = None,
) -> list[tuple[str, list, list]]:
    dtypes = tuple(dtypes) if dtypes else _DEFAULT_DTYPES
    shapes = shapes if shapes is not None else _SHAPES
    return [
        (
            "sparse_copy: dispatch",
            make_cases("dispatch", dtypes, shapes, _make_dispatch_inputs, _dispatch_bytes),
            _dispatch_variants(),
        ),
        (
            "sparse_copy: combine",
            make_cases("combine", dtypes, shapes, _make_combine_inputs, _combine_bytes),
            _combine_variants(),
        ),
    ]


run = bench_main(benchmarks)


if __name__ == "__main__":
    run()
