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
from tools.benchmark.runner import Case, Variant, run_benchmark
from tools.benchmark.utils import case_name, device

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
    for e in range(sparse_map.num_experts):
        pad_begin = int(sparse_map.expert_pad_begins[e])
        pad_end = int(sparse_map.expert_ends[e])
        if pad_end > pad_begin:
            mask[pad_begin:pad_end] = True
    tail_begin = int(sparse_map.expert_ends[-1])
    if sparse_map.num_rows > tail_begin:
        mask[tail_begin:] = True
    return mask


def _make_dispatch_inputs(tokens: int, top_k: int, num_experts: int, hidden: int, dtype: torch.dtype) -> dict:
    sparse_map = _make_sparse_map(tokens, top_k, num_experts)
    return {
        "dense_input": torch.randn(tokens, hidden, dtype=dtype, device=device(), requires_grad=True),
        "sparse_map": sparse_map,
        "phantom_mask": _make_phantom_mask(sparse_map),
        "backward_grad": torch.ones(sparse_map.num_rows, hidden, dtype=dtype, device=device()),
    }


def _make_combine_inputs(tokens: int, top_k: int, num_experts: int, hidden: int, dtype: torch.dtype) -> dict:
    sparse_map = _make_sparse_map(tokens, top_k, num_experts)
    return {
        "sparse_input": torch.randn(sparse_map.num_rows, hidden, dtype=dtype, device=device(), requires_grad=True),
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


_dispatch_compiled_default = torch.compile(_dispatch_pytorch, mode="default", dynamic=False)
_dispatch_compiled_max = torch.compile(_dispatch_pytorch, mode="max-autotune-no-cudagraphs", dynamic=False)


def _run_dispatch_fwd(inp: dict, fn) -> dict:
    return {"output": fn(inp["dense_input"], inp["sparse_map"])}


def _run_dispatch_fwd_bwd(inp: dict, fn) -> dict:
    output = fn(inp["dense_input"], inp["sparse_map"])
    output.backward(inp["backward_grad"])
    return {"output": output.detach(), "grad_dense": inp["dense_input"].grad}


def _run_dispatch_fwd_triton(inp: dict) -> dict:
    return {"output": copy_dense_to_sparse_autograd(inp["dense_input"], inp["sparse_map"])}


def _run_dispatch_fwd_bwd_triton(inp: dict) -> dict:
    output = copy_dense_to_sparse_autograd(inp["dense_input"], inp["sparse_map"])
    output.backward(inp["backward_grad"])
    return {"output": output.detach(), "grad_dense": inp["dense_input"].grad}


def _dispatch_postprocess(out: dict[str, torch.Tensor], inp: dict) -> dict[str, torch.Tensor]:
    out["output"].masked_fill_(inp["phantom_mask"], 0)
    return out


def _dispatch_variants() -> list[Variant]:
    variants = [
        Variant(
            name="pytorch_eager",
            fwd=lambda inp: _run_dispatch_fwd(inp, _dispatch_pytorch),
            fwd_bwd=lambda inp: _run_dispatch_fwd_bwd(inp, _dispatch_pytorch),
            is_reference=True,
        ),
        Variant(
            name="pytorch_compiled",
            fwd=lambda inp: _run_dispatch_fwd(inp, _dispatch_compiled_default),
            fwd_bwd=lambda inp: _run_dispatch_fwd_bwd(inp, _dispatch_compiled_default),
        ),
        Variant(
            name="pytorch_compiled_max",
            fwd=lambda inp: _run_dispatch_fwd(inp, _dispatch_compiled_max),
            fwd_bwd=lambda inp: _run_dispatch_fwd_bwd(inp, _dispatch_compiled_max),
        ),
    ]
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_run_dispatch_fwd_triton,
                fwd_bwd=_run_dispatch_fwd_bwd_triton,
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


_combine_compiled_default = torch.compile(_combine_pytorch, mode="default", dynamic=False)
_combine_compiled_max = torch.compile(_combine_pytorch, mode="max-autotune-no-cudagraphs", dynamic=False)


def _run_combine_fwd(inp: dict, fn) -> dict:
    return {"output": fn(inp["sparse_input"], inp["scores"], inp["sparse_map"])}


def _run_combine_fwd_bwd(inp: dict, fn) -> dict:
    output = fn(inp["sparse_input"], inp["scores"], inp["sparse_map"])
    output.backward(inp["backward_grad"])
    return {
        "output": output.detach(),
        "grad_sparse": inp["sparse_input"].grad,
        "grad_scores": inp["scores"].grad,
    }


def _run_combine_fwd_triton(inp: dict) -> dict:
    return {"output": copy_sparse_to_dense_autograd(inp["sparse_input"], inp["scores"], inp["sparse_map"])}


def _run_combine_fwd_bwd_triton(inp: dict) -> dict:
    output = copy_sparse_to_dense_autograd(inp["sparse_input"], inp["scores"], inp["sparse_map"])
    output.backward(inp["backward_grad"])
    return {
        "output": output.detach(),
        "grad_sparse": inp["sparse_input"].grad,
        "grad_scores": inp["scores"].grad,
    }


def _combine_postprocess(out: dict[str, torch.Tensor], inp: dict) -> dict[str, torch.Tensor]:
    if "grad_sparse" in out:
        out["grad_sparse"].masked_fill_(inp["phantom_mask"], 0)
    return out


def _combine_variants() -> list[Variant]:
    variants = [
        Variant(
            name="pytorch_eager",
            fwd=lambda inp: _run_combine_fwd(inp, _combine_pytorch),
            fwd_bwd=lambda inp: _run_combine_fwd_bwd(inp, _combine_pytorch),
            is_reference=True,
        ),
        Variant(
            name="pytorch_compiled",
            fwd=lambda inp: _run_combine_fwd(inp, _combine_compiled_default),
            fwd_bwd=lambda inp: _run_combine_fwd_bwd(inp, _combine_compiled_default),
        ),
        Variant(
            name="pytorch_compiled_max",
            fwd=lambda inp: _run_combine_fwd(inp, _combine_compiled_max),
            fwd_bwd=lambda inp: _run_combine_fwd_bwd(inp, _combine_compiled_max),
        ),
    ]
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_run_combine_fwd_triton,
                fwd_bwd=_run_combine_fwd_bwd_triton,
                output_postprocess=_combine_postprocess,
            )
        )
    return variants


# --------------------------------------------------------------------------- cases / bytes


def _bytes_per_elem(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _dispatch_bytes(tokens: int, top_k: int, hidden: int, dtype: torch.dtype) -> int:
    elem = _bytes_per_elem(dtype)
    # fwd: read dense (tokens×h) + write sparse (top_k×tokens×h)
    # bwd: read sparse grad + write dense grad  → same traffic reversed
    return 2 * (1 + top_k) * tokens * hidden * elem


def _combine_bytes(tokens: int, top_k: int, hidden: int, dtype: torch.dtype) -> int:
    elem = _bytes_per_elem(dtype)
    sparse_rows = top_k * tokens
    # fwd: read sparse (sparse×h) + read scores (tokens×top_k) + write dense (tokens×h)
    # bwd: read dense grad + read scores + write sparse grad + write score grad
    return 2 * (sparse_rows + tokens) * hidden * elem + 4 * tokens * top_k * elem


def _dispatch_cases(dtypes: tuple[torch.dtype, ...]) -> list[Case]:
    return [
        Case(
            name=case_name("dispatch", (tokens, top_k, num_experts, hidden), dtype),
            make_inputs=lambda t=tokens, k=top_k, n=num_experts, h=hidden, d=dtype: _make_dispatch_inputs(
                t, k, n, h, d
            ),
            expected_bytes=_dispatch_bytes(tokens, top_k, hidden, dtype),
            expected_flops=0,
            compute_dtype=dtype,
        )
        for dtype in dtypes
        for tokens, top_k, num_experts, hidden in _SHAPES
    ]


def _combine_cases(dtypes: tuple[torch.dtype, ...]) -> list[Case]:
    return [
        Case(
            name=case_name("combine", (tokens, top_k, num_experts, hidden), dtype),
            make_inputs=lambda t=tokens, k=top_k, n=num_experts, h=hidden, d=dtype: _make_combine_inputs(
                t, k, n, h, d
            ),
            expected_bytes=_combine_bytes(tokens, top_k, hidden, dtype),
            expected_flops=0,
            compute_dtype=dtype,
        )
        for dtype in dtypes
        for tokens, top_k, num_experts, hidden in _SHAPES
    ]


# --------------------------------------------------------------------------- entry point


def run(verbose: bool = False, dtypes: tuple[torch.dtype, ...] | None = None) -> None:
    dtypes = tuple(dtypes) if dtypes else _DEFAULT_DTYPES
    run_benchmark("sparse_copy: dispatch", _dispatch_cases(dtypes), _dispatch_variants(), verbose=verbose)
    run_benchmark("sparse_copy: combine", _combine_cases(dtypes), _combine_variants(), verbose=verbose)


if __name__ == "__main__":
    run()
