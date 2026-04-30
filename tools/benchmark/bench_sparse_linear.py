"""
Benchmark MoE sparse grouped GEMM kernels.

Two operations are benchmarked, corresponding to the two linear layers in a MoE FFN:

output_sparse  (layer 1 / up-proj):
  out[i, :] = lhs[i, :] @ rhs[:, expert(i)*ffn_per_expert : (expert(i)+1)*ffn_per_expert]
  lhs: (sparse_tokens, hidden),  rhs: (hidden, ffn_per_expert × num_experts)
  Each token's output columns come from its assigned expert's slice of rhs.
  OutputSparseLinear.apply handles fwd+bwd.

input_inner_sparse  (layer 2 / down-proj):
  out[i, :] = lhs[i, :] @ rhs[expert(i)*ffn_per_expert : (expert(i)+1)*ffn_per_expert, :]
  lhs: (sparse_tokens, ffn_per_expert),  rhs: (ffn_per_expert × num_experts, hidden)
  Each token's inner dimension comes from its assigned expert's slice of rhs.
  InputSparseLinear.apply handles fwd+bwd.

Comparisons:
- pytorch_loop: loop over experts with torch.mm per expert (the obvious PyTorch approach)
- pytorch_compiled: torch.compile of the loop
- fast_llm_triton: OutputSparseLinear / InputSparseLinear autograd functions

Shapes: (tokens, top_k, num_experts, hidden, ffn_per_expert) matching MoE FFN configs.
"""

from functools import partial

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.sparse_copy import SparseMap, get_sparse_map
from fast_llm.functional.triton.sparse_linear import InputSparseLinear, OutputSparseLinear
from tools.benchmark.runner import Case, Variant, run_benchmark
from tools.benchmark.utils import case_name, device

# (tokens, top_k, num_experts, hidden, ffn_per_expert)
_SHAPES = [
    (4096, 2, 8, 4096, 14336),  # Mixtral-8x7B: 8 experts, ffn=14336
    (4096, 2, 64, 4096, 1792),  # fine-grained MoE: 64 experts, same total capacity
    (4096, 2, 8, 8192, 28672),  # large hidden / wide FFN
]
_DEFAULT_DTYPES = (torch.bfloat16,)

# Triton autotuning warmup only needs to run once per shape. make_inputs is
# called multiple times per case (per variant, per fwd/fwd_bwd/memory pass),
# so cache which shapes have already been warmed up.
_output_sparse_warmed_up: set[tuple] = set()
_input_inner_sparse_warmed_up: set[tuple] = set()


def _make_sparse_map(tokens: int, top_k: int, num_experts: int) -> SparseMap:
    top_experts = torch.randint(0, num_experts, (tokens, top_k), device=device())
    return get_sparse_map(top_experts, num_experts)


def _zero_padded_rows(tensor: torch.Tensor, sparse_map: SparseMap) -> torch.Tensor:
    for e in range(sparse_map.num_experts):
        pad_start = int(sparse_map.expert_pad_begins[e])
        pad_end = int(sparse_map.expert_ends[e])
        if pad_end > pad_start:
            tensor[pad_start:pad_end] = 0
    return tensor


def _make_output_sparse_inputs(
    tokens: int, top_k: int, num_experts: int, hidden: int, ffn_per_expert: int, dtype: torch.dtype
) -> dict:
    sparse_map = _make_sparse_map(tokens, top_k, num_experts)
    lhs_data = _zero_padded_rows(torch.randn(sparse_map.num_rows, hidden, dtype=dtype, device=device()), sparse_map)
    rhs_data = torch.randn(hidden, ffn_per_expert * num_experts, dtype=dtype, device=device())
    backward_grad = _zero_padded_rows(
        torch.ones(sparse_map.num_rows, ffn_per_expert, dtype=dtype, device=device()), sparse_map
    )
    _warmup_key = (tokens, top_k, num_experts, hidden, ffn_per_expert, dtype)
    if TritonConfig.enabled() and _warmup_key not in _output_sparse_warmed_up:
        _w_lhs = lhs_data.detach().requires_grad_(True)
        _w_rhs = rhs_data.detach().requires_grad_(True)
        _w_out = OutputSparseLinear.apply(_w_lhs, _w_rhs, sparse_map)
        _w_out.backward(backward_grad)
        del _w_lhs, _w_rhs, _w_out
        _output_sparse_warmed_up.add(_warmup_key)
    return {
        "lhs": lhs_data.requires_grad_(True),
        "rhs": rhs_data.requires_grad_(True),
        "sparse_map": sparse_map,
        "ffn_per_expert": ffn_per_expert,
        "backward_grad": backward_grad,
    }


def _make_input_inner_sparse_inputs(
    tokens: int, top_k: int, num_experts: int, hidden: int, ffn_per_expert: int, dtype: torch.dtype
) -> dict:
    sparse_map = _make_sparse_map(tokens, top_k, num_experts)
    lhs_data = _zero_padded_rows(
        torch.randn(sparse_map.num_rows, ffn_per_expert, dtype=dtype, device=device()), sparse_map
    )
    rhs_data = torch.randn(ffn_per_expert * num_experts, hidden, dtype=dtype, device=device())
    backward_grad = _zero_padded_rows(
        torch.ones(sparse_map.num_rows, hidden, dtype=dtype, device=device()), sparse_map
    )
    _warmup_key = (tokens, top_k, num_experts, hidden, ffn_per_expert, dtype)
    if TritonConfig.enabled() and _warmup_key not in _input_inner_sparse_warmed_up:
        _w_lhs = lhs_data.detach().requires_grad_(True)
        _w_rhs = rhs_data.detach().requires_grad_(True)
        _w_out = InputSparseLinear.apply(_w_lhs, _w_rhs, sparse_map)
        _w_out.backward(backward_grad)
        del _w_lhs, _w_rhs, _w_out
        _input_inner_sparse_warmed_up.add(_warmup_key)
    return {
        "lhs": lhs_data.requires_grad_(True),
        "rhs": rhs_data.requires_grad_(True),
        "sparse_map": sparse_map,
        "ffn_per_expert": ffn_per_expert,
        "backward_grad": backward_grad,
    }


# --------------------------------------------------------------------------- output_sparse references


def _output_sparse_loop(lhs: torch.Tensor, rhs: torch.Tensor, sparse_map: SparseMap) -> torch.Tensor:
    ffn_per_expert = rhs.shape[1] // sparse_map.num_experts
    out = lhs.new_zeros(sparse_map.num_rows, ffn_per_expert)
    for e in range(sparse_map.num_experts):
        row_begin = int(sparse_map.expert_ends[e - 1]) if e > 0 else 0
        row_end = int(sparse_map.expert_pad_begins[e])
        if row_end > row_begin:
            col_begin = e * ffn_per_expert
            out[row_begin:row_end] = lhs[row_begin:row_end] @ rhs[:, col_begin : col_begin + ffn_per_expert]
    return out


_output_sparse_compiled = torch.compile(_output_sparse_loop, mode="default", dynamic=False)


def _run_output_sparse_fwd(inputs: dict, fn) -> dict:
    return {"output": fn(inputs["lhs"], inputs["rhs"], inputs["sparse_map"])}


def _run_output_sparse_fwd_bwd(inputs: dict, fn) -> dict:
    output = fn(inputs["lhs"], inputs["rhs"], inputs["sparse_map"])
    output.backward(inputs["backward_grad"])
    return {"output": output.detach(), "grad_lhs": inputs["lhs"].grad, "grad_rhs": inputs["rhs"].grad}


def _run_output_sparse_fwd_fp32(inputs: dict) -> dict:
    lhs_fp32 = inputs["lhs"].float().detach().requires_grad_(True)
    rhs_fp32 = inputs["rhs"].float().detach().requires_grad_(True)
    return {"output": _output_sparse_loop(lhs_fp32, rhs_fp32, inputs["sparse_map"])}


def _run_output_sparse_fwd_bwd_fp32(inputs: dict) -> dict:
    lhs_fp32 = inputs["lhs"].float().detach().requires_grad_(True)
    rhs_fp32 = inputs["rhs"].float().detach().requires_grad_(True)
    output = _output_sparse_loop(lhs_fp32, rhs_fp32, inputs["sparse_map"])
    output.backward(inputs["backward_grad"].float())
    return {"output": output.detach(), "grad_lhs": lhs_fp32.grad, "grad_rhs": rhs_fp32.grad}


def _run_output_sparse_fwd_triton(inputs: dict) -> dict:
    return {"output": OutputSparseLinear.apply(inputs["lhs"], inputs["rhs"], inputs["sparse_map"])}


def _run_output_sparse_fwd_bwd_triton(inputs: dict) -> dict:
    output = OutputSparseLinear.apply(inputs["lhs"], inputs["rhs"], inputs["sparse_map"])
    output.backward(inputs["backward_grad"])
    return {"output": output.detach(), "grad_lhs": inputs["lhs"].grad, "grad_rhs": inputs["rhs"].grad}


def _output_sparse_variants() -> list[Variant]:
    variants = [
        Variant(
            name="fp32_reference",
            fwd=_run_output_sparse_fwd_fp32,
            fwd_bwd=_run_output_sparse_fwd_bwd_fp32,
            is_reference=True,
        ),
        Variant(
            name="pytorch_loop",
            fwd=lambda inputs: _run_output_sparse_fwd(inputs, _output_sparse_loop),
            fwd_bwd=lambda inputs: _run_output_sparse_fwd_bwd(inputs, _output_sparse_loop),
        ),
        Variant(
            name="pytorch_compiled",
            fwd=lambda inputs: _run_output_sparse_fwd(inputs, _output_sparse_compiled),
            fwd_bwd=lambda inputs: _run_output_sparse_fwd_bwd(inputs, _output_sparse_compiled),
        ),
    ]
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_run_output_sparse_fwd_triton,
                fwd_bwd=_run_output_sparse_fwd_bwd_triton,
            )
        )
    return variants


# --------------------------------------------------------------------------- input_inner_sparse references


def _input_inner_sparse_loop(lhs: torch.Tensor, rhs: torch.Tensor, sparse_map: SparseMap) -> torch.Tensor:
    ffn_per_expert = rhs.shape[0] // sparse_map.num_experts
    out = lhs.new_zeros(sparse_map.num_rows, rhs.shape[1])
    for e in range(sparse_map.num_experts):
        row_begin = int(sparse_map.expert_ends[e - 1]) if e > 0 else 0
        row_end = int(sparse_map.expert_pad_begins[e])
        if row_end > row_begin:
            inner_begin = e * ffn_per_expert
            out[row_begin:row_end] = lhs[row_begin:row_end] @ rhs[inner_begin : inner_begin + ffn_per_expert]
    return out


_input_inner_sparse_compiled = torch.compile(_input_inner_sparse_loop, mode="default", dynamic=False)


def _run_input_inner_sparse_fwd(inputs: dict, fn) -> dict:
    return {"output": fn(inputs["lhs"], inputs["rhs"], inputs["sparse_map"])}


def _run_input_inner_sparse_fwd_bwd(inputs: dict, fn) -> dict:
    output = fn(inputs["lhs"], inputs["rhs"], inputs["sparse_map"])
    output.backward(inputs["backward_grad"])
    return {"output": output.detach(), "grad_lhs": inputs["lhs"].grad, "grad_rhs": inputs["rhs"].grad}


def _run_input_inner_sparse_fwd_fp32(inputs: dict) -> dict:
    lhs_fp32 = inputs["lhs"].float().detach().requires_grad_(True)
    rhs_fp32 = inputs["rhs"].float().detach().requires_grad_(True)
    return {"output": _input_inner_sparse_loop(lhs_fp32, rhs_fp32, inputs["sparse_map"])}


def _run_input_inner_sparse_fwd_bwd_fp32(inputs: dict) -> dict:
    lhs_fp32 = inputs["lhs"].float().detach().requires_grad_(True)
    rhs_fp32 = inputs["rhs"].float().detach().requires_grad_(True)
    output = _input_inner_sparse_loop(lhs_fp32, rhs_fp32, inputs["sparse_map"])
    output.backward(inputs["backward_grad"].float())
    return {"output": output.detach(), "grad_lhs": lhs_fp32.grad, "grad_rhs": rhs_fp32.grad}


def _run_input_inner_sparse_fwd_triton(inputs: dict) -> dict:
    return {"output": InputSparseLinear.apply(inputs["lhs"], inputs["rhs"], inputs["sparse_map"])}


def _run_input_inner_sparse_fwd_bwd_triton(inputs: dict) -> dict:
    output = InputSparseLinear.apply(inputs["lhs"], inputs["rhs"], inputs["sparse_map"])
    output.backward(inputs["backward_grad"])
    return {"output": output.detach(), "grad_lhs": inputs["lhs"].grad, "grad_rhs": inputs["rhs"].grad}


def _input_inner_sparse_variants() -> list[Variant]:
    variants = [
        Variant(
            name="fp32_reference",
            fwd=_run_input_inner_sparse_fwd_fp32,
            fwd_bwd=_run_input_inner_sparse_fwd_bwd_fp32,
            is_reference=True,
        ),
        Variant(
            name="pytorch_loop",
            fwd=lambda inputs: _run_input_inner_sparse_fwd(inputs, _input_inner_sparse_loop),
            fwd_bwd=lambda inputs: _run_input_inner_sparse_fwd_bwd(inputs, _input_inner_sparse_loop),
        ),
        Variant(
            name="pytorch_compiled",
            fwd=lambda inputs: _run_input_inner_sparse_fwd(inputs, _input_inner_sparse_compiled),
            fwd_bwd=lambda inputs: _run_input_inner_sparse_fwd_bwd(inputs, _input_inner_sparse_compiled),
        ),
    ]
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_run_input_inner_sparse_fwd_triton,
                fwd_bwd=_run_input_inner_sparse_fwd_bwd_triton,
            )
        )
    return variants


# --------------------------------------------------------------------------- cases / bytes / flops


def _bytes_per_element(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _sparse_linear_bytes(
    sparse_tokens: int, hidden: int, ffn_per_expert: int, num_experts: int, dtype: torch.dtype
) -> int:
    element_size = _bytes_per_element(dtype)
    # fwd: read lhs + read rhs_full + write output
    # bwd: read grad_output + read rhs_full + write grad_lhs + read lhs + read grad_output + write grad_rhs
    # Simplification: 3× lhs traffic + 3× rhs traffic + 2× output traffic
    lhs_bytes = sparse_tokens * hidden * element_size
    rhs_bytes = hidden * ffn_per_expert * num_experts * element_size
    out_bytes = sparse_tokens * ffn_per_expert * element_size
    return 3 * lhs_bytes + 3 * rhs_bytes + 2 * out_bytes


def _sparse_linear_flops(sparse_tokens_unpadded: int, hidden: int, ffn_per_expert: int) -> int:
    # fwd + bwd ≈ 3 matmuls (fwd: lhs@rhs, bwd_lhs: grad@rhs.T, bwd_rhs: lhs.T@grad)
    return 3 * 2 * sparse_tokens_unpadded * hidden * ffn_per_expert


def _output_sparse_cases(
    dtypes: tuple[torch.dtype, ...], shapes: list[tuple[int, int, int, int, int]] | None = None
) -> list[Case]:
    shapes = shapes if shapes is not None else _SHAPES
    return [
        Case(
            name=case_name("output_sparse", (tokens, top_k, num_experts, hidden, ffn_per_expert), dtype),
            make_inputs=partial(_make_output_sparse_inputs, tokens, top_k, num_experts, hidden, ffn_per_expert, dtype),
            expected_bytes=_sparse_linear_bytes(tokens * top_k, hidden, ffn_per_expert, num_experts, dtype),
            expected_flops=_sparse_linear_flops(tokens * top_k, hidden, ffn_per_expert),
            compute_dtype=dtype,
        )
        for dtype in dtypes
        for tokens, top_k, num_experts, hidden, ffn_per_expert in shapes
    ]


def _input_inner_sparse_cases(
    dtypes: tuple[torch.dtype, ...], shapes: list[tuple[int, int, int, int, int]] | None = None
) -> list[Case]:
    shapes = shapes if shapes is not None else _SHAPES
    return [
        Case(
            name=case_name("input_inner_sparse", (tokens, top_k, num_experts, hidden, ffn_per_expert), dtype),
            make_inputs=partial(
                _make_input_inner_sparse_inputs, tokens, top_k, num_experts, hidden, ffn_per_expert, dtype
            ),
            expected_bytes=_sparse_linear_bytes(tokens * top_k, ffn_per_expert, hidden, num_experts, dtype),
            expected_flops=_sparse_linear_flops(tokens * top_k, ffn_per_expert, hidden),
            compute_dtype=dtype,
        )
        for dtype in dtypes
        for tokens, top_k, num_experts, hidden, ffn_per_expert in shapes
    ]


# --------------------------------------------------------------------------- entry point


def run(
    verbose: bool = False,
    dtypes: tuple[torch.dtype, ...] | None = None,
    shapes: list[tuple[int, int, int, int, int]] | None = None,
) -> None:
    dtypes = tuple(dtypes) if dtypes else _DEFAULT_DTYPES
    run_benchmark(
        "sparse_linear: output_sparse (layer 1 / up-proj)",
        _output_sparse_cases(dtypes, shapes),
        _output_sparse_variants(),
        verbose=verbose,
    )
    run_benchmark(
        "sparse_linear: input_inner_sparse (layer 2 / down-proj)",
        _input_inner_sparse_cases(dtypes, shapes),
        _input_inner_sparse_variants(),
        verbose=verbose,
    )


if __name__ == "__main__":
    run()
