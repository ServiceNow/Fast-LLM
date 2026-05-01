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

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.sparse_copy import SparseMap, get_sparse_map
from fast_llm.functional.triton.sparse_linear import InputSparseLinear, OutputSparseLinear
from tools.benchmark.runner import Variant
from tools.benchmark.utils import bench_main, device, make_cases, standard_fwd_bwd_pytorch_variants

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


def _mask_padded_rows(candidate: dict[str, torch.Tensor], inputs: dict) -> dict[str, torch.Tensor]:
    # Two regions in the kernel's forward output and grad_lhs are by-design garbage that
    # downstream consumers ignore: per-expert padding [pad_begin, expert_end) (where the
    # kernel does a matmul on random padding inputs) and phantom rows [expert_ends[-1],
    # num_rows) past the last expert (where the kernel early-returns and leaves the output
    # buffer uninitialized). The loop reference produces zeros in both regions, so without
    # masking those mismatches would dominate rel_rms. grad_rhs already excludes padded
    # contributions in both the kernel and reference, so it needs no masking.
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


def _make_output_sparse_inputs(
    tokens: int, top_k: int, num_experts: int, hidden: int, ffn_per_expert: int, dtype: torch.dtype
) -> dict:
    sparse_map = _make_sparse_map(tokens, top_k, num_experts)
    lhs_data = torch.randn(sparse_map.num_rows, hidden, dtype=dtype, device=device())
    rhs_data = torch.randn(hidden, ffn_per_expert * num_experts, dtype=dtype, device=device())
    backward_grad = torch.ones(sparse_map.num_rows, ffn_per_expert, dtype=dtype, device=device())
    warmup_key = (tokens, top_k, num_experts, hidden, ffn_per_expert, dtype)
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


def _make_input_inner_sparse_inputs(
    tokens: int, top_k: int, num_experts: int, hidden: int, ffn_per_expert: int, dtype: torch.dtype
) -> dict:
    sparse_map = _make_sparse_map(tokens, top_k, num_experts)
    lhs_data = torch.randn(sparse_map.num_rows, ffn_per_expert, dtype=dtype, device=device())
    rhs_data = torch.randn(ffn_per_expert * num_experts, hidden, dtype=dtype, device=device())
    backward_grad = torch.ones(sparse_map.num_rows, hidden, dtype=dtype, device=device())
    warmup_key = (tokens, top_k, num_experts, hidden, ffn_per_expert, dtype)
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


# --------------------------------------------------------------------------- output_sparse


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


def _output_sparse_triton_fwd(inputs: dict) -> dict:
    return {"output": OutputSparseLinear.apply(inputs["lhs"], inputs["rhs"], inputs["sparse_map"])}


def _output_sparse_triton_fwd_bwd(inputs: dict) -> dict:
    output = OutputSparseLinear.apply(inputs["lhs"], inputs["rhs"], inputs["sparse_map"])
    output.backward(inputs["backward_grad"])
    return {"output": output.detach(), "grad_lhs": inputs["lhs"].grad, "grad_rhs": inputs["rhs"].grad}


def _output_sparse_variants() -> list[Variant]:
    variants = standard_fwd_bwd_pytorch_variants(
        _output_sparse_loop,
        input_keys=("lhs", "rhs", "sparse_map"),
        grad_input_keys=("lhs", "rhs"),
        grad_output_key="backward_grad",
        eager_name="pytorch_loop",
        enable_max_autotune=False,
    )
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_output_sparse_triton_fwd,
                fwd_bwd=_output_sparse_triton_fwd_bwd,
                output_postprocess=_mask_padded_rows,
            )
        )
    return variants


# --------------------------------------------------------------------------- input_inner_sparse


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


def _input_inner_sparse_triton_fwd(inputs: dict) -> dict:
    return {"output": InputSparseLinear.apply(inputs["lhs"], inputs["rhs"], inputs["sparse_map"])}


def _input_inner_sparse_triton_fwd_bwd(inputs: dict) -> dict:
    output = InputSparseLinear.apply(inputs["lhs"], inputs["rhs"], inputs["sparse_map"])
    output.backward(inputs["backward_grad"])
    return {"output": output.detach(), "grad_lhs": inputs["lhs"].grad, "grad_rhs": inputs["rhs"].grad}


def _input_inner_sparse_variants() -> list[Variant]:
    variants = standard_fwd_bwd_pytorch_variants(
        _input_inner_sparse_loop,
        input_keys=("lhs", "rhs", "sparse_map"),
        grad_input_keys=("lhs", "rhs"),
        grad_output_key="backward_grad",
        eager_name="pytorch_loop",
        enable_max_autotune=False,
    )
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_input_inner_sparse_triton_fwd,
                fwd_bwd=_input_inner_sparse_triton_fwd_bwd,
                output_postprocess=_mask_padded_rows,
            )
        )
    return variants


# --------------------------------------------------------------------------- bytes / flops


def _sparse_linear_bytes(
    tokens: int, top_k: int, num_experts: int, hidden: int, ffn_per_expert: int, dtype: torch.dtype
) -> int:
    # fwd: read lhs + read rhs_full + write output
    # bwd: read grad_output + read rhs_full + write grad_lhs + read lhs + read grad_output + write grad_rhs
    # Simplification: 3× lhs traffic + 3× rhs traffic + 2× output traffic
    sparse_tokens = tokens * top_k
    lhs_bytes = sparse_tokens * hidden * dtype.itemsize
    rhs_bytes = hidden * ffn_per_expert * num_experts * dtype.itemsize
    output_bytes = sparse_tokens * ffn_per_expert * dtype.itemsize
    return 3 * lhs_bytes + 3 * rhs_bytes + 2 * output_bytes


def _sparse_linear_flops(tokens: int, top_k: int, num_experts: int, hidden: int, ffn_per_expert: int) -> int:
    # fwd + bwd ≈ 3 matmuls (fwd: lhs@rhs, bwd_lhs: grad@rhs.T, bwd_rhs: lhs.T@grad)
    return 3 * 2 * tokens * top_k * hidden * ffn_per_expert


# --------------------------------------------------------------------------- entry point


def benchmarks(
    dtypes: tuple[torch.dtype, ...] | None = None,
    shapes: list[tuple[int, int, int, int, int]] | None = None,
) -> list[tuple[str, list, list]]:
    dtypes = tuple(dtypes) if dtypes else _DEFAULT_DTYPES
    shapes = shapes if shapes is not None else _SHAPES
    return [
        (
            "sparse_linear: output_sparse (layer 1 / up-proj)",
            make_cases(
                "output_sparse", dtypes, shapes, _make_output_sparse_inputs, _sparse_linear_bytes, _sparse_linear_flops
            ),
            _output_sparse_variants(),
        ),
        (
            "sparse_linear: input_inner_sparse (layer 2 / down-proj)",
            make_cases(
                "input_inner_sparse",
                dtypes,
                shapes,
                _make_input_inner_sparse_inputs,
                _sparse_linear_bytes,
                _sparse_linear_flops,
            ),
            _input_inner_sparse_variants(),
        ),
    ]


run = bench_main(benchmarks)


if __name__ == "__main__":
    run()
