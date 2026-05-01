"""
Smoke tests for all benchmark modules.

One test per sub-benchmark (kernel): inputs are tiny so the runner code path is
exercised quickly without requiring a full benchmark run.

Patches applied to keep each test under ~100 ms:
- torch.compile disabled (avoids JIT cold-start).
- fast_llm_triton variants replaced with fp32 reference (no Triton compilation;
  kernel correctness is covered by the main test suite).
- TritonConfig.enabled → False (prevents make_inputs warmup in sparse_linear).
- _cudagraph_mark_step_begin → None and synchronize → no-op (both cause C-level
  CUDA syncs per fn() call that dominate the wall time without this).
"""

import dataclasses

import pytest
import torch

import tools.benchmark.triton_kernels.runner as _bench_runner
from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton import triton_interpret
from fast_llm.utils import Assert
from tools.benchmark.triton_kernels import (
    bench_entropy_loss,
    bench_grpo_loss,
    bench_mlp_activation,
    bench_normalization,
    bench_pointwise,
    bench_rotary,
    bench_sparse_copy,
    bench_sparse_linear,
)
from tools.benchmark.triton_kernels.runner import run_benchmark

_DTYPES = (torch.float32,)

# sparse_copy and sparse_linear use tl.histogram, which has unfixed bugs in the
# Triton interpreter. Skip them in interpreter mode; they're covered on GPU.
_INTERPRETER_SKIP = {
    "sparse_copy: dispatch",
    "sparse_copy: combine",
    "sparse_linear: output_sparse (layer 1 / up-proj)",
    "sparse_linear: input_inner_sparse (layer 2 / down-proj)",
}

_SKIP_VARIANTS = {"pytorch_compiled", "pytorch_compiled_max"}


def _build_params() -> list:
    modules_and_shapes = [
        (bench_entropy_loss, {"shapes": [(64, 256)]}),
        (bench_grpo_loss, {"shapes": [(64, 256)]}),
        (bench_mlp_activation, {"shapes": [(64, 128)]}),
        (bench_normalization, {"shapes": [(64, 128)]}),
        (bench_pointwise, {"shapes": [1024]}),
        (bench_rotary, {"shapes": [(64, 4, 64)]}),
        (bench_sparse_copy, {"shapes": [(64, 2, 4, 128)]}),
        (bench_sparse_linear, {"shapes": [(64, 2, 4, 256, 256)]}),
    ]
    params = []
    for module, kwargs in modules_and_shapes:
        for name, cases, variants in module.benchmarks(dtypes=_DTYPES, **kwargs):
            params.append(pytest.param(name, cases, variants, id=name))
    return params


_PARAMS = _build_params()

# Guard against silent drift if a benchmark or variant is renamed: every entry
# in _INTERPRETER_SKIP / _SKIP_VARIANTS must match at least one real name.
_actual_benchmark_names = {p.id for p in _PARAMS}
_actual_variant_names = {v.name for p in _PARAMS for v in p.values[2]}
Assert.custom(set.issubset, _INTERPRETER_SKIP, _actual_benchmark_names)
Assert.custom(set.issubset, _SKIP_VARIANTS, _actual_variant_names)


@pytest.fixture(autouse=True)
def _patch_benchmark_env(monkeypatch):
    import torch._dynamo

    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    monkeypatch.setattr(TritonConfig, "enabled", lambda *a, **kw: False)
    monkeypatch.setattr(_bench_runner, "_cudagraph_mark_step_begin", None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


@pytest.mark.parametrize("name,cases,variants", _PARAMS)
def test_triton_benchmark(name, cases, variants):
    if triton_interpret and name in _INTERPRETER_SKIP:
        pytest.skip("tl.histogram is broken in the Triton interpreter")

    # Replace fast_llm_triton fwd/fwd_bwd with the fp32 reference so no Triton
    # kernels are compiled. The runner still exercises the full variant code path.
    variants = [v for v in variants if v.name not in _SKIP_VARIANTS]
    ref = next(v for v in variants if v.is_reference)
    variants = [
        dataclasses.replace(v, fwd=ref.fwd, fwd_bwd=ref.fwd_bwd) if v.name == "fast_llm_triton" else v
        for v in variants
    ]
    run_benchmark(name, cases, variants, warmup_ms=0, rep_ms=0, min_reps=1)
