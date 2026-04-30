"""
Smoke tests for all benchmark modules.

One test per sub-benchmark (kernel): inputs are tiny so the runner code path is
exercised quickly on CPU. torch.compile is disabled via a fixture to avoid the
JIT cold-start (~20 s on CPU per variant). warmup_ms=0 and min_reps=1 cap the
rep count to one timed call per variant so the suite stays fast.
"""

import pytest
import torch

from tools.benchmark import (
    bench_entropy_loss,
    bench_grpo_loss,
    bench_mlp_activation,
    bench_normalization,
    bench_pointwise,
    bench_rotary,
    bench_sparse_copy,
    bench_sparse_linear,
)
from tools.benchmark.runner import run_benchmark

_DTYPES = (torch.float32,)


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


@pytest.fixture(autouse=True)
def _disable_dynamo():
    import torch._dynamo

    orig = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    yield
    torch._dynamo.config.disable = orig


@pytest.mark.parametrize("name,cases,variants", _PARAMS)
def test_triton_benchmark(name, cases, variants):
    run_benchmark(name, cases, variants, warmup_ms=0, rep_ms=0, min_reps=1)
