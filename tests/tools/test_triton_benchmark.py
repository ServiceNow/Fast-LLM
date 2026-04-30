"""
Smoke tests for all benchmark modules.

Each test runs a single benchmark module with one tiny shape and float32 dtype
so the full runner code path is exercised quickly on CPU.  torch.compile is
disabled via a fixture to avoid the JIT cold-start (~20 s on CPU per variant).
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

_DTYPES = (torch.float32,)

_PARAMS = [
    pytest.param(bench_entropy_loss, {"shapes": [(64, 256)]}, id="entropy_loss"),
    pytest.param(bench_grpo_loss, {"shapes": [(64, 256)]}, id="grpo_loss"),
    pytest.param(bench_mlp_activation, {"shapes": [(64, 128)]}, id="mlp_activation"),
    pytest.param(bench_normalization, {"shapes": [(64, 128)]}, id="normalization"),
    pytest.param(bench_pointwise, {"shapes": [1024]}, id="pointwise"),
    pytest.param(bench_rotary, {"shapes": [(64, 4, 64)]}, id="rotary"),
    pytest.param(bench_sparse_copy, {"shapes": [(64, 2, 4, 128)]}, id="sparse_copy"),
    pytest.param(bench_sparse_linear, {"shapes": [(64, 2, 4, 256, 256)]}, id="sparse_linear"),
]


@pytest.fixture(autouse=True)
def _disable_dynamo():
    import torch._dynamo

    orig = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    yield
    torch._dynamo.config.disable = orig


@pytest.mark.parametrize("module,kwargs", _PARAMS)
def test_triton_benchmark(module, kwargs):
    module.run(dtypes=_DTYPES, **kwargs)
