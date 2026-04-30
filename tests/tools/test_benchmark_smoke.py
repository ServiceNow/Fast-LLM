"""
Smoke tests for the benchmark runner and bench_pointwise wiring.

Both tests run without a GPU: bench_fn falls back to wall-clock timing (one
warmup call + one timed call) when CUDA is unavailable, and device() returns
"cpu".  The compiled variants may fail on CPU — that is expected and does not
cause the test to fail; the runner records the error per-variant rather than
raising.
"""

import torch

import tools.benchmark.bench_pointwise as bench_pointwise
from tools.benchmark.runner import Case, Variant, run_benchmark


def test_run_benchmark_wiring():
    """Core runner machinery (Case, Variant, correctness comparison, table printing) works end-to-end."""

    def _relu_fp32(inputs: dict) -> dict:
        return {"out": torch.relu(inputs["x"].float())}

    def _relu(inputs: dict) -> dict:
        return {"out": torch.relu(inputs["x"])}

    cases = [Case(name="relu_256", make_inputs=lambda: {"x": torch.randn(256)})]
    variants = [
        Variant(name="fp32_reference", fwd=_relu_fp32, is_reference=True),
        Variant(name="eager", fwd=_relu),
    ]
    results = run_benchmark("smoke: relu", cases, variants)
    assert len(results) == 1
    _case, variant_results = results[0]
    assert all(r.error is None for r in variant_results), [r.error for r in variant_results]


def test_bench_pointwise_smoke(monkeypatch):
    """bench_pointwise case/variant wiring is intact end-to-end with tiny inputs."""
    monkeypatch.setattr(bench_pointwise, "_SIZES_NUMEL", [1024])

    for make_cases, variants, label in [
        (bench_pointwise._copy_cases, bench_pointwise._COPY_VARIANTS, "copy"),
        (bench_pointwise._fill_cases, bench_pointwise._FILL_VARIANTS, "fill"),
        (bench_pointwise._add_cases, bench_pointwise._ADD_VARIANTS, "add"),
    ]:
        results = run_benchmark(f"smoke: {label}", make_cases((torch.float32,)), variants)
        assert len(results) == 1, f"{label}: expected 1 case, got {len(results)}"
        _case, variant_results = results[0]
        for r in variant_results:
            if r.variant_name in ("fp32_reference", "pytorch_eager"):
                assert r.error is None, f"{label}/{r.variant_name}: {r.error}"
