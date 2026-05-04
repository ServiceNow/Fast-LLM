"""
CLI entry point for the Fast-LLM Triton kernel benchmarking suite.

Usage:
    python -m tools.benchmark.triton_kernels <kernel>
"""

import argparse
import logging
import warnings

# Each bench file compiles the same function with multiple shapes/dtypes; the
# default cache size (8) is too small, causing dynamo to give up and fall back
# to eager. Bump it before any `torch.compile`-decorated code runs.
import torch._dynamo

from fast_llm.engine.config_utils.data_type import DataType
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

torch._dynamo.config.cache_size_limit = 64

# In-place ops (copy_, fill_, add with out=) emit "skipping cudagraphs due to
# mutated inputs" when using max-autotune. The fallback is correct; suppress noise.
warnings.filterwarnings("ignore", message=".*[Ss]kipping (cuda|CUDA)[Gg]raphs.*")
logging.getLogger("torch._inductor.cudagraph_trees").setLevel(logging.ERROR)

_BENCHMARKS = {
    "entropy_loss": bench_entropy_loss,
    "grpo_loss": bench_grpo_loss,
    "mlp_activation": bench_mlp_activation,
    "normalization": bench_normalization,
    "pointwise": bench_pointwise,
    "rotary": bench_rotary,
    "sparse_copy": bench_sparse_copy,
    "sparse_linear": bench_sparse_linear,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tools.benchmark.triton_kernels",
        description="Benchmark Fast-LLM Triton kernels against PyTorch alternatives.",
    )
    parser.add_argument(
        "kernels",
        nargs="*",
        choices=sorted(_BENCHMARKS),
        metavar="kernel",
        help="Which kernels to benchmark. If omitted, run all kernels.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show additional timing columns (mean, min, max).",
    )
    parser.add_argument(
        "-d",
        "--dtypes",
        nargs="+",
        default=["bfloat16"],
        metavar="DTYPE",
        help="Fast-LLM DataType names to sweep (default: bfloat16). "
        "Examples: bfloat16 float32 fp16. Accepts alternate names like bf16.",
    )
    args = parser.parse_args()
    dtypes = tuple(DataType(d).torch for d in args.dtypes)

    selected = args.kernels or sorted(_BENCHMARKS)
    for kernel in selected:
        for name, cases, variants in _BENCHMARKS[kernel].benchmarks(dtypes=dtypes):
            run_benchmark(name, cases, variants, verbose=args.verbose)


if __name__ == "__main__":
    main()
