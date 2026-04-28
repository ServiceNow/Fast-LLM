"""
CLI entry point for the Fast-LLM Triton kernel benchmarking suite.

Usage:
    python -m tools.benchmark <kernel>

Available kernels are discovered dynamically from `bench_*.py` files in this
package. Each such module must expose a `run(verbose: bool = False)` callable.
"""

import argparse
import importlib
import logging
import pkgutil
import warnings

# Each bench file compiles the same function with multiple shapes/dtypes; the
# default cache size (8) is too small, causing dynamo to give up and fall back
# to eager. Bump it before any `torch.compile`-decorated code runs.
import torch._dynamo

import tools.benchmark as _pkg
from fast_llm.engine.config_utils.data_type import DataType

torch._dynamo.config.cache_size_limit = 64

# In-place ops (copy_, fill_, add with out=) emit "skipping cudagraphs due to
# mutated inputs" when using max-autotune. The fallback is correct; suppress noise.
warnings.filterwarnings("ignore", message=".*[Ss]kipping (cuda|CUDA)[Gg]raphs.*")
logging.getLogger("torch._inductor.cudagraph_trees").setLevel(logging.ERROR)


def _list_benchmarks() -> dict[str, str]:
    """Map short kernel name → fully-qualified bench module name."""
    names = {}
    for info in pkgutil.iter_modules(_pkg.__path__):
        if info.name.startswith("bench_"):
            names[info.name.removeprefix("bench_")] = f"tools.benchmark.{info.name}"
    return names


def main() -> None:
    benches = _list_benchmarks()
    parser = argparse.ArgumentParser(
        prog="python -m tools.benchmark",
        description="Benchmark Fast-LLM Triton kernels against PyTorch alternatives.",
    )
    parser.add_argument(
        "kernels",
        nargs="*",
        choices=sorted(benches),
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
    dtypes = [DataType(d).torch for d in args.dtypes]

    selected = args.kernels or sorted(benches)
    for kernel in selected:
        importlib.import_module(benches[kernel]).run(verbose=args.verbose, dtypes=dtypes)


if __name__ == "__main__":
    main()
