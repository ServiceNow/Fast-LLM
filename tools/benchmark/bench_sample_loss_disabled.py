"""Compare unweighted entropy graphs and GPU timing with a supplied pre-change module."""

import argparse
import importlib.util
import pathlib
import re
import statistics

import torch

from fast_llm.functional.entropy_loss import fused_entropy_loss_forward_backward
from fast_llm.functional.triton import entropy_loss as triton_entropy


def compare(
    baseline_path: pathlib.Path,
    rows: int,
    vocabulary: int,
    iterations: int,
    baseline_triton_path: pathlib.Path | None = None,
) -> None:
    spec = importlib.util.spec_from_file_location("sample_loss_baseline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits = torch.randn(rows, vocabulary, device=device, dtype=torch.bfloat16)
    target = torch.randint(vocabulary, (rows,), device=device)
    arguments = (logits, target, None)
    keywords = {"grad_output": 1.0, "divisor": rows}
    functions = [baseline.fused_entropy_loss_forward_backward, fused_entropy_loss_forward_backward]
    graphs = []
    for function in functions:
        explained = torch._dynamo.explain(function)(*arguments, **keywords)
        graphs.append(
            [
                [(node.op, str(node.target)) for node in graph.graph.nodes if node.op not in {"placeholder", "output"}]
                for graph in explained.graphs
            ]
        )
    assert graphs[0] == graphs[1], "Unweighted operation graphs changed"
    print("PASS: unweighted graph operation sequences match the supplied baseline")
    expected = functions[0](*arguments, **keywords)
    actual = functions[1](*arguments, **keywords)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference)
    if device == "cpu":
        print("GPU timing skipped: CUDA unavailable")
        return
    for function in functions:
        for _ in range(10):
            function(*arguments, **keywords)
    torch.cuda.synchronize()
    measurements = [[], []]
    for _ in range(7):
        for index, function in enumerate(functions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                function(*arguments, **keywords)
            end.record()
            end.synchronize()
            measurements[index].append(start.elapsed_time(end) / iterations)
    before, after = [statistics.median(values) for values in measurements]
    print(
        f"Unweighted fused backend ({rows} rows, {vocabulary} vocabulary): baseline={before:.5f} ms current={after:.5f} ms ratio={after / before:.4f}"
    )
    if baseline_triton_path is not None:
        spec = importlib.util.spec_from_file_location("sample_triton_baseline", baseline_triton_path)
        baseline_triton = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline_triton)
        kernels = []
        losses = torch.empty(rows, device=device)
        gradient = torch.empty_like(logits)
        for module in [baseline_triton, triton_entropy]:
            kernel = module.triton_cross_entropy_forward_backward_from_labels_kernel[(rows,)](
                logits,
                target,
                n_cols=vocabulary,
                logits_stride_0=vocabulary,
                block_size=min(32768, 2 ** (vocabulary - 1).bit_length()),
                losses_ptr=losses,
                grad_losses=1.0 / rows,
                grad_logits_ptr=gradient,
                grad_logits_stride_0=vocabulary,
                num_warps=16,
            )
            kernels.append(kernel)
        instructions = []
        for kernel in kernels:
            instructions.append(
                [
                    match.group(1)
                    for line in kernel.asm["ptx"].splitlines()
                    if (match := re.match(r"\s*(?:@!?%\w+\s+)?([a-z][\w.]*)\s+[^;]*;", line))
                ]
            )
        assert instructions[0] == instructions[1], "Unweighted Triton PTX instruction sequences changed"
        assert kernels[0].n_regs == kernels[1].n_regs
        assert kernels[0].metadata.shared == kernels[1].metadata.shared
        print(f"PASS: unweighted Triton PTX instruction sequences and resources match ({kernels[1].n_regs} registers)")
    print("Timing is diagnostic; representative training-step measurements are still required.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-triton", type=pathlib.Path)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--vocabulary", type=int, default=262144)
    parser.add_argument("--iterations", type=int, default=100)
    options = parser.parse_args()
    compare(options.baseline, options.rows, options.vocabulary, options.iterations, options.baseline_triton)
