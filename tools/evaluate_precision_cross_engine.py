"""Cross-engine per-token log-probability comparison for Fast-LLM, DeepSpeed and vLLM.

The within-engine tools (`tools/evaluate_precision{,_deepspeed,_vllm}.py`) each measure a variant
against *its own* fp32 reference — that isolates one engine's internal rounding. This tool measures
the gap *across* engines on the identical input: the per-token log-ratio

    δ = log π_A − log π_B

over the chosen tokens. When A is a trainer and B is the vLLM sampler, δ is the log of the RL
importance ratio exp(log π_train − log π_old) that multiplies the advantage, so δ is exactly the
quantity that perturbs the gradient — and the gap the literature quotes in nats.

Each engine's per-token log π is a plain fp32 vector saved by its within-engine tool
(`<dir>/logprobs_<variant>.pt`), all length L−1 and aligned 1:1 on the shared input. This tool maps
each engine's variant names to three canonical configs and reports two things per regime:

  1. the δ distribution (mean = systematic bias, RMS, max, per-sequence sum, PPO clip fraction) over
     every engine pair, for the fp32 floor and, per low precision (bf16, fp16), the full 2×2 of head
     choice (fp32 upcast vs body-dtype head) on each side — so both matched cases (both fp32, both
     body) and both mismatch directions appear. In production vLLM emits fp32 logits, so the relevant
     mismatch is a body-dtype head on the *training* side against vLLM's fp32 head;
  2. the error-correlation decomposition. With e = log π_bf16 − log π_fp32 per engine,
     δ_AB = (fp32 floor) + (e_A − e_B), and RMS(e_A − e_B) is governed by ρ = corr(e_A, e_B):
     ρ→1 means the engines round the same way and the errors cancel (gap collapses to the floor);
     ρ→0 means independent rounding (errors add in quadrature). fp32-head matching works by removing
     the large, cross-engine-uncorrelated head-rounding component, which is why it matters across
     engines while being nearly invisible within one.

Run (any subset of the three dirs; pairs are formed from whatever is available):

    python -m tools.evaluate_precision_cross_engine \\
        --fast-llm-dir <shared>/text --deepspeed-dir <shared>/text/ds --vllm-dir <shared>/text/vllm \\
        --label text
"""

import argparse
import itertools
import pathlib

import torch

# Canonical cross-engine configs, mapped to each within-engine tool's variant names. For a precision
# `prec`, `<prec>_fp32head` keeps the LM head / logits in fp32 while `<prec>_<prec>head` runs the head
# in the body dtype. vLLM has no fp16+fp32 head (its quant rejects an fp16 body), so that entry is absent.
_ENGINE_VARIANTS: dict[str, dict[str, str]] = {
    "fast_llm": {
        "fp32": "reference",
        "bf16_fp32head": "bf16_fp32_lm_head",
        "bf16_bf16head": "bf16",
        "fp16_fp32head": "fp16_fp32_lm_head",
        "fp16_fp16head": "fp16",
    },
    "deepspeed": {
        "fp32": "fp32",
        "bf16_fp32head": "bf16",
        "bf16_bf16head": "bf16_head_bf16",
        "fp16_fp32head": "fp16",
        "fp16_fp16head": "fp16_head_fp16",
    },
    "vllm": {
        "fp32": "fp32",
        "bf16_fp32head": "bf16_fp32_head",
        "bf16_bf16head": "bf16",
        "fp16_fp16head": "fp16",
    },
}
_PRECISIONS = ("bf16", "fp16")
_ENGINE_ORDER = ("fast_llm", "deepspeed", "vllm")
_ENGINE_LABELS = {"fast_llm": "Fast-LLM", "deepspeed": "DeepSpeed", "vllm": "vLLM"}


# Head precision per side: "fp32" upcasts the head / logits to fp32; "body" runs the head in the body
# dtype. In production vLLM always emits fp32 logits, so the relevant mismatch is a body-dtype head on
# the *training* side against vLLM's fp32 head.
_HEADS = ("fp32", "body")


def _head_config(precision: str, head: str) -> str:
    return f"{precision}_fp32head" if head == "fp32" else f"{precision}_{precision}head"


def _head_label(precision: str, head: str) -> str:
    return "fp32" if head == "fp32" else precision


def _rms(x: torch.Tensor) -> float:
    return x.pow(2).mean().sqrt().item()


def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    return (x @ y / denom).item() if denom > 0 else float("nan")


def _slope(reference: torch.Tensor, test: torch.Tensor) -> float:
    # Regression of `test` on `reference` (test ≈ slope · reference); slope ≠ 1 is a multiplicative
    # mismatch (e.g. a temperature/scale discrepancy), distinct from the additive offset mean(δ) catches.
    reference = reference - reference.mean()
    test = test - test.mean()
    denom = reference.pow(2).sum()
    return (reference @ test / denom).item() if denom > 0 else float("nan")


def _delta_stats(a: torch.Tensor, b: torch.Tensor, epsilon: float) -> dict[str, float]:
    delta = a - b
    abs_delta = delta.abs()
    return {
        "mean": delta.mean().item(),
        "rms": _rms(delta),
        "max": abs_delta.max().item(),
        "sum": delta.sum().item(),
        "clip": (abs_delta > epsilon).float().mean().item(),
    }


def _load_engine(engine: str, directory: pathlib.Path) -> dict[str, list[torch.Tensor]]:
    """Load each config's per-token log π as a list of per-sequence vectors. A single-sequence run saves a
    bare tensor (wrapped as a one-element list); a multi-sequence run saves a list of per-sequence tensors."""
    vectors: dict[str, list[torch.Tensor]] = {}
    for config, variant in _ENGINE_VARIANTS[engine].items():
        path = directory / f"logprobs_{variant}.pt"
        if path.exists():
            loaded = torch.load(path, map_location="cpu")
            sequences = loaded if isinstance(loaded, list) else [loaded]
            vectors[config] = [sequence.float().flatten() for sequence in sequences]
        else:
            print(f"  [{_ENGINE_LABELS[engine]}] {config}: missing {path} — skipping")
    return vectors


def _completion_starts(prompt_lengths: list[int], lengths: list[int]) -> list[int | None]:
    """The per-sequence index into the (length L−1) next-token log π vector where the completion begins.
    Vector position j is log P(token j+1 | ≤ j); token j+1 is in the completion iff j+1 ≥ prompt_length,
    i.e. j ≥ prompt_length − 1. Sequences with no completion token (start ≥ length) map to None (skipped)."""
    starts: list[int | None] = []
    for prompt_length, length in zip(prompt_lengths, lengths, strict=True):
        start = max(prompt_length - 1, 0)
        starts.append(start if start < length else None)
    return starts


def _completion_flat(sequences: list[torch.Tensor], starts: list[int | None]) -> torch.Tensor:
    """Concatenate the completion region of every (non-empty) sequence into one flat per-token vector."""
    return torch.cat(
        [sequence[start:] for sequence, start in zip(sequences, starts, strict=True) if start is not None]
    )


def _completion_means(sequences: list[torch.Tensor], starts: list[int | None]) -> torch.Tensor:
    """Per-sequence mean log π over the completion region — the length-normalized (GSPO) quantity."""
    return torch.tensor(
        [sequence[start:].mean() for sequence, start in zip(sequences, starts, strict=True) if start is not None]
    )


def _sequence_stats(per_sequence_delta: torch.Tensor) -> dict[str, float]:
    # Across-sequence distribution of the per-sequence mean δ (the length-normalized log importance ratio):
    # mean = systematic cross-engine bias, std = sequence-to-sequence spread, max = worst sequence.
    return {
        "n": per_sequence_delta.numel(),
        "mean": per_sequence_delta.mean().item(),
        "std": per_sequence_delta.std().item() if per_sequence_delta.numel() > 1 else 0.0,
        "rms": _rms(per_sequence_delta),
        "max": per_sequence_delta.abs().max().item(),
    }


def _print_delta_table(
    rows: list[tuple[str, str, str, dict[str, float]]], completion_tokens: int, epsilon: float, label: str
) -> None:
    print(f"\n=== Cross-engine log π gap{f' [{label}]' if label else ''} (δ = A − B, nats, per token) ===")
    print(f"(Σδ summed over all {completion_tokens} completion tokens; clip = fraction with |δ| > {epsilon})")
    print("(head = fp32 upcast vs body-dtype; production has vLLM head fp32, so the relevant")
    print(" mismatch is a body-dtype head on the trainer side against vLLM's fp32 head)")
    header = (
        f"{'group':<11} {'A − B':<22} {'A head':>7} {'B head':>7} {'mean δ':>10} {'RMS δ':>9}"
        f" {'max|δ|':>9} {'Σδ (seq)':>10} {'clip%':>7}"
    )
    print(header)
    print("-" * len(header))
    for group, engine_a, engine_b, head_a, head_b, stats in rows:
        pair = f"{_ENGINE_LABELS[engine_a]} − {_ENGINE_LABELS[engine_b]}"
        print(
            f"{group:<11} {pair:<22} {head_a:>7} {head_b:>7} {stats['mean']:>+10.4f} {stats['rms']:>9.4f}"
            f" {stats['max']:>9.4f} {stats['sum']:>+10.2f} {stats['clip'] * 100:>6.2f}%"
        )


def _print_decomposition_table(rows: list[tuple[str, str, str, dict[str, float]]], label: str) -> None:
    print(
        f"\n=== Error-correlation decomposition{f' [{label}]' if label else ''}"
        " (e = low-precision − fp32, nats) ==="
    )
    print("(δ = floor + (e_A − e_B); ρ = corr(e_A, e_B): ρ→1 errors cancel, ρ→0 add in quadrature)")
    header = (
        f"{'prec':<5} {'A − B':<22} {'A head':>7} {'B head':>7} {'ρ(err)':>8} {'RMS e_A':>9} {'RMS e_B':>9}"
        f" {'RMS(e_A−e_B)':>13} {'RMS floor':>10} {'slope':>8}"
    )
    print(header)
    print("-" * len(header))
    for precision, engine_a, engine_b, head_a, head_b, stats in rows:
        pair = f"{_ENGINE_LABELS[engine_a]} − {_ENGINE_LABELS[engine_b]}"
        print(
            f"{precision:<5} {pair:<22} {head_a:>7} {head_b:>7} {stats['rho']:>8.4f} {stats['rms_a']:>9.4f}"
            f" {stats['rms_b']:>9.4f} {stats['rms_diff']:>13.4f} {stats['rms_floor']:>10.4f} {stats['slope']:>+8.4f}"
        )


def _print_sequence_table(
    rows: list[tuple[str, str, str, str, str, dict[str, float]]], num_sequences: int, label: str
) -> None:
    print(f"\n=== Per-sequence (length-normalized / GSPO) log π gap{f' [{label}]' if label else ''}, nats ===")
    print(f"(per-sequence mean δ over each sequence's completion tokens, across {num_sequences} sequences;")
    print(" mean = systematic cross-engine bias, std = sequence-to-sequence spread, max = worst sequence)")
    header = f"{'group':<11} {'A − B':<22} {'A head':>7} {'B head':>7} {'mean':>10} {'std':>9} {'RMS':>9} {'max':>9}"
    print(header)
    print("-" * len(header))
    for group, engine_a, engine_b, head_a, head_b, stats in rows:
        pair = f"{_ENGINE_LABELS[engine_a]} − {_ENGINE_LABELS[engine_b]}"
        print(
            f"{group:<11} {pair:<22} {head_a:>7} {head_b:>7} {stats['mean']:>+10.5f} {stats['std']:>9.5f}"
            f" {stats['rms']:>9.5f} {stats['max']:>9.5f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast-llm-dir", default=None, help="Dir with Fast-LLM logprobs_<variant>.pt files.")
    parser.add_argument("--deepspeed-dir", default=None, help="Dir with DeepSpeed logprobs_<variant>.pt files.")
    parser.add_argument("--vllm-dir", default=None, help="Dir with vLLM logprobs_<variant>.pt files.")
    parser.add_argument(
        "--epsilon", type=float, default=0.2, help="PPO clip band for the clip-fraction column (default 0.2)."
    )
    parser.add_argument("--label", default="", help="Regime label for the table headers (e.g. 'text', 'random').")
    parser.add_argument(
        "--inputs-file",
        default=None,
        help="Path to the multi-sequence inputs.pt (dict with 'prompt_lengths'). When set, δ statistics are"
        " computed over the completion region of each sequence and an extra per-sequence (length-normalized"
        " / GSPO) table reports the across-sequence distribution of the per-sequence mean δ. Without it,"
        " the whole vector is treated as a single sequence's completion (the single-sequence default).",
    )
    args = parser.parse_args()

    dirs = {"fast_llm": args.fast_llm_dir, "deepspeed": args.deepspeed_dir, "vllm": args.vllm_dir}
    print("Loading per-engine log π vectors:")
    engines: dict[str, dict[str, list[torch.Tensor]]] = {}
    for engine in _ENGINE_ORDER:
        if dirs[engine] is not None:
            engines[engine] = _load_engine(engine, pathlib.Path(dirs[engine]))

    length_signatures = {
        tuple(vector.numel() for vector in sequences) for configs in engines.values() for sequences in configs.values()
    }
    if len(length_signatures) > 1:
        raise ValueError("Per-sequence log π lengths differ across engines/configs — inputs not aligned.")
    lengths = list(next(iter(length_signatures))) if length_signatures else []
    num_sequences = len(lengths)

    if args.inputs_file is not None:
        prompt_lengths = list(torch.load(args.inputs_file, map_location="cpu")["prompt_lengths"])
        if len(prompt_lengths) != num_sequences:
            raise ValueError(f"inputs-file has {len(prompt_lengths)} prompt lengths but {num_sequences} sequences.")
    else:
        # No prompt boundary: score the whole vector (start 0).
        prompt_lengths = [1] * num_sequences
    starts = _completion_starts(prompt_lengths, lengths)

    # Reduce each (engine, config) to a flat completion-token vector (for the per-token δ / decomposition
    # tables) and a per-sequence completion-mean vector (for the length-normalized per-sequence table).
    flat: dict[str, dict[str, torch.Tensor]] = {
        engine: {config: _completion_flat(sequences, starts) for config, sequences in configs.items()}
        for engine, configs in engines.items()
    }
    per_sequence_mean: dict[str, dict[str, torch.Tensor]] = {
        engine: {config: _completion_means(sequences, starts) for config, sequences in configs.items()}
        for engine, configs in engines.items()
    }
    completion_tokens = next((vector.numel() for configs in flat.values() for vector in configs.values()), 0)
    scored_sequences = sum(start is not None for start in starts)
    available = [engine for engine in _ENGINE_ORDER if engine in engines]
    pairs = list(itertools.combinations(available, 2))

    # δ table: fp32 floor, then per precision the full 2×2 of head choice (fp32 vs body-dtype) on each
    # side, over every available pair. The decomposition mirrors each precision/head/pair combination.
    # The per-sequence rows mirror the same group/precision/head/pair structure.
    delta_rows: list[tuple[str, str, str, str, str, dict[str, float]]] = []
    sequence_rows: list[tuple[str, str, str, str, str, dict[str, float]]] = []
    for engine_a, engine_b in pairs:
        if "fp32" in flat[engine_a] and "fp32" in flat[engine_b]:
            delta_rows.append(
                (
                    "fp32 floor",
                    engine_a,
                    engine_b,
                    "fp32",
                    "fp32",
                    _delta_stats(flat[engine_a]["fp32"], flat[engine_b]["fp32"], args.epsilon),
                )
            )
            sequence_rows.append(
                (
                    "fp32 floor",
                    engine_a,
                    engine_b,
                    "fp32",
                    "fp32",
                    _sequence_stats(per_sequence_mean[engine_a]["fp32"] - per_sequence_mean[engine_b]["fp32"]),
                )
            )

    decomposition_rows: list[tuple[str, str, str, str, str, dict[str, float]]] = []
    for precision in _PRECISIONS:
        for head_a, head_b in itertools.product(_HEADS, repeat=2):
            config_a, config_b = _head_config(precision, head_a), _head_config(precision, head_b)
            label_a, label_b = _head_label(precision, head_a), _head_label(precision, head_b)
            for engine_a, engine_b in pairs:
                if config_a not in flat[engine_a] or config_b not in flat[engine_b]:
                    continue
                delta_rows.append(
                    (
                        precision,
                        engine_a,
                        engine_b,
                        label_a,
                        label_b,
                        _delta_stats(flat[engine_a][config_a], flat[engine_b][config_b], args.epsilon),
                    )
                )
                sequence_rows.append(
                    (
                        precision,
                        engine_a,
                        engine_b,
                        label_a,
                        label_b,
                        _sequence_stats(per_sequence_mean[engine_a][config_a] - per_sequence_mean[engine_b][config_b]),
                    )
                )
                if "fp32" not in flat[engine_a] or "fp32" not in flat[engine_b]:
                    continue
                error_a = flat[engine_a][config_a] - flat[engine_a]["fp32"]
                error_b = flat[engine_b][config_b] - flat[engine_b]["fp32"]
                floor = flat[engine_a]["fp32"] - flat[engine_b]["fp32"]
                decomposition_rows.append(
                    (
                        precision,
                        engine_a,
                        engine_b,
                        label_a,
                        label_b,
                        {
                            "rho": _corr(error_a, error_b),
                            "rms_a": _rms(error_a),
                            "rms_b": _rms(error_b),
                            "rms_diff": _rms(error_a - error_b),
                            "rms_floor": _rms(floor),
                            "slope": _slope(flat[engine_b][config_b], flat[engine_a][config_a]),
                        },
                    )
                )

    _print_delta_table(delta_rows, completion_tokens, args.epsilon, args.label)
    _print_decomposition_table(decomposition_rows, args.label)
    if num_sequences > 1:
        _print_sequence_table(sequence_rows, scored_sequences, args.label)


if __name__ == "__main__":
    main()
