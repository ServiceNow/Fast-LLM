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

  1. the δ distribution (mean = systematic bias, RMS, max, per-sequence sum, PPO clip fraction) for
     the fp32 floor, the matched production config (bf16 body + fp32 head on both sides), and the
     mismatched config (vLLM's as-shipped bf16 head vs the trainer's fp32 head);
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

# Canonical cross-engine configs, mapped to each within-engine tool's variant names.
_ENGINE_VARIANTS: dict[str, dict[str, str]] = {
    "fast_llm": {"fp32": "reference", "bf16_fp32head": "bf16_fp32_lm_head", "bf16_bf16head": "bf16"},
    "deepspeed": {"fp32": "fp32", "bf16_fp32head": "bf16", "bf16_bf16head": "bf16_head_bf16"},
    "vllm": {"fp32": "fp32", "bf16_fp32head": "bf16_fp32_head", "bf16_bf16head": "bf16"},
}
_ENGINE_ORDER = ("fast_llm", "deepspeed", "vllm")
_ENGINE_LABELS = {"fast_llm": "Fast-LLM", "deepspeed": "DeepSpeed", "vllm": "vLLM"}


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


def _load_engine(engine: str, directory: pathlib.Path) -> dict[str, torch.Tensor]:
    vectors: dict[str, torch.Tensor] = {}
    for config, variant in _ENGINE_VARIANTS[engine].items():
        path = directory / f"logprobs_{variant}.pt"
        if path.exists():
            vectors[config] = torch.load(path, map_location="cpu").float().flatten()
        else:
            print(f"  [{_ENGINE_LABELS[engine]}] {config}: missing {path} — skipping")
    return vectors


def _error_vector(vectors: dict[str, torch.Tensor], engine: str, mode: str) -> torch.Tensor | None:
    # bf16 rounding error against this engine's own fp32. In the mismatched config only vLLM keeps a
    # bf16 head; the trainers always use the fp32 head.
    config = "bf16_bf16head" if (mode == "mismatched" and engine == "vllm") else "bf16_fp32head"
    if config not in vectors or "fp32" not in vectors:
        return None
    return vectors[config] - vectors["fp32"]


def _pair_config(engine: str, mode: str) -> str:
    if mode == "fp32":
        return "fp32"
    if mode == "matched":
        return "bf16_fp32head"
    return "bf16_bf16head" if engine == "vllm" else "bf16_fp32head"


def _print_delta_table(
    rows: list[tuple[str, str, str, dict[str, float]]], sequence_length: int, epsilon: float, label: str
) -> None:
    print(f"\n=== Cross-engine log π gap{f' [{label}]' if label else ''} (δ = A − B, nats) ===")
    print(f"(per-sequence sum over {sequence_length} tokens; clip = fraction with |δ| > {epsilon})")
    header = f"{'group':<22} {'A − B':<22} {'mean δ':>10} {'RMS δ':>9} {'max|δ|':>9} {'Σδ (seq)':>10} {'clip%':>7}"
    print(header)
    print("-" * len(header))
    for group, engine_a, engine_b, stats in rows:
        pair = f"{_ENGINE_LABELS[engine_a]} − {_ENGINE_LABELS[engine_b]}"
        print(
            f"{group:<22} {pair:<22} {stats['mean']:>+10.4f} {stats['rms']:>9.4f} {stats['max']:>9.4f}"
            f" {stats['sum']:>+10.2f} {stats['clip'] * 100:>6.2f}%"
        )


def _print_decomposition_table(rows: list[tuple[str, str, str, dict[str, float]]], label: str) -> None:
    print(f"\n=== Error-correlation decomposition{f' [{label}]' if label else ''} (e = bf16 − fp32, nats) ===")
    print("(δ = floor + (e_A − e_B); ρ = corr(e_A, e_B): ρ→1 errors cancel, ρ→0 add in quadrature)")
    header = (
        f"{'config':<12} {'A − B':<22} {'ρ(err)':>8} {'RMS e_A':>9} {'RMS e_B':>9}"
        f" {'RMS(e_A−e_B)':>13} {'RMS floor':>10} {'slope':>8}"
    )
    print(header)
    print("-" * len(header))
    for config, engine_a, engine_b, stats in rows:
        pair = f"{_ENGINE_LABELS[engine_a]} − {_ENGINE_LABELS[engine_b]}"
        print(
            f"{config:<12} {pair:<22} {stats['rho']:>8.4f} {stats['rms_a']:>9.4f} {stats['rms_b']:>9.4f}"
            f" {stats['rms_diff']:>13.4f} {stats['rms_floor']:>10.4f} {stats['slope']:>+8.4f}"
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
    args = parser.parse_args()

    dirs = {"fast_llm": args.fast_llm_dir, "deepspeed": args.deepspeed_dir, "vllm": args.vllm_dir}
    print("Loading per-engine log π vectors:")
    engines: dict[str, dict[str, torch.Tensor]] = {}
    for engine in _ENGINE_ORDER:
        if dirs[engine] is not None:
            engines[engine] = _load_engine(engine, pathlib.Path(dirs[engine]))

    lengths = {vector.numel() for vectors in engines.values() for vector in vectors.values()}
    if len(lengths) > 1:
        raise ValueError(f"Per-token log π vectors have mismatched lengths {sorted(lengths)} — inputs not aligned.")
    sequence_length = next(iter(lengths)) if lengths else 0
    available = [engine for engine in _ENGINE_ORDER if engine in engines]
    pairs = list(itertools.combinations(available, 2))

    delta_rows: list[tuple[str, str, str, dict[str, float]]] = []
    for mode, group_label in (("fp32", "fp32 floor"), ("matched", "matched (fp32 head)")):
        for engine_a, engine_b in pairs:
            config_a, config_b = _pair_config(engine_a, mode), _pair_config(engine_b, mode)
            if config_a in engines[engine_a] and config_b in engines[engine_b]:
                stats = _delta_stats(engines[engine_a][config_a], engines[engine_b][config_b], args.epsilon)
                delta_rows.append((group_label, engine_a, engine_b, stats))
    for engine_a, engine_b in pairs:
        if "vllm" not in (engine_a, engine_b):
            continue
        config_a, config_b = _pair_config(engine_a, "mismatched"), _pair_config(engine_b, "mismatched")
        if config_a in engines[engine_a] and config_b in engines[engine_b]:
            stats = _delta_stats(engines[engine_a][config_a], engines[engine_b][config_b], args.epsilon)
            delta_rows.append(("mismatched (vLLM bf16 head)", engine_a, engine_b, stats))

    decomposition_rows: list[tuple[str, str, str, dict[str, float]]] = []
    for mode in ("matched", "mismatched"):
        for engine_a, engine_b in pairs:
            if mode == "mismatched" and "vllm" not in (engine_a, engine_b):
                continue
            error_a = _error_vector(engines[engine_a], engine_a, mode)
            error_b = _error_vector(engines[engine_b], engine_b, mode)
            if error_a is None or error_b is None:
                continue
            floor = engines[engine_a]["fp32"] - engines[engine_b]["fp32"]
            config_a, config_b = _pair_config(engine_a, mode), _pair_config(engine_b, mode)
            stats = {
                "rho": _corr(error_a, error_b),
                "rms_a": _rms(error_a),
                "rms_b": _rms(error_b),
                "rms_diff": _rms(error_a - error_b),
                "rms_floor": _rms(floor),
                "slope": _slope(engines[engine_b][config_b], engines[engine_a][config_a]),
            }
            decomposition_rows.append((mode, engine_a, engine_b, stats))

    _print_delta_table(delta_rows, sequence_length, args.epsilon, args.label)
    _print_decomposition_table(decomposition_rows, args.label)


if __name__ == "__main__":
    main()
