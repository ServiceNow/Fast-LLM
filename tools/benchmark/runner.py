"""
Core benchmarking infrastructure for Fast-LLM Triton kernels.

Each benchmark file defines a list of `Case` objects (input shape/dtype
sweep) and a list of `Variant` objects (implementations to compare — e.g.
pytorch eager, pytorch compiled, Triton). The runner invokes each variant
on each case, measures timing (median + mean + percentiles via CUDA events),
measures peak/final memory, and compares outputs against an fp32 reference
using RMS error. Results are printed as a table per case.
"""

import dataclasses
import gc
import math
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch

from fast_llm.utils import header
from tools.benchmark.gpu_specs import GpuSpec, detect_gpu_spec

# Before each timed CUDA-graph-backed call we must mark a new step so the graph
# system knows the previous rep's output buffers are no longer live. Without
# this, max-autotune compiled functions raise "tensor output of CUDAGraphs has
# been overwritten by a subsequent run" on the second call.
_cudagraph_mark_step_begin: Callable[[], None] | None = getattr(
    getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None
)


def _guarded(fn: Callable[[], Any]) -> Callable[[], Any]:
    """Wrap fn so cudagraph_mark_step_begin() is called before each invocation.
    This tells the CUDA graph system that previous outputs are no longer live
    and can be overwritten, preventing 'overwritten by subsequent run' errors
    when a max-autotune compiled function is called more than once.
    When CUDA graphs are not in use the wrapper is a no-op pass-through."""
    if _cudagraph_mark_step_begin is None:
        return fn

    def _wrapped() -> Any:
        _cudagraph_mark_step_begin()
        return fn()

    return _wrapped


Inputs = dict[str, Any]
VariantFn = Callable[[Inputs], Any]


@dataclasses.dataclass
class Variant:
    """A single implementation being compared. Provide `fwd` for forward-only
    timing. Provide `fwd_bwd` for forward+backward timing; when both are set,
    backward-only time is reported as `fwd_bwd - fwd`."""

    name: str
    fwd: VariantFn | None = None
    fwd_bwd: VariantFn | None = None
    # The fp32 reference variant. Exactly one per benchmark; its outputs are
    # the ground truth for RMS-error comparison.
    is_reference: bool = False


@dataclasses.dataclass
class Case:
    """A single input configuration for the kernel under test. `make_inputs`
    builds fresh input tensors on demand. It is called once per variant per
    mode, after a global seed reset, so every variant sees identical inputs."""

    name: str
    make_inputs: Callable[[], Inputs]
    # Minimum bytes read+written by the op. Used for GB/s + %BW. Optional.
    expected_bytes: int | None = None
    # Minimum floating-point ops performed by the op. Used for TFLOP/s + %FLOPs. Optional.
    expected_flops: int | None = None
    # For %FLOPs: which peak column to use (dtype of the hot inputs).
    compute_dtype: torch.dtype | None = None


def _seeded_inputs(case: Case, seed: int = 0) -> Inputs:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return case.make_inputs()


@dataclasses.dataclass
class TimingStats:
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    n_reps: int


@dataclasses.dataclass
class MemoryStats:
    peak_mib: float
    final_mib: float
    delta_peak_mib: float


@dataclasses.dataclass
class VariantResult:
    variant_name: str
    fwd_timing: TimingStats | None = None
    fwd_bwd_timing: TimingStats | None = None
    memory: MemoryStats | None = None
    rms_errors: dict[str, float] | None = None  # output name → RMS rel error vs reference
    error: str | None = None  # If the variant failed, the error message


# --------------------------------------------------------------------------- timing


def _make_cache_flusher(size_bytes: int = 256 * 1024 * 1024) -> Callable[[], None]:
    """Allocate a scratch buffer larger than any GPU L2 and zero it between reps
    to invalidate cached values (avoids over-optimistic timings)."""
    if not torch.cuda.is_available():
        return lambda: None
    buffer = torch.empty(size_bytes // 4, dtype=torch.int32, device="cuda")

    def flush() -> None:
        buffer.zero_()

    return flush


def bench_fn(
    fn: Callable[[], Any],
    warmup_ms: float = 25.0,
    rep_ms: float = 100.0,
    min_reps: int = 5,
    max_reps: int = 10_000,
) -> TimingStats:
    """Benchmark `fn` — it should be a no-arg callable that invokes the kernel
    being timed (close over inputs). Returns timing statistics in ms.

    Mirrors `triton.testing.do_bench` logic but returns raw per-rep list so we
    can compute {median, mean, min, max, std} from one set of runs.
    """
    if not torch.cuda.is_available():
        # CPU / Triton interpret: single timed run with wall clock.
        fn()  # warmup
        start = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return TimingStats(elapsed_ms, elapsed_ms, elapsed_ms, elapsed_ms, 0.0, 1)

    flush = _make_cache_flusher()

    # Warmup to JIT-compile, autotune, etc.
    torch.cuda.synchronize()
    warmup_start = torch.cuda.Event(enable_timing=True)
    warmup_end = torch.cuda.Event(enable_timing=True)
    warmup_start.record()
    fn()
    warmup_end.record()
    torch.cuda.synchronize()
    one_rep_ms = warmup_start.elapsed_time(warmup_end)

    # Additional warmup to stabilize (covers autotune misses on first call)
    n_warmup = max(1, int(warmup_ms / max(one_rep_ms, 0.01)))
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # Re-estimate after warmup (autotune usually settles to a faster config).
    post_start = torch.cuda.Event(enable_timing=True)
    post_end = torch.cuda.Event(enable_timing=True)
    post_start.record()
    fn()
    post_end.record()
    torch.cuda.synchronize()
    one_rep_ms = max(post_start.elapsed_time(post_end), 0.001)

    n_reps = max(min_reps, min(max_reps, int(rep_ms / one_rep_ms)))

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_reps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_reps)]
    for i in range(n_reps):
        flush()
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [start_events[i].elapsed_time(end_events[i]) for i in range(n_reps)]
    return TimingStats(
        median_ms=statistics.median(times),
        mean_ms=statistics.fmean(times),
        min_ms=min(times),
        max_ms=max(times),
        std_ms=statistics.pstdev(times) if len(times) > 1 else 0.0,
        n_reps=n_reps,
    )


# --------------------------------------------------------------------------- memory


def measure_memory(fn: Callable[[], Any]) -> MemoryStats:
    """Run `fn` once and capture peak and final device memory. Must be called
    on a fresh GPU state (the caller resets stats before constructing inputs)."""
    if not torch.cuda.is_available():
        return MemoryStats(0.0, 0.0, 0.0)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    final = torch.cuda.memory_allocated()
    # Hold onto the result until after the measurement so it stays in `final`.
    del result
    return MemoryStats(
        peak_mib=peak / 1024 / 1024,
        final_mib=final / 1024 / 1024,
        delta_peak_mib=(peak - baseline) / 1024 / 1024,
    )


# --------------------------------------------------------------------------- correctness


def rms_relative_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """Root-mean-squared error of `candidate - reference`, normalized by the
    RMS of `reference`. Both are cast to fp32 before comparison."""
    cand = candidate.detach().float()
    ref = reference.detach().float()
    diff_rms = (cand - ref).pow(2).mean().sqrt().item()
    ref_rms = ref.pow(2).mean().sqrt().item()
    return diff_rms / max(ref_rms, 1e-30)


def _as_output_dict(output: Any) -> dict[str, torch.Tensor]:
    """Normalize a variant's output into a {name: tensor} dict for comparison."""
    if isinstance(output, torch.Tensor):
        return {"out": output}
    if isinstance(output, dict):
        return {k: v for k, v in output.items() if isinstance(v, torch.Tensor)}
    if isinstance(output, (tuple, list)):
        return {f"out{i}": v for i, v in enumerate(output) if isinstance(v, torch.Tensor)}
    raise TypeError(f"Cannot extract tensors from variant output of type {type(output).__name__}")


# --------------------------------------------------------------------------- runner


def _run_one_variant(
    variant: Variant,
    case: Case,
    reference_outputs: dict[str, dict[str, torch.Tensor]] | None,
    warmup_ms: float,
    rep_ms: float,
) -> VariantResult:
    result = VariantResult(variant_name=variant.name)
    try:
        # --- correctness + memory: one fresh invocation per mode
        # fwd mode
        if variant.fwd is not None:
            inputs = _seeded_inputs(case)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            def _fwd_once() -> Any:
                return variant.fwd(inputs)

            _guarded_fwd = _guarded(_fwd_once)

            # First: correctness. Run once, capture output for comparison.
            fwd_output = _guarded_fwd()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if reference_outputs is not None and not variant.is_reference:
                ref_fwd = reference_outputs.get("fwd")
                if ref_fwd is not None:
                    cand = _as_output_dict(fwd_output)
                    rms = {name: rms_relative_error(cand[name], ref_fwd[name]) for name in ref_fwd if name in cand}
                    result.rms_errors = (result.rms_errors or {}) | {f"fwd.{k}": v for k, v in rms.items()}
            del fwd_output

            # Timing: reuse the same input tensors, fn closes over them.
            result.fwd_timing = bench_fn(_guarded_fwd, warmup_ms=warmup_ms, rep_ms=rep_ms)
            del inputs

        # fwd+bwd mode
        if variant.fwd_bwd is not None:
            inputs = _seeded_inputs(case)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            def _fwd_bwd_once() -> Any:
                return variant.fwd_bwd(inputs)

            _guarded_fwd_bwd = _guarded(_fwd_bwd_once)

            fwd_bwd_output = _guarded_fwd_bwd()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if reference_outputs is not None and not variant.is_reference:
                ref_fb = reference_outputs.get("fwd_bwd")
                if ref_fb is not None:
                    cand = _as_output_dict(fwd_bwd_output)
                    rms = {name: rms_relative_error(cand[name], ref_fb[name]) for name in ref_fb if name in cand}
                    result.rms_errors = (result.rms_errors or {}) | {f"fb.{k}": v for k, v in rms.items()}
            del fwd_bwd_output

            # Memory measurement: one fresh call on fresh inputs.
            fresh_inputs = _seeded_inputs(case)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result.memory = measure_memory(_guarded(lambda: variant.fwd_bwd(fresh_inputs)))
            del fresh_inputs

            # Timing.
            result.fwd_bwd_timing = bench_fn(_guarded_fwd_bwd, warmup_ms=warmup_ms, rep_ms=rep_ms)
            del inputs
        elif variant.fwd is not None and result.memory is None:
            # No backward — measure fwd-mode memory.
            fresh_inputs = _seeded_inputs(case)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result.memory = measure_memory(_guarded(lambda: variant.fwd(fresh_inputs)))
            del fresh_inputs
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def _collect_reference_outputs(
    variant: Variant,
    case: Case,
) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    if variant.fwd is not None:
        out["fwd"] = _as_output_dict(variant.fwd(_seeded_inputs(case)))
    if variant.fwd_bwd is not None:
        out["fwd_bwd"] = _as_output_dict(variant.fwd_bwd(_seeded_inputs(case)))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Detach+clone to guard against in-place mutation by later variants
    return {mode: {k: v.detach().clone() for k, v in tensors.items()} for mode, tensors in out.items()}


# --------------------------------------------------------------------------- table


def _column_decimals(values: list[float | None]) -> int:
    """Number of decimal places to give the smallest non-zero value in a column
    at least 4 significant digits. Capped at 6 so one tiny value doesn't bloat
    the whole column (e.g. 0.00001 alongside 100 would otherwise force 8 decimals)."""
    nonzero = [abs(v) for v in values if v is not None and v != 0]
    if not nonzero:
        return 0
    min_magnitude = math.floor(math.log10(min(nonzero)))
    return min(6, max(0, 3 - min_magnitude))


def _format_aligned(values: list[float | None]) -> list[str]:
    """Format a column with the same number of decimals for every entry, so
    decimal points line up. Zeros get the trailing zeros too (e.g. '0.0000')."""
    decimals = _column_decimals(values)
    out: list[str] = []
    for value in values:
        if value is None:
            out.append("—")
        else:
            out.append(f"{value:.{decimals}f}")
    return out


def _pick_unit(max_value: float, table: list[tuple[str, float]]) -> tuple[str, float]:
    """Given a magnitude-ordered list of (unit_label, scale_to_unit) pairs and
    the column's max absolute value, return the unit where max_value*scale is
    in [1, 1000) when possible. `table` must be ordered by ascending magnitude
    (largest unit last)."""
    chosen_label, chosen_scale = table[0]
    for label, scale in table:
        if max_value * scale >= 1:
            chosen_label, chosen_scale = label, scale
        else:
            break
    return chosen_label, chosen_scale


# Each table is ordered ascending (small unit → large unit). `scale` converts
# from the canonical storage unit (ms / bytes-per-second / flops-per-second /
# MiB) into the display unit.
_TIME_UNITS = [("ns", 1e6), ("us", 1e3), ("ms", 1.0), ("s", 1e-3)]
_BANDWIDTH_UNITS = [("B/s", 1.0), ("KB/s", 1e-3), ("MB/s", 1e-6), ("GB/s", 1e-9), ("TB/s", 1e-12)]
_FLOPS_UNITS = [
    ("FLOP/s", 1.0),
    ("KFLOP/s", 1e-3),
    ("MFLOP/s", 1e-6),
    ("GFLOP/s", 1e-9),
    ("TFLOP/s", 1e-12),
    ("PFLOP/s", 1e-15),
]
_MEMORY_UNITS = [("KiB", 1024.0), ("MiB", 1.0), ("GiB", 1 / 1024), ("TiB", 1 / 1024 / 1024)]


def _unit_column(
    prefix: str, canonical_values: list[float | None], units: list[tuple[str, float]]
) -> tuple[str, list[str]]:
    """Pick the best display unit for a column's magnitude and format with
    aligned decimals. Header is '<prefix> <unit>'."""
    non_none = [abs(v) for v in canonical_values if v is not None]
    max_value = max(non_none, default=0.0)
    if max_value > 0:
        label, scale = _pick_unit(max_value, units)
    else:
        # All values are zero / None. Fall back to the canonical unit (scale=1.0)
        # so e.g. memory defaults to MiB rather than the middle of the table.
        label, scale = next(((l, s) for (l, s) in units if s == 1.0), units[0])
    scaled = [v * scale if v is not None else None for v in canonical_values]
    header = f"{prefix} {label}" if prefix else label
    return header, _format_aligned(scaled)


def _percent_column(values: list[float | None]) -> list[str]:
    """Format a column of ratios as aligned percentages."""
    scaled = [v * 100 if v is not None else None for v in values]
    formatted = _format_aligned(scaled)
    return [f if f == "—" else f"{f}%" for f in formatted]


def _rms_column(values: list[float | None]) -> list[str]:
    """Align RMS errors in scientific notation with a shared exponent-free width."""
    decimals = 3  # 4 sig figs
    out: list[str] = []
    for value in values:
        if value is None:
            out.append("—")
        elif value == 0.0:
            out.append(f"{0.0:.{decimals}e}")
        else:
            out.append(f"{value:.{decimals}e}")
    return out


def _simplify_rms_key(key: str, all_keys: list[str]) -> str:
    """Turn internal keys like 'fwd.out' / 'fb.loss' into concise display labels.

    Rules:
    - strip the mode prefix ('fwd.'/'fb.') when all keys share the same mode
    - rename 'fb' → 'bwd' for display when it survives
    - drop the trailing '.out' / standalone 'out' (the placeholder key used
      when a variant returns a single unnamed tensor)
    """
    mode, _, tensor = key.partition(".")
    all_modes = {k.partition(".")[0] for k in all_keys}
    if len(all_modes) <= 1:
        remainder = tensor
    else:
        pretty = "bwd" if mode == "fb" else mode
        remainder = f"{pretty}.{tensor}" if tensor else pretty
    if remainder == "out":
        return ""
    return remainder.removesuffix(".out")


def _rms_header(key: str, all_keys: list[str]) -> str:
    simplified = _simplify_rms_key(key, all_keys)
    return f"rel_rms({simplified})" if simplified else "rel_rms"


def _render_table(
    case: Case,
    results: list[VariantResult],
    gpu_spec: GpuSpec | None,
    has_fwd: bool,
    has_fwd_bwd: bool,
    rms_keys: list[str],
    verbose: bool,
) -> str:
    # First column header carries the case name so the per-case label and the
    # variant-name column are merged into one (avoids a redundant title line).
    columns: list[tuple[str, list[str]]] = [(case.name, [r.variant_name for r in results])]

    def _add(header: str, values: list[str]) -> None:
        columns.append((header, values))

    if has_fwd:
        _add(*_unit_column("fwd", [r.fwd_timing.median_ms if r.fwd_timing else None for r in results], _TIME_UNITS))
        if verbose:
            _add(
                *_unit_column(
                    "fwd mean", [r.fwd_timing.mean_ms if r.fwd_timing else None for r in results], _TIME_UNITS
                )
            )
            _add(
                *_unit_column("fwd min", [r.fwd_timing.min_ms if r.fwd_timing else None for r in results], _TIME_UNITS)
            )
            _add(
                *_unit_column("fwd max", [r.fwd_timing.max_ms if r.fwd_timing else None for r in results], _TIME_UNITS)
            )

    if has_fwd_bwd:
        # Backward-only derived: fwd+bwd − fwd.
        bwd_values: list[float | None] = []
        total_values: list[float | None] = []
        for r in results:
            if r.fwd_bwd_timing is None:
                bwd_values.append(None)
                total_values.append(None)
                continue
            total = r.fwd_bwd_timing.median_ms
            bwd_values.append(total - r.fwd_timing.median_ms if r.fwd_timing else None)
            total_values.append(total)
        _add(*_unit_column("bwd", bwd_values, _TIME_UNITS))
        _add(*_unit_column("total", total_values, _TIME_UNITS))

    def _time_for_throughput(r: VariantResult) -> float | None:
        if r.fwd_bwd_timing is not None:
            return r.fwd_bwd_timing.median_ms
        if r.fwd_timing is not None:
            return r.fwd_timing.median_ms
        return None

    if case.expected_bytes is not None:
        bandwidths: list[float | None] = []
        for r in results:
            t_ms = _time_for_throughput(r)
            bandwidths.append(case.expected_bytes / (t_ms / 1000) if t_ms is not None else None)
        header, values = _unit_column("", bandwidths, _BANDWIDTH_UNITS)
        _add(header, values)
        if gpu_spec is not None:
            peak_bytes_per_s = gpu_spec.peak_bandwidth_gbps * 1e9
            pct = [bw / peak_bytes_per_s if bw is not None else None for bw in bandwidths]
            _add("%BW", _percent_column(pct))

    if case.expected_flops is not None:
        flop_rates: list[float | None] = []
        for r in results:
            t_ms = _time_for_throughput(r)
            flop_rates.append(case.expected_flops / (t_ms / 1000) if t_ms is not None else None)
        header, values = _unit_column("", flop_rates, _FLOPS_UNITS)
        _add(header, values)
        peak_tflops = gpu_spec.peak_tflops(case.compute_dtype) if gpu_spec and case.compute_dtype else None
        if peak_tflops is not None:
            peak_flops_per_s = peak_tflops * 1e12
            pct = [fr / peak_flops_per_s if fr is not None else None for fr in flop_rates]
            _add("%FLOPs", _percent_column(pct))

    peak_mib = [r.memory.peak_mib if r.memory else None for r in results]
    delta_mib = [r.memory.delta_peak_mib if r.memory else None for r in results]
    _add(*_unit_column("peak", peak_mib, _MEMORY_UNITS))
    _add(*_unit_column("Δpeak", delta_mib, _MEMORY_UNITS))

    for key in rms_keys:
        _add(_rms_header(key, rms_keys), _rms_column([(r.rms_errors or {}).get(key) for r in results]))

    _add("error", [r.error or "" for r in results])
    # Drop the error column if nothing failed
    if not any(r.error for r in results):
        columns.pop()

    widths = [max(len(header), *(len(v) for v in values)) for header, values in columns]
    sep = "  "

    # First column (case name + variant names) is text — left-justify. All other
    # columns are numeric — right-justify so decimal points line up across rows.
    def _justify(text: str, width: int, column_index: int) -> str:
        return text.ljust(width) if column_index == 0 else text.rjust(width)

    header_line = sep.join(_justify(h, w, i) for i, ((h, _), w) in enumerate(zip(columns, widths)))
    divider = sep.join("-" * w for w in widths)
    body_lines = []
    for row in range(len(results)):
        body_lines.append(
            sep.join(_justify(values[row], w, i) for i, ((_, values), w) in enumerate(zip(columns, widths)))
        )
    return "\n".join([header_line, divider, *body_lines])


# --------------------------------------------------------------------------- orchestration


def run_benchmark(
    benchmark_name: str,
    cases: list[Case],
    variants: list[Variant],
    *,
    warmup_ms: float = 25.0,
    rep_ms: float = 100.0,
    verbose: bool = False,
    print_fn: Callable[[str], None] = print,
) -> list[tuple[Case, list[VariantResult]]]:
    """Run all (case, variant) combinations and print one table per case.

    Exactly one variant should have `is_reference=True` — its outputs are the
    ground truth for RMS-error comparisons. That variant should compute in
    fp32, eager, using the most straightforward reference implementation."""
    reference = [v for v in variants if v.is_reference]
    if len(reference) != 1:
        raise ValueError(
            f"Expected exactly one reference variant (is_reference=True), got {len(reference)}. "
            f"Variants: {[v.name for v in variants]}"
        )
    gpu_spec = detect_gpu_spec()
    print_fn(header(benchmark_name))
    if gpu_spec is not None:
        print_fn(f"gpu: {gpu_spec.name} (peak BW {gpu_spec.peak_bandwidth_gbps:.0f} GB/s)")
    else:
        print_fn("gpu: unknown (no %-of-peak columns)")
    print_fn("")

    all_results: list[tuple[Case, list[VariantResult]]] = []
    for case in cases:
        ref_outputs = _collect_reference_outputs(reference[0], case)

        results = []
        has_fwd = False
        has_fwd_bwd = False
        rms_keys_seen: list[str] = []
        for variant in variants:
            r = _run_one_variant(variant, case, ref_outputs, warmup_ms=warmup_ms, rep_ms=rep_ms)
            results.append(r)
            has_fwd = has_fwd or r.fwd_timing is not None
            has_fwd_bwd = has_fwd_bwd or r.fwd_bwd_timing is not None
            for k in r.rms_errors or {}:
                if k not in rms_keys_seen:
                    rms_keys_seen.append(k)

        print_fn(
            _render_table(
                case,
                results,
                gpu_spec=gpu_spec,
                has_fwd=has_fwd,
                has_fwd_bwd=has_fwd_bwd,
                rms_keys=rms_keys_seen,
                verbose=verbose,
            )
        )
        print_fn("")
        all_results.append((case, results))
    return all_results
