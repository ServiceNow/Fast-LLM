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
import math
import statistics
import time
import typing

import torch

from fast_llm.utils import header
from tools.benchmark.triton_kernels.gpu_specs import GpuSpec, detect_gpu_spec

# Before each timed CUDA-graph-backed call we must mark a new step so the graph
# system knows the previous rep's output buffers are no longer live. Without
# this, max-autotune compiled functions raise "tensor output of CUDAGraphs has
# been overwritten by a subsequent run" on the second call.
_cudagraph_mark_step_begin: typing.Callable[[], None] | None = getattr(
    getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None
)


def _guarded(fn: typing.Callable[[], typing.Any]) -> typing.Callable[[], typing.Any]:
    if _cudagraph_mark_step_begin is None:
        return fn

    def _wrapped() -> typing.Any:
        _cudagraph_mark_step_begin()
        return fn()

    return _wrapped


Inputs = dict[str, typing.Any]
VariantFn = typing.Callable[[Inputs], typing.Any]


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
    # Applied to the output dict during the correctness check only — never during
    # timing. Receives {name: tensor} and the full inputs dict; returns the
    # (possibly modified) dict. Use this to mask out don't-care regions so they
    # don't inflate RMS errors (e.g. uninitialized phantom rows in sparse buffers).
    output_postprocess: typing.Callable[[dict[str, torch.Tensor], Inputs], dict[str, torch.Tensor]] | None = None
    # Called between timing reps (outside the timed region) to restore any
    # input tensors the variant mutates in-place. Use this instead of cloning
    # inside the timed callable so the mutation cost is not measured.
    reset_inputs: typing.Callable[[Inputs], typing.Any] | None = None


class Case:
    """Base for a single input configuration. Subclasses are dataclasses holding
    the kernel's shape parameters (e.g. rows, cols, dtype) and override `name`
    and `make_inputs`; the throughput properties are optional."""

    @property
    def name(self) -> str:
        raise NotImplementedError()

    # Optional — defaults skip the corresponding columns.
    @property
    def expected_bytes(self) -> int | None:
        """Bytes read+written; enables GB/s + %BW columns."""
        return None

    @property
    def expected_flops(self) -> int | None:
        """FLOPs performed; enables TFLOP/s + %FLOPs columns."""
        return None

    @property
    def compute_dtype(self) -> torch.dtype | None:
        """Dtype of hot inputs; picks the peak column for the %FLOPs computation."""
        return None

    def make_inputs(self, device: torch.device) -> Inputs:
        """Return a fresh dict of input tensors on `device`. Called once per
        variant per mode, after a global seed reset, so every variant sees
        identical inputs."""
        raise NotImplementedError()


class DtypedCase(Case):
    """Subclasses must declare a `dtype: torch.dtype` field. Provides
    `compute_dtype` automatically."""

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.dtype  # type: ignore[attr-defined]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seeded_inputs(case: Case) -> Inputs:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    return case.make_inputs(_device())


@dataclasses.dataclass
class TimingStats:
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    num_reps: int


@dataclasses.dataclass
class MemoryStats:
    # Both above the pre-call baseline, so input tensors don't inflate the numbers.
    # `peak` is the transient maximum during fn (matters when memory is tight);
    # `final` is what remains allocated after fn returns (output + saved-for-bwd
    # tensors + .grad on inputs) — usually the more interesting of the two.
    peak_mib: float
    final_mib: float


@dataclasses.dataclass
class VariantResult:
    variant_name: str
    fwd_timing: TimingStats | None = None
    fwd_bwd_timing: TimingStats | None = None
    memory: MemoryStats | None = None
    rms_errors: dict[str, float] | None = None  # output name → RMS rel error vs reference
    error: str | None = None  # If the variant failed, the error message


# --------------------------------------------------------------------------- timing


def bench_fn(
    fn: typing.Callable[[], typing.Any],
    reset: typing.Callable[[], None] | None = None,
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
        # CPU / Triton interpret: single timed run with wall clock. min_reps,
        # max_reps, warmup_ms, rep_ms are ignored — this path is for smoke
        # testing kernel correctness, not measurement.
        if reset is not None:
            reset()
        fn()  # warmup
        if reset is not None:
            reset()
        start = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return TimingStats(elapsed_ms, elapsed_ms, elapsed_ms, elapsed_ms, 0.0, 1)

    # Scratch buffer larger than any GPU L2; zeroed between reps to flush
    # cached values (avoids over-optimistic timings).
    flush_buffer = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device="cuda")

    # Warmup to JIT-compile, autotune, etc.
    torch.cuda.synchronize()
    warmup_start = torch.cuda.Event(enable_timing=True)
    warmup_end = torch.cuda.Event(enable_timing=True)
    if reset is not None:
        reset()
    warmup_start.record()
    fn()
    warmup_end.record()
    torch.cuda.synchronize()
    one_rep_ms = warmup_start.elapsed_time(warmup_end)

    # Additional warmup to stabilize (covers autotune misses on first call)
    num_warmup = max(1, int(warmup_ms / max(one_rep_ms, 0.01)))
    for _ in range(num_warmup):
        if reset is not None:
            reset()
        fn()
    torch.cuda.synchronize()

    # Re-estimate after warmup (autotune usually settles to a faster config).
    post_start = torch.cuda.Event(enable_timing=True)
    post_end = torch.cuda.Event(enable_timing=True)
    if reset is not None:
        reset()
    post_start.record()
    fn()
    post_end.record()
    torch.cuda.synchronize()
    one_rep_ms = max(post_start.elapsed_time(post_end), 0.001)

    num_reps = max(min_reps, min(max_reps, int(rep_ms / one_rep_ms)))

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_reps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_reps)]
    for i in range(num_reps):
        if reset is not None:
            reset()
        # The L2 flush is enqueued before start_events[i] on the same stream, so
        # the timed window starts after the zero completes — only fn() is timed.
        flush_buffer.zero_()
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_reps)]
    return TimingStats(
        median_ms=statistics.median(times),
        mean_ms=statistics.fmean(times),
        min_ms=min(times),
        max_ms=max(times),
        std_ms=statistics.pstdev(times) if len(times) > 1 else 0.0,
        num_reps=num_reps,
    )


# --------------------------------------------------------------------------- memory


def measure_memory(fn: typing.Callable[[], typing.Any]) -> MemoryStats:
    """Run `fn` once and capture peak and final device memory. Must be called
    on a fresh GPU state (the caller resets stats before constructing inputs)."""
    if not torch.cuda.is_available():
        return MemoryStats(0.0, 0.0)
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
        peak_mib=(peak - baseline) / 1024 / 1024,
        final_mib=(final - baseline) / 1024 / 1024,
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


def _as_output_dict(output: typing.Any) -> dict[str, torch.Tensor]:
    """Normalize a variant's output into a {name: tensor} dict for comparison."""
    if isinstance(output, torch.Tensor):
        return {"out": output}
    if isinstance(output, dict):
        return {k: v for k, v in output.items() if isinstance(v, torch.Tensor)}
    if isinstance(output, (tuple, list)):
        return {f"out{i}": v for i, v in enumerate(output) if isinstance(v, torch.Tensor)}
    raise TypeError(f"Cannot extract tensors from variant output of type {type(output).__name__}")


# --------------------------------------------------------------------------- runner


def _measure_mode(
    variant: Variant,
    case: Case,
    variant_fn: VariantFn,
    mode_label: str,
    reference_outputs: dict[str, torch.Tensor] | None,
    warmup_ms: float,
    rep_ms: float,
    min_reps: int,
) -> tuple[TimingStats, dict[str, float]]:
    """Run one mode (fwd or fwd_bwd) of one variant: correctness check against
    the reference, then timing. Returns (timing, rms_errors keyed by mode_label)."""
    inputs = _seeded_inputs(case)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    guarded = _guarded(lambda: variant_fn(inputs))

    output = guarded()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    rms_errors: dict[str, float] = {}
    if reference_outputs is not None and not variant.is_reference:
        cand = _as_output_dict(output)
        if variant.output_postprocess is not None:
            cand = variant.output_postprocess(cand, inputs)
        rms_errors = {
            f"{mode_label}.{name}": rms_relative_error(cand[name], reference_outputs[name])
            for name in reference_outputs
            if name in cand
        }
    del output

    reset = (lambda: variant.reset_inputs(inputs)) if variant.reset_inputs else None
    timing = bench_fn(guarded, reset=reset, warmup_ms=warmup_ms, rep_ms=rep_ms, min_reps=min_reps)
    return timing, rms_errors


def _run_one_variant(
    variant: Variant,
    case: Case,
    reference_outputs: dict[str, dict[str, torch.Tensor]] | None,
    warmup_ms: float,
    rep_ms: float,
    min_reps: int = 5,
) -> VariantResult:
    result = VariantResult(variant_name=variant.name)
    try:
        if variant.fwd is not None:
            ref_fwd = reference_outputs.get("fwd") if reference_outputs is not None else None
            result.fwd_timing, fwd_rms = _measure_mode(
                variant, case, variant.fwd, "fwd", ref_fwd, warmup_ms, rep_ms, min_reps
            )
            if fwd_rms:
                result.rms_errors = (result.rms_errors or {}) | fwd_rms

        if variant.fwd_bwd is not None:
            ref_fb = reference_outputs.get("fwd_bwd") if reference_outputs is not None else None
            result.fwd_bwd_timing, bwd_rms = _measure_mode(
                variant, case, variant.fwd_bwd, "bwd", ref_fb, warmup_ms, rep_ms, min_reps
            )
            if bwd_rms:
                result.rms_errors = (result.rms_errors or {}) | bwd_rms

        # Memory measurement: one fresh call on fresh inputs. fwd_bwd is preferred
        # since it captures saved-for-bwd tensors and .grad allocation.
        memory_fn = variant.fwd_bwd or variant.fwd
        if memory_fn is not None:
            fresh_inputs = _seeded_inputs(case)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result.memory = measure_memory(_guarded(lambda: memory_fn(fresh_inputs)))
            del fresh_inputs
    except (
        Exception
    ) as exc:  # noqa: BLE001 — variant failures are reported in the result column, not propagated, so a single broken kernel doesn't kill the rest of the sweep.
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def _collect_reference_outputs(
    variant: Variant,
    case: Case,
) -> dict[str, dict[str, torch.Tensor]]:
    # Reference outputs are taken raw — output_postprocess is only applied to
    # candidate variants. The reference is therefore expected to natively
    # produce zeros (or whatever value) in regions that output_postprocess
    # masks, so the comparison is symmetric. Sparse benches honor this by
    # zeroing padded/phantom rows in their loop reference.
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
_TIME_UNITS = [("ns", 1e6), ("μs", 1e3), ("ms", 1.0), ("s", 1e-3)]
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
    variable: str, canonical_values: list[float | None], units: list[tuple[str, float]]
) -> tuple[str, str, list[str]]:
    """Pick the best display unit for a column's magnitude and format with
    aligned decimals. Returns (variable, unit, values) for the 2-line header."""
    non_none = [abs(v) for v in canonical_values if v is not None]
    max_value = max(non_none, default=0.0)
    if max_value > 0:
        label, scale = _pick_unit(max_value, units)
    else:
        # All values are zero / None. Fall back to the canonical unit (scale=1.0)
        # so e.g. memory defaults to MiB rather than the middle of the table.
        label, scale = next(
            ((unit_label, unit_scale) for (unit_label, unit_scale) in units if unit_scale == 1.0), units[0]
        )
    scaled = [v * scale if v is not None else None for v in canonical_values]
    return variable, label, _format_aligned(scaled)


def _percent_column(values: list[float | None]) -> list[str]:
    """Format a column of ratios as aligned percentages."""
    scaled = [v * 100 if v is not None else None for v in values]
    formatted = _format_aligned(scaled)
    return [f if f == "—" else f"{f}%" for f in formatted]


def _rms_column(values: list[float | None]) -> list[str]:
    """Align RMS errors in scientific notation with a shared exponent-free width."""
    decimals = 3  # 4 sig figs
    return ["—" if value is None else f"{value:.{decimals}e}" for value in values]


def _rms_header(key: str, all_keys: list[str]) -> str:
    """Header for an `rms_errors` column. Strip the `<mode>.` prefix when every
    key shares the same mode (one-mode benches don't need to repeat 'fwd.')."""
    mode, _, tensor = key.partition(".")
    if len({k.partition(".")[0] for k in all_keys}) <= 1:
        return f"rel_rms({tensor})"
    return f"rel_rms({key})"


def _render_table(
    case: Case,
    results: list[VariantResult],
    gpu_spec: GpuSpec | None,
    has_fwd: bool,
    has_fwd_bwd: bool,
    rms_keys: list[str],
    verbose: bool,
) -> str:
    # First column carries the case name on the variable line; the variant-name
    # column lives on the unit line so the table needs no separate title row.
    columns: list[tuple[str, str, list[str]]] = [(case.name, "", [r.variant_name for r in results])]

    def _add(variable: str, unit: str, values: list[str]) -> None:
        columns.append((variable, unit, values))

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
        _add(*_unit_column("bw", bandwidths, _BANDWIDTH_UNITS))
        if gpu_spec is not None:
            peak_bytes_per_s = gpu_spec.peak_bandwidth_gbps * 1e9
            pct = [bw / peak_bytes_per_s if bw is not None else None for bw in bandwidths]
            _add("bw", "%peak", _percent_column(pct))

    if case.expected_flops is not None:
        flop_rates: list[float | None] = []
        for r in results:
            t_ms = _time_for_throughput(r)
            flop_rates.append(case.expected_flops / (t_ms / 1000) if t_ms is not None else None)
        _add(*_unit_column("flops", flop_rates, _FLOPS_UNITS))
        peak_tflops = gpu_spec.peak_tflops(case.compute_dtype) if gpu_spec and case.compute_dtype else None
        if peak_tflops is not None:
            peak_flops_per_s = peak_tflops * 1e12
            pct = [fr / peak_flops_per_s if fr is not None else None for fr in flop_rates]
            _add("flops", "%peak", _percent_column(pct))

    peak_mib = [r.memory.peak_mib if r.memory else None for r in results]
    final_mib = [r.memory.final_mib if r.memory else None for r in results]
    _add(*_unit_column("Δpeak", peak_mib, _MEMORY_UNITS))
    _add(*_unit_column("Δfinal", final_mib, _MEMORY_UNITS))

    for key in rms_keys:
        _add(_rms_header(key, rms_keys), "", _rms_column([(r.rms_errors or {}).get(key) for r in results]))

    _add("error", "", [r.error or "" for r in results])
    # Drop the error column if nothing failed
    if not any(r.error for r in results):
        columns.pop()

    widths = [max(len(variable), len(unit), *(len(v) for v in values)) for variable, unit, values in columns]
    separator = "  "

    # First column (case name + variant names) is text — left-justify. All other
    # columns are numeric — right-justify so decimal points line up across rows.
    def _justify(text: str, width: int, column_index: int) -> str:
        return text.ljust(width) if column_index == 0 else text.rjust(width)

    variable_line = separator.join(_justify(var, w, i) for i, ((var, _, _), w) in enumerate(zip(columns, widths)))
    unit_line = separator.join(_justify(unit, w, i) for i, ((_, unit, _), w) in enumerate(zip(columns, widths)))
    divider = separator.join("-" * w for w in widths)
    body_lines = []
    for row in range(len(results)):
        body_lines.append(
            separator.join(_justify(values[row], w, i) for i, ((_, _, values), w) in enumerate(zip(columns, widths)))
        )
    return "\n".join([variable_line, unit_line, divider, *body_lines])


# --------------------------------------------------------------------------- orchestration


def run_benchmark(
    benchmark_name: str,
    cases: list[Case],
    variants: list[Variant],
    *,
    warmup_ms: float = 25.0,
    rep_ms: float = 100.0,
    min_reps: int = 5,
    verbose: bool = False,
    print_fn: typing.Callable[[str], None] = print,
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
            r = _run_one_variant(variant, case, ref_outputs, warmup_ms=warmup_ms, rep_ms=rep_ms, min_reps=min_reps)
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
