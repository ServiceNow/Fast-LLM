import contextlib
import enum
import logging
import traceback
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.engine.config_utils.run import get_run
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class ProfileType(str, enum.Enum):
    cpu = "cpu"
    cuda = "cuda"


class NoProfiler(contextlib.nullcontext):
    def step(self):
        pass


@config_class()
class ProfilingConfig(Config):
    cpu: bool = Field(default=False, desc="Profile the CUDA operations on the CPU side.", hint=FieldHint.feature)
    cuda: bool = Field(default=False, desc="Profile the CUDA operations on the CPU side.", hint=FieldHint.core)
    skip: int = Field(
        default=1,
        desc="Skip this many iterations before starting the profiler for the first time.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 0),
    )
    # Skip on every cycle (profiler disabled)
    wait: int = Field(
        default=0,
        desc="Wait this many iterations before each profiling cycle.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 0),
    )
    # Warmup on every cycle (profiler enabled, results ignored)
    warmup: int = Field(
        default=1,
        desc="Warmup the profiler for this many iterations before each profiling cycle, i.e., enable the profiler but discard the results.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 0),
    )
    # Profile on every cycle (profiler enabled, results kept)
    cycles: int = Field(
        default=1,
        desc="Profile this many iterations in each profiling cycle.",
        hint=FieldHint.optional,
        valid=check_field(Assert.gt, 0),
    )
    averages: bool = Field(
        default=False,
        desc="Log a table of average and total properties for each CUDA operation.",
        hint=FieldHint.logging,
    )
    trace: bool = Field(
        default=False, desc="Log a table of every CUDA operation in chronological order.", hint=FieldHint.logging
    )
    table_width: int = Field(
        default=80,
        desc="Target width for logged tables, in characters.",
        hint=FieldHint.logging,
        valid=check_field(Assert.geq, 40),
    )
    # The ranks to profile (all by default)
    ranks: set[int] = Field(default_factory=set, desc="Profile only on the specified ranks.", hint=FieldHint.feature)
    # Print the profile table(s), otherwise save to file.
    log: bool = Field(
        default=False,
        desc="Log the profile tables to stdout, otherwise save them as artifacts.",
        hint=FieldHint.logging,
    )
    # Export for chrome/tensorboard
    export: bool = Field(
        default=False,
        desc="Export the raw profile as an artifact in chrome trace format.",
        doc="The profile is saved to profile_chrome_step_{step}. "
        "It can be load with Google chrome (by typing `chrome://tracing/` in the address bar) or with tensorboard.",
        hint=FieldHint.logging,
    )

    def _validate(self):
        if isinstance(self.ranks, str):
            # This happens with yaml serialization
            Assert.eq(self.ranks, "set()")
            self.global_attention_layers = set()
        profile_ranks = set(self.ranks or [])
        Assert.eq(len(profile_ranks), len(self.ranks or []))
        self.ranks = profile_ranks  # noqa

    def get_profiler(
        self, *, distributed_config: DistributedConfig | None = None, start_step: int = 0
    ) -> typing.Union["torch.profiler.profile", NoProfiler]:
        import torch

        activities = ([torch.profiler.ProfilerActivity.CPU] if self.cpu else []) + (
            [torch.profiler.ProfilerActivity.CUDA] if self.cuda else []
        )
        if (
            not activities
            or not (self.averages or self.trace or self.export)
            or not (distributed_config is None or not self.ranks or distributed_config.rank in self.ranks)
        ):
            return NoProfiler()
        schedule = torch.profiler.schedule(
            skip_first=self.skip,
            warmup=self.warmup,
            wait=self.wait,
            active=self.cycles,
        )
        return torch.profiler.profile(
            schedule=schedule,
            activities=activities,
            on_trace_ready=get_trace_fn(self, start_step),
            with_modules=True,
        )


def get_trace_fn(config: ProfilingConfig, start_step: int = 0):
    config.validate()

    def trace_fn(
        profiler: "torch.profiler.profile",
    ):
        run = get_run()

        try:
            step = start_step + profiler.step_num
            f"self_{'cuda' if config.cuda else 'cpu'}_time_total"
            if config.trace:
                table = build_trace_table(
                    profiler,
                    cuda=config.cuda,
                    cpu=config.cpu,
                    column_width=config.table_width,
                    header=f"Trace for step {step}",
                )
                if config.log:
                    logger.info(table)
                else:
                    run.open_artifact(f"profile_trace_step_{step}").write(table)

            if config.averages:
                table = build_average_table(
                    profiler,
                    cuda=config.cuda,
                    cpu=config.cpu,
                    column_width=config.table_width,
                    header=f"Averages for step {step}",
                )
                if config.log:
                    logger.info(table)
                else:
                    run.open_artifact(f"profile_averages_step_{step}").write(table)

            if config.export:
                profiler.export_chrome_trace(str(run.open_artifact(f"profile_chrome_step_{step}", mode=None)))

            # Store results for future use.
            profiler.bc_profile_result = profiler.profiler.function_events
        except Exception:
            # Pytorch explodes without showing the error.
            traceback.print_exc()
            raise

    return trace_fn


_COLUMN_HEADERS = {
    "name": "Name",
    "cpu_self": "Self CPU",
    "cpu_self_percent": "Self CPU %",
    "cpu_total": "CPU total",
    "cpu_total_percent": "CPU total %",
    "cpu_avg": "CPU time avg",
    "cuda": "Self CUDA",
    "cuda_percent": "Self CUDA %",
    "cuda_total": "CUDA total",
    "cuda_avg": "CUDA time avg",
    "start_time": "Start",
    "end_time": "End",
    "cpu_mem": "CPU Mem",
    "cpu_mem_self": "Self CPU Mem",
    "cuda_mem": "CUDA Mem",
    "cuda_mem_self": "Self CUDA Mem",
    "calls": "# of Calls",
    "input_shapes": "Input Shapes",
    "source_loc": "Source Location",
    "node_id": "Node ID",
    "total_flops": "Total xflops",
}

_CPU_TRACE_COLUMNS = {"name", "cpu_self", "cpu_total", "start_time", "end_time"}
_CUDA_TRACE_COLUMNS = {"name", "cuda", "start_time", "end_time"}

_CPU_AVERAGES_COLUMNS = {"name", "cpu_self", "cpu_self_percent", "cpu_total", "cpu_total_percent", "cpu_avg", "calls"}
_CUDA_AVERAGES_COLUMNS = {"name", "cuda", "cuda_percent", "cuda_avg", "calls"}

_MISC_CUDA_OPS = (
    {
        "cuLaunchKernel",
        "INVALID",
        "cudaLaunchKernel",
        "cudaMemcpyAsync",
        "cudaStreamIsCapturing",
        "cudaDeviceGetAttribute",
        "cudaFuncSetAttribute",
        "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
        "cudaMemsetAsync",
    },
)


def build_trace_table(
    profiler: "torch.profiler.profile", *, cuda: bool = True, cpu: bool = False, column_width=80, header="Trace"
):
    var_name = f"self_{'cuda' if cuda else 'cpu'}_time_total"
    events = [evt for evt in profiler.profiler.function_events if getattr(evt, var_name) > 0]
    return _build_table(
        events,
        (_CPU_TRACE_COLUMNS if cpu else set()) | (_CUDA_TRACE_COLUMNS if cuda else set()),
        name_column_width=column_width,
        filter_by=None if cuda and cpu else var_name,
        header=header,
    )


def build_average_table(
    profiler: "torch.profiler.profile", *, cuda: bool = True, cpu: bool = False, column_width=80, header="Averages"
):
    var_name = f"self_{'cuda' if cuda else 'cpu'}_time_total"
    return _build_table(
        profiler.key_averages(),
        (_CPU_AVERAGES_COLUMNS if cpu else set()) | (_CUDA_AVERAGES_COLUMNS if cuda else set()),
        name_column_width=column_width,
        sort_by=var_name,
        header=header,
    )


def _build_table(
    events,
    columns: set[str],
    sort_by=None,
    header=None,
    row_limit=None,
    name_column_width=80,
    top_level_events_only=False,
    col_width=12,
    spacing_size=2,
    exclude=None,
    filter_by=None,
):
    """Similar to the pytorch method, but more configurable."""
    if sort_by is not None:
        events = sorted(events, key=lambda evt: getattr(evt, sort_by), reverse=True)
    if filter_by is not None:
        events = [evt for evt in events if getattr(evt, filter_by) > 0]
    if exclude is not None:
        events = [evt for evt in events if evt.key not in exclude]
    if top_level_events_only:
        events = [evt for evt in events if evt.cpu_parent is None]
    if row_limit is not None:
        events = events[:row_limit]
    if len(events) == 0:
        return ""

    row_format = ""
    header_sep = ""
    line_length = -spacing_size

    name_column_width = min(max([len(evt.key) for evt in events]) + 4, name_column_width)

    for i, _ in enumerate(columns):
        width = name_column_width if i == 0 else col_width
        row_format += "{: " + ">" + str(width) + "}" + (" " * spacing_size)
        header_sep += "-" * width + (" " * spacing_size)
        line_length += width + spacing_size

    result = []

    sum_self_cpu_time_total = sum(event.self_cpu_time_total for event in events)
    sum_self_cuda_time_total = sum(event.self_cuda_time_total for event in events)  # if evt.device_type == DeviceType.

    if header is not None:
        result.extend(["=" * line_length, header])
    if top_level_events_only:
        result.extend(["=" * line_length, "This report only display top-level ops statistics"])

    result.extend(
        [
            header_sep,
            row_format.format(*[header for col, header in _COLUMN_HEADERS.items() if col in columns]),
            header_sep,
        ]
    )

    for evt in events:
        row_values = []
        if "name" in columns:
            row_values.append(_format_name(evt.key, name_column_width))
        if "cpu_self" in columns:
            row_values.append(_format_time_us(evt.self_cpu_time_total))
        if "cpu_self_percent" in columns:
            row_values.append(_format_time_share(evt.self_cpu_time_total, sum_self_cpu_time_total))
        if "cpu_total" in columns:
            row_values.append(_format_time_us(evt.cpu_time_total))
        if "cpu_total_percent" in columns:
            row_values.append(
                _format_time_share(evt.cpu_time_total, sum_self_cpu_time_total) if not evt.is_async else 0
            )
        if "cpu_avg" in columns:
            row_values.append(_format_time_us(evt.cpu_time))
        if "cuda" in columns:
            row_values.append(_format_time_us(evt.self_cuda_time_total))
        if "cuda_percent" in columns:
            row_values.append(_format_time_share(evt.self_cuda_time_total, sum_self_cuda_time_total))
        if "cuda_total" in columns:
            row_values.append(_format_time_us(evt.cuda_time_total))
        if "cuda_avg" in columns:
            row_values.append(_format_time_us(evt.cuda_time))
        if "start_time" in columns:
            row_values.append(_format_time_us(evt.time_range.start))
        if "end_time" in columns:
            row_values.append(_format_time_us(evt.time_range.end))
        if "calls" in columns:
            row_values.append(evt.count)

        result.append(row_format.format(*row_values))

    result.append(header_sep)
    if sum_self_cpu_time_total > 0:
        result.append(f"CPU time total: {_format_time_ms(sum_self_cpu_time_total)}")
    if sum_self_cuda_time_total > 0:
        result.append(f"CUDA time total: {_format_time_ms(sum_self_cuda_time_total)}")
    result.append("")
    return "\n".join(result)


def _format_name(name, max_width):
    return name[: (max_width - 3)] + "..." if len(name) >= max_width - 3 else name


def _format_time_us(time_us):
    return f"{time_us:,.0f} us"


def _format_time_ms(time_us):
    return f"{time_us/1e3:,.3f} ms"


def _format_time_share(time_us, total_time_us):
    """Defines how to format time in FunctionEvent"""
    if total_time_us == 0:
        return "NaN"
    return f"{time_us * 100 / total_time_us:.2f} %"
