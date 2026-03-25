import enum
import functools

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, test_field
from fast_llm.utils import Assert


class StepType(str, enum.Enum):
    forward = "forward"
    backward = "backward"


@config_class()
class ScheduleConfig(Config):
    depth_first_micro_batches: int = Field(
        default=1,
        desc="Size of individual micro-batches. May be derived or constrained be other quantities.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    breadth_first_micro_batches: int = Field(
        default=1,
        desc="Size of individual micro-batches. May be derived or constrained be other quantities.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    micro_batch_splits: int = Field(
        default=1,
        desc="Number of splits for each micro-batch.",
        hint=FieldHint.performance,
        valid=check_field(Assert.gt, 0),
    )
    pipeline_overlap: bool = Field(
        default=True, desc="Overlap the pipeline-parallel network communication.", hint=FieldHint.testing
    )
    data_overlap: bool = Field(
        default=True, desc="Overlap the data-parallel network communication.", hint=FieldHint.testing
    )
    data_batch_warn_time_ms: float = Field(
        default=1000.0,
        desc="Warn if a batch takes too long to load.",
        hint=FieldHint.optional,
        valid=check_field(Assert.gt, 0),
    )
    log_data_pipeline: bool = Field(
        default=False,
        desc="Write per-micro-batch timing to data_pipeline_log/rank_{rank}.jsonl for pipeline diagnostics.",
        hint=FieldHint.optional,
    )
    # Enable cpu throttling to avoid lag spikes, see https://arxiv.org/pdf/2211.05953.pdf, appendix D.2.
    throttle_cpu: bool = Field(
        default=True,
        desc="Avoid scheduling too many operations in advance to limit memory fragmentation and prevent costly memory cache flushes.",
        hint=FieldHint.expert,
    )
    # Throttle every n steps
    throttle_cpu_rate: int = Field(
        default=1,
        desc="Number of schedule steps between each cpu throttling.",
        hint=FieldHint.expert,
        valid=check_field(Assert.gt, 0),
    )
    # Wait on m steps earlier.
    throttle_cpu_delay: int = Field(
        default=1,
        desc="Synchronize with a cuda event registered this many steps before, to avoid a full synchronization.",
        hint=FieldHint.expert,
        valid=check_field(Assert.geq, 0),
    )
    debug_schedule: bool = Field(default=False, desc="Log the whole schedule.", hint=FieldHint.logging)
    debug_send_recv: bool = Field(default=False, desc="Log the pipeline-parallel operations.", hint=FieldHint.logging)
    profile_schedule: bool = Field(
        default=False,
        desc="Detailed time table for the schedule execution (cpu and gpu times).",
        hint=FieldHint.logging,
    )
    # TODO: Remove
    estimate_critical_batch: bool = Field(
        default=False, desc="No longer supported.", hint=FieldHint.deprecated, valid=test_field(lambda x: not x)
    )
    # Skip the weight update and related ops (debug)
    skip_step: bool = Field(
        default=False,
        desc="Skip the weight update during training steps. Still runs the forward and backward passes.",
        hint=FieldHint.testing,
    )

    @functools.cached_property
    def sequential_micro_batches(self) -> int:
        return self.breadth_first_micro_batches * self.depth_first_micro_batches

    @functools.cached_property
    def num_inputs(self) -> int:
        return self.sequential_micro_batches * self.micro_batch_splits


class StreamType(str, enum.Enum):
    compute = "compute"
    data = "data"
    pipeline = "pipeline"


class StepScheduleType(str, enum.Enum):
    breadth_first = "breadth_first"
    depth_first = "depth_first"


class EventType(str, enum.Enum):
    # Global events
    batch_begin = "batch_begin"
    batch_end = "batch_end"
    get_batch = "get_batch"
    pre_restore = "pre_restore"
    post_reduce = "post_reduce"
    optimizer = "optimizer"
    # Step events (compute stream)
    run = "run"
    compute_wait_data = "compute_wait_data"
    compute_wait_pipe = "compute_wait_pipe"
    # Step events (data stream)
    restore = "restore"
    reduce = "reduce"
    data_wait_compute = "data_wait_compute"
    # Step events (pipeline stream)
    send = "send"
    recv = "recv"
    pipe_wait_compute = "pipe_wait_compute"


class MockStream:
    stream_id: int = 0

    def wait_stream(self, stream):
        pass

    def __eq__(self, other):
        return isinstance(other, MockStream)


class MockEvent:
    def record(self, stream=None):
        pass

    def wait(self):
        pass
