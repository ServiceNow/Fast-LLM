import dataclasses
import enum
import typing
import warnings

import torch

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, test_field
from fast_llm.distributed import DistributedConfig
from fast_llm.utils import Assert, div


class StepType(str, enum.Enum):
    forward = "forward"
    backward = "backward"


@config_class()
class BatchConfig(Config):
    micro_batch_size: int = Field(
        default=None,
        desc="Size of individual micro-batches, in samples. May be derived or constrained be other quantities.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    depth_first_micro_batches: int = Field(
        default=None,
        desc="Size of individual micro-batches. May be derived or constrained be other quantities.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    breadth_first_micro_batches: int = Field(
        default=None,
        desc="Size of individual micro-batches. May be derived or constrained be other quantities.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    sequential_micro_batches: int = Field(
        default=None,
        desc="Total number of sequential micro-batches. May be derived or constrained be other quantities (= depth-first * breadth-first).",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    batch_size: int = Field(
        default=None,
        desc="Global batch size, in samples. May be derived or constrained be other quantities (= micro-batch size * sequential micro-batches * batch-data-parallel).",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    sequence_length: int = Field(
        default=2048,
        desc="Number of tokens in a sample.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    micro_sequence_length: int = Field(
        default=None,
        desc="Number of tokens in a micro-sequence (must divide the sequence length).",
        hint=FieldHint.performance,
        valid=check_field(Assert.gt, 0),
    )
    num_micro_sequences: int = Field(
        init=False,
        desc="Number of micro-sequences to split each sample (= seqence length / micro-sequence length).",
        hint=FieldHint.derived,
        valid=check_field(Assert.gt, 0),
    )
    _distributed: DistributedConfig = Field(
        init=False,
        desc="Pointer to a distributed configuration, required to know the data-parallel split of the batch.",
        hint=FieldHint.setup,
    )

    def setup(self, distributed_config: DistributedConfig):
        self._distributed = distributed_config

    @property
    def num_inputs(self):
        return self.sequential_micro_batches * self.num_micro_sequences

    @property
    def _is_setup(self):
        return hasattr(self, "_distributed")

    def _validate(self):
        # Use the distributed properties to determine the batch size and its breakdown.
        # Requires post-processed distributed config args
        if self.batch_size is None or self.micro_batch_size is None:
            if self.depth_first_micro_batches is None:
                self.depth_first_micro_batches = 1
            if self.breadth_first_micro_batches is None:
                self.breadth_first_micro_batches = 1
            self.sequential_micro_batches = self.depth_first_micro_batches * self.breadth_first_micro_batches
            if self.batch_size is None:
                if self.micro_batch_size is None:
                    self.micro_batch_size = 1
                self.batch_size = (
                    self.micro_batch_size * self.sequential_micro_batches * self._distributed.batch_data_parallel
                )
            elif self.micro_batch_size is None:
                self.micro_batch_size = div(
                    self.batch_size, self.sequential_micro_batches * self._distributed.batch_data_parallel
                )
        else:
            self.sequential_micro_batches = div(
                self.batch_size, self.micro_batch_size * self._distributed.batch_data_parallel
            )
            if self.depth_first_micro_batches is None:
                if self.breadth_first_micro_batches is None:
                    if self._distributed.pipeline_parallel > 1:
                        self.depth_first_micro_batches = 1
                        self.breadth_first_micro_batches = self.sequential_micro_batches
                    else:
                        self.depth_first_micro_batches = self.sequential_micro_batches
                        self.breadth_first_micro_batches = 1
                else:
                    self.depth_first_micro_batches = div(
                        self.sequential_micro_batches, self.breadth_first_micro_batches
                    )
            elif self.breadth_first_micro_batches is None:
                self.breadth_first_micro_batches = div(self.sequential_micro_batches, self.depth_first_micro_batches)
            else:
                Assert.eq(
                    self.sequential_micro_batches, self.breadth_first_micro_batches * self.depth_first_micro_batches
                )

        if self._distributed.pipeline_parallel > 1 and self.depth_first_micro_batches > 1:
            raise NotImplementedError("Depth-first pipeline parallelism not yet implemented")
        if self.depth_first_micro_batches > 1 and self.breadth_first_micro_batches > 1:
            warnings.warn(
                "Mixing of breadth-first and depth-first gradient accumulation is not thoroughly tested."
                " Use at your own risk."
            )
        if self.micro_sequence_length is None:
            self.micro_sequence_length = self.sequence_length
        self.num_micro_sequences = div(self.sequence_length, self.micro_sequence_length)
        super()._validate()


@config_class()
class ScheduleConfig(Config):
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


@dataclasses.dataclass()
class Step:
    config: BatchConfig
    # The step type (forward or backward).
    type_: StepType
    # Index of the stage to be processed.
    stage: int
    # Data index (combines micro-batch and micro-sequence)
    data_index: int
    pipeline_rank: int = 0
    # Estimated relative duration of the step.
    duration: float = 1.0
    # Estimated begin time of the step
    start: float | None = None
    # Estimated end time of the compute part of the step
    compute_end: float | None = None
    # Estimated true end time of the step, including send
    end: float | None = None
    # Index of the step.
    local_index: int | None = None
    global_index: int | None = None
    reduce: bool = False
    reduce_accumulate: bool = False
    # Related steps
    next_step: typing.Optional["Step"] = None
    prev_step: typing.Optional["Step"] = None
    forward_step: typing.Optional["Step"] = None
    backward_step: typing.Optional["Step"] = None
    # The step in which the restore op for this layer is launched.
    restore_step: typing.Optional["Step"] = None
    # A cuda event for the restore operation, used for stream synchronization.
    restore_event: typing.Optional[torch.cuda.Event] = None
    # The step that launches the recv for this step.
    recv_step: typing.Optional["Step"] = None
    # A cuda event for the recv operation, used for stream synchronization.
    recv_event: torch.cuda.Event | None = None
    # The layer input (forward) or output gradients (backward)
    recv_launch: list["Step"] = dataclasses.field(default_factory=list)
    # The `recv_launch` step associated to this step send.
    send_to: typing.Optional["Step"] = None
    # List of steps with other.restore_step==self.step
    restore_launch: list["Step"] = dataclasses.field(default_factory=list)
    # Synchronize with that step's throttle event before running this step.
    throttle_step: typing.Optional["Step"] = None
    # Event for cpu throttling.
    throttle_event: torch.cuda.Event | None = None
    # Input and output meta.
    meta_input: torch.Tensor | None = None
    meta_output: torch.Tensor | None = None
    meta_kwargs: dict | None = None

    @property
    def micro_sequence(self):
        return self.data_index % self.config.num_micro_sequences

    @property
    def micro_batch(self):
        return self.data_index // self.config.num_micro_sequences

    @property
    def depth_first_micro_batch(self):
        return self.micro_batch % self.config.depth_first_micro_batches

    @property
    def breadth_first_micro_batch(self):
        return self.micro_batch // self.config.depth_first_micro_batches

    @property
    def map_index(self):
        return (
            self.type_,
            self.stage,
            self.data_index,
        )

    def __repr__(self):
        misc = ""
        if self.start is not None and self.compute_end is not None:
            misc += f", time = ({self.start}, {self.compute_end})"
        if self.restore_step:
            misc += f", restore={self.restore_step.global_index}"
        if self.recv_launch:
            misc += f", recv={[step.global_index for step in self.recv_launch]}"
        if self.reduce:
            misc += f", reduce"
        return (
            f"Step(idx={self.global_index},"
            f" local_idx={self.local_index},"
            f" stage={self.stage}{'f' if self.type_ == StepType.forward else 'b'},"
            f" dfmb={self.depth_first_micro_batch}, bfmb={self.breadth_first_micro_batch},"
            f" ms={self.micro_sequence}{misc})"
        )

    def get_stage_index(self, num_stages):
        return self.stage if self.type_ == StepType.forward else 2 * num_stages - 1 - self.stage


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
