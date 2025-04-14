import enum
import warnings

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, test_field
from fast_llm.engine.distributed.config import DistributedConfig
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
    cross_document_attention: bool = Field(
        default=True,
        desc="Applies attention to tokens from other documents in the packed sequence. Set to False for masking attention to other documents.",
        hint=FieldHint.feature,
    )
    _distributed: DistributedConfig = Field(
        init=False,
        desc="Pointer to a distributed configuration, required to know the data-parallel split of the batch.",
        hint=FieldHint.setup,
    )

    def setup(self, distributed_config: DistributedConfig) -> None:
        self._distributed = distributed_config

    @property
    def num_inputs(self) -> int:
        return self.sequential_micro_batches * self.num_micro_sequences

    def _validate(self) -> None:
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
            with self._set_implicit_default():
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
