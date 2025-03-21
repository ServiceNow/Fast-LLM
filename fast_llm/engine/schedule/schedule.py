import abc
import dataclasses
import logging
import typing
import warnings

import numpy as np
import torch
import torch.utils
import torch.utils.data

from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.multi_stage.multi_stage import MultiStageModel
from fast_llm.engine.schedule.config import BatchConfig, ScheduleConfig, StepType
from fast_llm.tensor import TensorMeta
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


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
    def micro_sequence(self) -> int:
        return self.data_index % self.config.num_micro_sequences

    @property
    def micro_batch(self) -> int:
        return self.data_index // self.config.num_micro_sequences

    @property
    def depth_first_micro_batch(self) -> int:
        return self.micro_batch % self.config.depth_first_micro_batches

    @property
    def breadth_first_micro_batch(self) -> int:
        return self.micro_batch // self.config.depth_first_micro_batches

    @property
    def map_index(self) -> tuple[StepType, int, int]:
        return (
            self.type_,
            self.stage,
            self.data_index,
        )

    def __repr__(self) -> str:
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

    def get_stage_index(self, num_stages) -> int:
        return self.stage if self.type_ == StepType.forward else 2 * num_stages - 1 - self.stage


class Schedule(abc.ABC):
    def __init__(
        self,
        multi_stage: MultiStageModel,
        batch_config: BatchConfig,
        schedule_config: ScheduleConfig,
        distributed_config: DistributedConfig,
        phase: PhaseType,
    ):
        self._multi_stage = multi_stage
        self._batch_config = batch_config
        self._schedule_config = schedule_config
        self._distributed = distributed_config
        self._num_stages = len(self._multi_stage.stages)
        self._phase = phase
        self._is_training = self._phase.is_training

        if self._batch_config.num_inputs < self._distributed.pipeline_parallel:
            warnings.warn("Not enough input to achieve true pipeline parallelism.")

        # Setup the activation metas.
        self._preprocessed_meta = self._multi_stage.base_model.preprocess_meta(
            self._batch_config,
            phase=self._phase,
        )

        self._steps, self._first_grad_stage = self._create_steps()

        self._create_index()

        self._setup_restore_steps(self._multi_stage.weight_buffer_indices)
        self._setup_reduce_steps(self._multi_stage.grad_buffer_indices)
        self._setup_timeline()
        self._setup_send_recv_steps()
        self._validate_send_recv_steps()
        self._setup_throttle_steps()
        self._setup_metas()

        if self._schedule_config.debug_schedule:
            logger.info(f"{self._phase.value} schedule:\n{self._steps}")

    @property
    def phase(self) -> PhaseType:
        return self._phase

    @property
    def batch_config(self) -> BatchConfig:
        return self._batch_config

    @property
    def preprocessed_meta(self) -> list[tuple[TensorMeta, dict]]:
        return self._preprocessed_meta

    def iterate(self, pipeline_rank: int | None = None) -> typing.Iterator[Step]:
        return iter(self._steps if pipeline_rank is None else self._device_steps[pipeline_rank])

    def __iter__(self) -> typing.Iterator[Step]:
        return self.iterate(self._distributed.pipeline_rank)

    def __repr__(self) -> str:
        return "Schedule with steps:\n" + "\n".join(
            [
                f"  rank {i}:\n    " + "\n    ".join([str(step) for step in device_step])
                for i, device_step in enumerate(self._device_steps)
            ]
        )

    def get_step(
        self,
        type_: StepType,
        stage: int,
        data_index: int,
    ) -> Step:
        return self._step_map[(type_, stage, data_index)]

    def _create_index(self) -> None:
        self._device_steps: list[list[Step]] = [[] for _ in range(self._distributed.pipeline_parallel)]
        self._step_map = {}
        for i, step in enumerate(self._steps):
            Assert.in_range(step.stage, 0, self._num_stages)
            Assert.in_range(
                step.data_index,
                0,
                self._batch_config.sequential_micro_batches * self._batch_config.num_micro_sequences,
            )
            Assert.incl(step.type_, (StepType.forward, StepType.backward))
            step.global_index = i
            # TODO: More configurable placement?
            step.pipeline_rank = step.stage % self._distributed.pipeline_parallel
            step.local_index = len(self._device_steps[step.pipeline_rank])
            self._device_steps[step.pipeline_rank].append(step)
            Assert.not_incl(map_index := step.map_index, self._step_map)
            self._step_map[map_index] = step

        # Make sure all devices do something.
        Assert.custom(all, self._device_steps)
        # Consistency checks
        step_map = self._step_map.copy()
        for data_index in range(self._batch_config.num_inputs):
            for type_ in (StepType.forward, StepType.backward):
                for stage in range(0 if type_ == StepType.forward else self._first_grad_stage, self._num_stages):
                    assert (
                        step_map.pop((type_, stage, data_index), None) is not None
                    ), f"Missing {type_.value} step with stage={stage}, data_index={data_index}"
        Assert.empty(step_map)

        # Related steps
        for i, step in enumerate(self._steps):
            if self._is_training:
                if step.type_ == StepType.forward:
                    if step.stage >= self._first_grad_stage:
                        step.backward_step = self.get_step(StepType.backward, *step.map_index[1:])
                else:
                    step.forward_step = self.get_step(StepType.forward, *step.map_index[1:])
            if step.type_ == StepType.forward and step.stage == 0:
                step.prev_step = None
            elif step.type_ == StepType.backward and step.stage == self._num_stages - 1:
                step.prev_step = self.get_step(StepType.forward, *step.map_index[1:])
            else:
                step.prev_step = self.get_step(
                    step.type_, step.stage + (1 if step.type_ == StepType.backward else -1), *step.map_index[2:]
                )

            if step.type_ == StepType.backward and step.stage == self._first_grad_stage:
                step.next_step = None
            elif step.type_ == StepType.forward and step.stage == self._num_stages - 1:
                step.next_step = self.get_step(StepType.backward, *step.map_index[1:]) if self._is_training else None
            else:
                step.next_step = self.get_step(
                    step.type_, step.stage + (1 if step.type_ == StepType.forward else -1), *step.map_index[2:]
                )

        # Consistency and ordering checks
        for step in self._steps:
            if self._is_training:
                if step.type_ == StepType.forward:
                    if step.stage >= self._first_grad_stage:
                        Assert.gt(step.backward_step.global_index, step.global_index)
                        Assert.is_(step.backward_step.forward_step, step)
                    else:
                        assert step.backward_step is None
                else:
                    Assert.lt(step.forward_step.global_index, step.global_index)
                    if step.stage >= self._first_grad_stage:
                        Assert.is_(step.forward_step.backward_step, step)
            if step.next_step is not None:
                Assert.gt(step.next_step.global_index, step.global_index)
                Assert.is_(step.next_step.prev_step, step)
            if step.prev_step is not None:
                Assert.lt(step.prev_step.global_index, step.global_index)
                Assert.is_(step.prev_step.next_step, step)

    def _setup_restore_steps(self, weight_buffer_indices: dict[int, int]) -> None:
        for rank, device_steps in enumerate(self._device_steps):
            if rank != self._distributed.pipeline_rank:
                # TODO: Make restore schedule for all ranks (need all buffer indices)
                continue
            buffer_contents, buffer_last_used = {}, {}
            for step in device_steps:
                buffer_index = weight_buffer_indices[step.stage]
                if buffer_contents.get(buffer_index) != step.stage:
                    if self._schedule_config.data_overlap:
                        step.restore_step = device_steps[buffer_last_used.get(buffer_index, -1) + 1]
                        step.restore_event = torch.cuda.Event()
                    else:
                        step.restore_step = step
                    step.restore_step.restore_launch.append(step)
                    buffer_contents[buffer_index] = step.stage
                buffer_last_used[buffer_index] = step.local_index

    def _setup_reduce_steps(self, grad_buffer_indices: dict[int, int]) -> None:
        if not self._is_training:
            return
        for rank, device_steps in enumerate(self._device_steps):
            if rank != self._distributed.pipeline_rank:
                # TODO: Make restore schedule for all ranks (need all buffer indices)
                continue
            buffer_last_steps = {}
            reduction_count = [0 for _ in range(self._num_stages)]
            for step in device_steps:
                if step.type_ == StepType.forward:
                    continue
                buffer_index = grad_buffer_indices[step.stage]
                if buffer_index in buffer_last_steps and buffer_last_steps[buffer_index].stage != step.stage:
                    reduce_step = buffer_last_steps[buffer_index]
                    reduce_step.reduce = True
                    reduce_step.reduce_accumulate = reduction_count[reduce_step.stage] > 0
                    reduction_count[reduce_step.stage] += 1
                buffer_last_steps[buffer_index] = step
            for reduce_step in buffer_last_steps.values():
                reduce_step.reduce = True
                reduce_step.reduce_accumulate = reduction_count[reduce_step.stage] > 0
                reduction_count[reduce_step.stage] += 1
            for stage, count in enumerate(reduction_count):
                assert (count > 0) == (
                    stage >= self._first_grad_stage
                    and (stage % self._distributed.pipeline_parallel == self._distributed.pipeline_rank)
                )

    def _setup_timeline(self) -> None:
        # TODO: Include network time
        idx = [0] * self._distributed.pipeline_parallel
        done = False
        while not done:
            done = True
            for pipeline_rank, (i, device_steps) in enumerate(zip(idx, self._device_steps)):
                if i >= len(device_steps):
                    continue
                step = device_steps[i]
                if step.prev_step is None:
                    step.start = 0
                elif step.prev_step.end is None:
                    continue
                else:
                    step.start = step.prev_step.end
                if i > 0:
                    step.start = max(step.start, device_steps[i - 1].end)
                step.compute_end = step.start + step.duration
                step.end = step.compute_end
                idx[pipeline_rank] += 1
                done = False
        # Ensure a valid timeline was found.
        Assert.eq(idx, [len(device_steps) for device_steps in self._device_steps])

    def _setup_send_recv_steps(self) -> None:
        ends = [np.array([step.end for step in device_steps]) for device_steps in self._device_steps]
        for send_rank, device_steps in enumerate(self._device_steps):
            for send_step in device_steps:
                if (recv_step := send_step.next_step) is not None and (
                    recv_rank := recv_step.pipeline_rank
                ) != send_rank:
                    # Send the output asap so the pipeline doesn't get clogged
                    # TODO: Is there a better way?
                    #  (send/recv on separate nccl communicators so they can't block each other?)

                    launch_index = np.searchsorted(ends[recv_rank], send_step.end)
                    # searchsorted returns within range(0, len(ends)+1), but len(ends) should not happen.
                    Assert.in_range(launch_index, 0, len(ends[recv_rank]))
                    launch_step = self._device_steps[recv_step.pipeline_rank][launch_index]
                    if (
                        launch_step.end == send_step.end
                        and launch_step.next_step is not None
                        and launch_step.next_step.pipeline_rank != recv_rank
                    ):
                        # Send and recv happening around the same time for the recv_step,
                        # disambiguate by favoring earlier stages.
                        # (This makes sense because of the slight pipeline delay.)
                        # This also prevents a deadlocks from cyclic recv.
                        # In practice (breadth-first only?) this gives a lower recv priority
                        # to the first device which accumulates the extra inputs.
                        # TODO: Does it always work for any schedule?
                        if launch_step.get_stage_index(self._num_stages) < send_step.get_stage_index(self._num_stages):
                            launch_index += 1
                            launch_step = self._device_steps[recv_step.pipeline_rank][launch_index]

                    launch_step.recv_launch.append(recv_step)
                    send_step.send_to = launch_step
                    recv_step.recv_step = launch_step
                    if self._schedule_config.pipeline_overlap:
                        recv_step.recv_event = torch.cuda.Event()

    def _validate_send_recv_steps(self) -> None:
        times = [0.0] * self._distributed.pipeline_parallel
        idx = [0] * self._distributed.pipeline_parallel
        recv_idx = [0] * self._distributed.pipeline_parallel
        statuses = ["Ok"] * self._distributed.pipeline_parallel
        recv_queues: list[list[Step | None]] = [[] for _ in range(self._distributed.pipeline_parallel)]
        done = False
        while not done:
            done = True
            time = min(times)
            # Prevent associated send and recv from happening in the same iteration
            # (not needed but easier to follow)
            current_queues = [q.copy() for q in recv_queues]
            for pipeline_rank, (i, device_steps) in enumerate(zip(idx, self._device_steps)):
                if times[pipeline_rank] > time or statuses[pipeline_rank] == "done":
                    continue
                if i >= len(device_steps):
                    statuses[pipeline_rank] = "done"
                    times[pipeline_rank] = float("infinity")
                    done = False
                    continue
                step = device_steps[i]
                if statuses[pipeline_rank] == "Waiting on send":
                    if step.send_to is None or step.next_step not in current_queues[step.send_to.pipeline_rank]:
                        idx[pipeline_rank] += 1
                        statuses[pipeline_rank] = "Ok"
                        recv_idx[pipeline_rank] = 0
                        done = False
                else:
                    recv_queue = current_queues[pipeline_rank]
                    while recv_idx[pipeline_rank] < len(step.recv_launch) and recv_queue:
                        if recv_queue[0] != step.recv_launch[recv_idx[pipeline_rank]]:
                            statuses[pipeline_rank] = "Received wrong input!!!!!"
                            break
                        recv_idx[pipeline_rank] += 1
                        recv_queue.pop(0)
                        done = False

                    if recv_idx[pipeline_rank] == len(step.recv_launch):
                        if step.send_to is not None:
                            current_queues[step.send_to.pipeline_rank].append(step.next_step)
                        statuses[pipeline_rank] = "Waiting on send"
                        times[pipeline_rank] = step.end
                        done = False
                    elif statuses[pipeline_rank] == "Ok":
                        statuses[pipeline_rank] = "Waiting on recv"
                        times[pipeline_rank] = step.start
                        done = False
            recv_queues = current_queues

        if idx != [len(device_steps) for device_steps in self._device_steps] or sum([len(q) for q in recv_queues]):
            msg = "".join(
                f"\n  rank {pipeline_rank}:"
                f" step={-1 if i==len(device_steps) else device_steps[i].global_index},"
                f" time={t},"
                f" status={status}"
                + (
                    f" {-1 if i==len(device_steps) or device_steps[i].next_step is None else device_steps[i].next_step.global_index}"
                    if status == "Waiting on send"
                    else f" {device_steps[i].recv_launch[r].global_index}" if status == "Waiting on recv" else ""
                )
                for pipeline_rank, (device_steps, i, r, status, t) in enumerate(
                    zip(self._device_steps, idx, recv_idx, statuses, times)
                )
            )
            raise RuntimeError(f"Cannot find valid timeline for {self}, \nStatuses:{msg}")

    def _setup_throttle_steps(self) -> None:
        if not self._schedule_config.throttle_cpu:
            return
        for device_steps in self._device_steps:
            for i, step in enumerate(device_steps):
                if i >= self._schedule_config.throttle_cpu_delay and i % self._schedule_config.throttle_cpu_rate == 0:
                    throttle_step = device_steps[i - self._schedule_config.throttle_cpu_delay]
                    throttle_step.throttle_event = torch.cuda.Event()
                    step.throttle_step = throttle_step

    def _setup_metas(self) -> None:
        for step in self._steps:
            if step.type_ == StepType.forward:
                if step.prev_step is None:
                    assert step.stage == 0
                    step.meta_input, step.meta_kwargs = self._preprocessed_meta[step.micro_sequence]
                # meta_kwargs may be modified.
                meta_kwargs = step.meta_kwargs.copy()
                step.meta_output = self._multi_stage.stages[step.stage].forward_meta(step.meta_input, meta_kwargs)
                if step.next_step is not None:
                    step.next_step.meta_input = step.meta_output
                    step.next_step.meta_kwargs = step.meta_kwargs

    def get_data_index(self, micro_batch: int, micro_sequence: int) -> int:
        return micro_batch * self._batch_config.num_micro_sequences + micro_sequence

    def get_data_index_split(
        self, breadth_first_micro_batch: int, depth_first_micro_batch: int, micro_sequence: int
    ) -> int:
        return self.get_data_index(
            breadth_first_micro_batch * self._batch_config.depth_first_micro_batches + depth_first_micro_batch,
            micro_sequence,
        )

    def _create_steps(self) -> tuple[list[Step], int]:
        steps = []
        if self._is_training:
            # The first stage(s) may not have any trainable parameters,
            # in which case we shouldn't run the backward pass.
            first_grad_stage = 0
            while first_grad_stage < self._num_stages and not self._multi_stage.stages[first_grad_stage].requires_grad:
                first_grad_stage += 1
        else:
            first_grad_stage = self._num_stages
        for depth_first_micro_batch in range(self._batch_config.depth_first_micro_batches):
            for stage in range(self._num_stages):
                for breadth_first_micro_batch in range(self._batch_config.breadth_first_micro_batches):
                    for micro_sequence in range(self._batch_config.num_micro_sequences):
                        steps.append(
                            Step(
                                config=self._batch_config,
                                stage=stage,
                                data_index=self.get_data_index_split(
                                    breadth_first_micro_batch, depth_first_micro_batch, micro_sequence
                                ),
                                type_=StepType.forward,
                            )
                        )
            if self._is_training:
                for stage in reversed(range(first_grad_stage, self._num_stages)):
                    for breadth_first_micro_batch in range(self._batch_config.breadth_first_micro_batches):
                        for micro_sequence in reversed(range(self._batch_config.num_micro_sequences)):
                            steps.append(
                                Step(
                                    config=self._batch_config,
                                    stage=stage,
                                    data_index=self.get_data_index_split(
                                        breadth_first_micro_batch, depth_first_micro_batch, micro_sequence
                                    ),
                                    type_=StepType.backward,
                                )
                            )
        return steps, first_grad_stage
