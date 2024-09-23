import collections
import dataclasses
import logging
import time
import typing

import torch
import torch.cuda
import yaml

from fast_llm.core.distributed import all_reduce, recv, safe_barrier, send
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.multi_stage import MultiStageModel
from fast_llm.engine.multi_stage.stage import Stage
from fast_llm.engine.optimizer.optimizer import Optimizer
from fast_llm.engine.run.run import log_pipeline_parallel_main_rank, open_artifact
from fast_llm.engine.schedule.config import EventType, ScheduleConfig, StepType, StreamType
from fast_llm.engine.schedule.schedule import Schedule, Step
from fast_llm.logging import log_memory_usage
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


@dataclasses.dataclass()
class BatchContext:
    iteration: int
    schedule: Schedule
    # Index and data: (iteration, data_index, input, kwargs)
    data_iterator: typing.Iterator[tuple[int, torch.Tensor, dict]] = None
    inputs: dict[int, torch.Tensor] = dataclasses.field(default_factory=dict)
    batch: dict[int, dict] = dataclasses.field(default_factory=dict)
    contexts: dict[int, tuple[torch.Tensor, torch.Tensor]] = dataclasses.field(default_factory=dict)
    # Dictionary of losses, purely for logging purposes.
    # Losses will be reduced over DP and PP, and aggregated over steps.
    losses: dict | None = None
    profile: list[tuple[EventType, Step | None, torch.cuda.Event, StreamType, float]] = dataclasses.field(
        default_factory=list
    )
    # Store metrics like: grad norm, loss scale, learning-rate, etc.
    metrics: dict | None = None

    @property
    def phase(self):
        return self.schedule.phase

    @property
    def is_training(self):
        return self.phase.is_training

    @property
    def done(self):
        return not (self.inputs or self.contexts)

    def __repr__(self):
        return (
            f"BatchContext(batch_len={len(self.batch)},"
            f" inputs={list(self.inputs)},"
            f" contexts={list(self.contexts)},"
            f" losses={ {key: len(value) for key, value in self.losses.items()}},"
        )


class ScheduleRunner:
    _is_setup: bool = False
    _compute_stream: torch.cuda.Stream
    _data_stream: torch.cuda.Stream
    _pipeline_stream: torch.cuda.Stream
    _streams: dict[int, StreamType]
    _compute_event: torch.cuda.Event
    _reduce_event: torch.cuda.Event
    _send_event: torch.cuda.Event
    _data_stream_needs_sync: bool
    _profile_events: dict[tuple[EventType, tuple | None], torch.cuda.Event]
    _distributed: Distributed
    _optimizer: Optimizer | None
    _stages_on_device: list[Stage]
    _stages_owned: list[bool]
    _support_training: bool

    def __init__(
        self,
        *,
        multi_stage: MultiStageModel,
        config: ScheduleConfig,
        distributed_config: DistributedConfig,
    ):
        super().__init__()
        self._config = config
        self._distributed_config = distributed_config
        self._multi_stage = multi_stage
        self._stages: list[Stage] = self._multi_stage.stages
        self._tied_parameters = self._multi_stage.tied_parameters
        self._num_stages = len(self._stages)
        self._loss_defs = {loss_def.name: loss_def for loss_def in self._multi_stage.base_model.loss_defs}

    def setup(self, distributed: Distributed, optimizer: Optimizer | None = None):
        assert not self._is_setup
        assert distributed.config is self._distributed_config
        self._is_setup = True
        self._optimizer = optimizer
        assert self._multi_stage.support_forward
        self._support_training = self._multi_stage.support_training and self._optimizer is not None

        self._distributed = distributed
        self._stages_on_device = [stage for stage in self._stages if stage.mode.on_device]
        self._stages_owned = [stage.mode.on_device and not stage.is_tied_weight_copy for stage in self._stages]

        # Setup the streams
        self._compute_stream = torch.cuda.current_stream(self._distributed.device)
        self._data_stream = (
            torch.cuda.Stream(self._distributed.device) if self._config.data_overlap else self._compute_stream
        )
        self._pipeline_stream = (
            torch.cuda.Stream(self._distributed.device) if self._config.pipeline_overlap else self._compute_stream
        )
        # Putting compute stream last in the dict in case it's the same id.
        self._streams = {
            self._data_stream.stream_id: StreamType.data,
            self._pipeline_stream.stream_id: StreamType.pipeline,
            self._compute_stream.stream_id: StreamType.compute,
        }

        # Setup the synchronization and profiling events
        self._profile_events = collections.defaultdict(lambda: torch.cuda.Event(enable_timing=True))
        self._compute_event = torch.cuda.Event()
        self._reduce_event = torch.cuda.Event()
        self._send_event = torch.cuda.Event()
        self._data_stream_needs_sync = False

    def run_step(
        self,
        data_iterator,
        schedule: Schedule,
        *,
        iteration: int = 1,
        return_metrics: bool = False,
        preprocessed: bool = False,
    ):
        assert self._is_setup
        assert schedule._schedule_config is self._config  # Noqa
        if schedule.phase.is_training:
            assert self._support_training

        metrics = {} if return_metrics else None
        # Set the context.
        context = BatchContext(
            iteration=iteration,
            schedule=schedule,
            losses={loss_def: [] for loss_def in self._loss_defs},
            metrics=metrics,
        )
        context.data_iterator = self._preprocess_data(context, data_iterator, preprocessed)

        if self._multi_stage.multi_stage_config.debug_activation_memory:
            log_pipeline_parallel_main_rank(
                lambda: log_memory_usage(f"Beginning of {context.phase.value} iteration {iteration}", str)
            )
        self._multi_stage.train(context.is_training)
        self._distributed.set_step(iteration, schedule.phase)

        # Synchronize streams
        Assert.eq(torch.cuda.current_stream(self._distributed.device), self._compute_stream)
        if self._config.profile_schedule:
            # Synchronize clocks
            safe_barrier(self._distributed.world_group, f"clock sync {iteration}")
        self._record_event(context, EventType.batch_begin, None)
        self._data_stream.wait_stream(self._compute_stream)
        self._record_event(context, EventType.data_wait_compute, None, self._data_stream)
        self._pipeline_stream.wait_stream(self._compute_stream)
        self._record_event(context, EventType.pipe_wait_compute, None, self._pipeline_stream)

        # Reset gradients
        # TODO: This is incorrect with shared buffers.
        #   (still works because only the embedding layer doesn't share buffer)
        for stage in self._stages_on_device:
            if context.is_training:
                stage.reset_gradients()
            # TODO: Overlap this?
            if stage.is_tied_weight_copy:
                stage.restore_parameters()

        self._record_event(context, EventType.pre_restore, None)

        # Prepare the batch
        self._record_event(context, EventType.get_batch, None)

        if self._multi_stage.multi_stage_config.debug_activation_memory:
            log_pipeline_parallel_main_rank(lambda: log_memory_usage(f"Beginning of the schedule steps", str))

        # Run the steps according to the schedule
        for step in schedule:
            self._train_step(context, step)

        # Make sure we used all the data. This also ensures the generator terminates and prevents a memory leak.
        try:
            next(context.data_iterator)
        except StopIteration:
            pass
        else:
            raise AssertionError("Data iterator did not terminate")

        assert context.done, context

        if self._multi_stage.multi_stage_config.debug_activation_memory:
            log_pipeline_parallel_main_rank(lambda: log_memory_usage(f"End of the schedule steps", str))

        # Synchronize streams
        self._send_event.wait()
        self._record_event(context, EventType.compute_wait_pipe, None)
        self._reduce_event.wait()
        self._record_event(context, EventType.compute_wait_data, None)

        if not context.is_training or self._config.skip_step:
            return self._reduce_losses(context), True, metrics

        for name, tied_parameter in self._tied_parameters.items():
            if tied_parameter.group is not None:
                main_stage = self._stages[tied_parameter.main_stage]
                if main_stage.is_tied_weight_copy:
                    if not self._config.skip_step:
                        # Stage hasn't been reduced yet.
                        # TODO: Overlap this? (reduce with last local layer that uses it)
                        main_stage.reduce_gradients()
                # TODO: Overlap this? (not really useful for gpt)
                all_reduce(main_stage.grad_shard, group=tied_parameter.group)
                if self._multi_stage.multi_stage_config.debug_all_param_gradients:
                    main_stage.log_shard(
                        name="gradient",
                        shard=main_stage.grad_shard,
                        level=self._multi_stage.multi_stage_config.debug_all_param_gradients,
                    )

        self._record_event(context, EventType.post_reduce, None)
        # Update weights
        # TODO: Option to update with reduce (needs per-layer grad_norm and update_successful)
        # TODO: Avoid blocking synchronizations: async transfer, turn noop_flag into a real noop flag
        #  (uncomment line in apex).
        update_successful = self._optimizer.step(metrics)

        if self._multi_stage.multi_stage_config.debug_tensor_parallel and self._distributed.tensor_group is not None:
            for stage in self._stages_on_device:
                stage.check_tensor_parallel_synchronization()

        if update_successful:
            for stage in self._stages_on_device:
                stage.invalidate_buffer()
        if self._multi_stage.multi_stage_config.debug_param_update:
            for stage in self._stages_on_device:
                stage.log_shard(
                    name="param",
                    shard=stage.weight_shard,
                    level=self._multi_stage.multi_stage_config.debug_param_update,
                )

        self._record_event(context, EventType.optimizer, None)
        self._record_event(context, EventType.batch_end, None)
        self._handle_events(context)

        if metrics is not None:
            metrics["loss_scale"] = self._optimizer.grad_scale

        if self._multi_stage.multi_stage_config.debug_activation_memory:
            log_pipeline_parallel_main_rank(
                lambda: log_memory_usage(f"End of {context.phase.value} iteration {iteration}", str)
            )

        return self._reduce_losses(context), update_successful, metrics

    def _reduce_losses(self, context: BatchContext) -> dict[str, float | int]:
        reduced_losses = {}
        num_inputs = self._distributed_config.data_parallel * context.schedule.batch_config.num_inputs
        for name, losses in context.losses.items():
            if losses or self._distributed.pipeline_group:
                if losses:
                    reduced_loss = torch.stack(losses).sum() / num_inputs / self._loss_defs[name].count
                    if self._distributed.data_group:
                        all_reduce(reduced_loss, group=self._distributed.data_group)
                else:
                    reduced_loss = torch.zeros([1], dtype=self._loss_defs[name].dtype, device=self._distributed.device)
                if self._distributed.pipeline_group:
                    all_reduce(reduced_loss, group=self._distributed.pipeline_group)
            else:
                reduced_loss = 0.0
            reduced_losses[name] = reduced_loss
        return {
            name: reduced_loss.item() if isinstance(reduced_loss, torch.Tensor) else reduced_loss
            for name, reduced_loss in reduced_losses.items()
        }

    def _train_step(self, context: BatchContext, step: Step):
        if step.throttle_event is not None:
            step.throttle_event.record()
        if step.throttle_step is not None:
            step.throttle_step.throttle_event.synchronize()
        self._restore(context, step)
        self._recv(context, step)
        if step.type_ == StepType.forward:
            output = self._forward(context, step)
        elif step.type_ == StepType.backward:
            output = self._backward(context, step)
        else:
            raise NotImplementedError(step.type_)
        self._send(context, step, output)
        self._reduce(context, step)

    def _preprocess_data(self, context: BatchContext, data_iterator: typing.Iterator, preprocessed: bool):
        batch_config = context.schedule.batch_config
        grad_output = (
            (1 if self._optimizer is None else self._optimizer.grad_scale)
            / batch_config.sequential_micro_batches
            / batch_config.num_micro_sequences
        )
        for micro_batch in range(batch_config.sequential_micro_batches):
            micro_batch_data = next(data_iterator)
            if not preprocessed:
                micro_batch_data = self._multi_stage.base_model.preprocess(
                    micro_batch_data,
                    context.schedule.preprocessed_meta,
                    phase=context.phase,
                    iteration=context.iteration,
                    metrics=context.metrics,
                )
            for micro_sequence, (input_, kwargs) in enumerate(micro_batch_data):
                kwargs.update(
                    grad_output=grad_output,
                    micro_batch=micro_batch,
                    micro_sequence=micro_sequence,
                    num_micro_batches=batch_config.sequential_micro_batches,
                    num_micro_sequences=batch_config.num_micro_sequences,
                )
                for name, tied_parameter in self._tied_parameters.items():
                    if tied_parameter.on_device:
                        kwargs[name] = self._stages[tied_parameter.main_stage].get_parameter_buffer(
                            tied_parameter.meta
                        )
                data_index = context.schedule.get_data_index(micro_batch, micro_sequence)
                if self._stages_owned[0]:
                    context.inputs[context.schedule.get_step(StepType.forward, 0, data_index).global_index] = input_
                if context.is_training and self._stages_owned[-1]:
                    step = context.schedule.get_step(StepType.backward, self._num_stages - 1, data_index)
                    # TODO: Avoidable?
                    context.inputs[step.global_index] = torch.empty_like(
                        step.forward_step.meta_output,
                        device=self._distributed.device if self._stages_owned[-1] else "meta",
                    )
                context.batch[data_index] = kwargs
                yield

    def _restore(self, context: BatchContext, step: Step):
        if step.restore_launch:
            with torch.cuda.stream(self._data_stream):
                self._sync_data_stream(context, step)
                for restore_step in step.restore_launch:
                    self._stages[restore_step.stage].restore_parameters()
                    if restore_step.restore_event is not None:
                        restore_step.restore_event.record()
                    self._record_event(context, EventType.restore, restore_step)

        if step.restore_event is not None:
            step.restore_event.wait()
            self._record_event(context, EventType.compute_wait_data, step)

    def _recv(self, context: BatchContext, step: Step):
        if step.recv_launch:
            with torch.cuda.stream(self._pipeline_stream):
                for recv_step in step.recv_launch:
                    # TODO: Pre-allocated buffers
                    context.inputs[recv_step.global_index] = torch.empty_like(
                        recv_step.meta_input if step.type_ == StepType.forward else recv_step.forward_step.meta_output,
                        device=self._distributed.device,
                    )
                    if self._config.debug_send_recv:
                        data = torch.empty([2], dtype=torch.int64, device=self._distributed.device)
                        recv(data, src=recv_step.prev_step.pipeline_rank, group=self._distributed.pipeline_group)
                        idx, size = data.tolist()
                        Assert.eq(idx, recv_step.global_index)
                        Assert.eq(size, context.inputs[recv_step.global_index].numel())

                    recv(
                        context.inputs[recv_step.global_index],
                        src=recv_step.prev_step.pipeline_rank,
                        group=self._distributed.pipeline_group,
                    )
                    if recv_step.recv_event is not None:
                        recv_step.recv_event.record()
                    self._record_event(context, EventType.recv, recv_step)

        if step.recv_event is not None:
            step.recv_event.wait()
            self._record_event(context, EventType.compute_wait_pipe, step)

    def _forward(self, context: BatchContext, step: Step):
        output, grad_context = self._stages[step.stage].forward(
            self._get_forward_input(context, step),
            context.batch[step.data_index],
            losses=context.losses,
            metrics=context.metrics,
        )
        if context.is_training:
            context.contexts[step.backward_step.global_index] = grad_context
        self._record_compute(context, step)
        return output

    def _backward(self, context: BatchContext, step: Step):
        input_grad = self._stages[step.stage].backward(
            context.inputs.pop(step.global_index),
            context.contexts.pop(step.global_index),
        )
        self._record_compute(context, step)
        return input_grad

    def _get_forward_input(self, context: BatchContext, step: Step):
        if step.data_index not in context.batch:
            start_time = time.perf_counter()

            while step.data_index not in context.batch:
                next(context.data_iterator)

            data_time = (time.perf_counter() - start_time) * 1000
            if data_time > self._config.data_batch_warn_time_ms:
                logger.warning(f"Data loading took {data_time:,.2f} ms")
        return context.inputs.pop(step.global_index).detach().requires_grad_(step.stage != 0)

    def _send(self, context: BatchContext, step: Step, output: torch.Tensor):
        if step.next_step is not None:
            if step.next_step.recv_step is None:
                context.inputs[step.next_step.global_index] = output
            else:
                with torch.cuda.stream(self._pipeline_stream):
                    self._compute_event.wait()
                    self._record_event(context, EventType.pipe_wait_compute, step, self._pipeline_stream)
                    if self._config.debug_send_recv:
                        data = torch.tensor(
                            [step.next_step.global_index, output.numel()],
                            dtype=torch.int64,
                            device=self._distributed.device,
                        )
                        send(data, dst=step.next_step.pipeline_rank, group=self._distributed.pipeline_group)
                    # The pipeline will hang if there is a shape error.
                    meta = step.forward_step.meta_input if step.type_ == StepType.backward else step.meta_output
                    Assert.eq(output.shape, meta.shape)
                    Assert.eq(output.dtype, meta.dtype)
                    send(output, dst=step.next_step.pipeline_rank, group=self._distributed.pipeline_group)
                    self._send_event.record()
                    self._record_event(context, EventType.send, step)

    def _reduce(self, context: BatchContext, step: Step):
        if step.reduce:
            with torch.cuda.stream(self._data_stream):
                self._sync_data_stream(context, step)
                stage = self._stages[step.stage]
                if not self._config.skip_step:
                    stage.reduce_gradients(accumulate=step.reduce_accumulate)
                stage.reset_gradients()
                self._reduce_event.record()
                self._record_event(context, EventType.reduce, step)

    def _record_event(
        self, context: BatchContext, type_: EventType, step: Step | None, stream: torch.cuda.Stream = None
    ):
        if not self._config.profile_schedule:
            return
        if stream is None:
            stream = torch.cuda.current_stream()
        event = self._profile_events[(type_, None if step is None else step.map_index)]
        event.record(stream)
        cpu_time = time.perf_counter()
        context.profile.append((type_, step, event, self._streams[stream.stream_id], cpu_time))

    def _handle_events(self, context: BatchContext):
        if not context.profile:
            return
        events = []
        _, _, gpu_begin, stream, cpu_begin = context.profile[0]
        for type_, step, event, stream, cpu_time in context.profile:
            event.synchronize()
            # Cuda events are measured in milliseconds
            events.append((type_, step, stream, gpu_begin.elapsed_time(event) / 1000, cpu_time - cpu_begin))
        self._save_events(events, context)

    def _save_events(self, events, context: BatchContext):
        out = {
            "iteration": context.iteration,
            "phase": context.phase.value,
            "rank": self._distributed_config.rank,
            "events": [
                {
                    "event_type": type_.value,
                    "stream": stream.value,
                    "gpu_time": gpu_time,
                    "cpu_time": cpu_time,
                    **(
                        {}
                        if step is None
                        else {
                            "step_idx": step.global_index,
                            "step_type": step.type_.value,
                            "step_stage": step.stage,
                            "step_depth_first_micro_batch": step.depth_first_micro_batch,
                            "step_breadth_first_micro_batch": step.breadth_first_micro_batch,
                            "step_micro_sequence": step.micro_sequence,
                        }
                    ),
                }
                for type_, step, stream, gpu_time, cpu_time in events
            ],
        }
        yaml.safe_dump(
            out,
            open_artifact(
                f"schedule_profile_rank_{self._distributed_config.rank}_{context.phase.value}_step_{context.iteration}"
            ),
        )

    def _sync_data_stream(self, context: BatchContext, step: Step):
        if self._data_stream_needs_sync:
            self._compute_event.wait()
            self._data_stream_needs_sync = False
            self._record_event(context, EventType.data_wait_compute, step)

    def _record_compute(self, context: BatchContext, step: Step):
        self._compute_event.record()
        self._record_event(context, EventType.run, step)
        if self._config.data_overlap:
            self._data_stream_needs_sync = True
