import logging
import typing

import torch
from torch.distributed import all_reduce, reduce_scatter_tensor

from fast_llm.core.distributed import ReduceOp, check_parallel_match
from fast_llm.engine.config_utils.run import log_pipeline_parallel_main_rank
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import StageMode
from fast_llm.engine.multi_stage.stage_base import StageBase
from fast_llm.functional.triton.pointwise import triton_add, triton_copy, triton_fill
from fast_llm.logging import log_distributed_grad, log_distributed_tensor, log_memory_usage, log_tensor
from fast_llm.tensor import ParameterMeta, TensorMeta, accumulate_gradient
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


def _accumulate_grad_hook(buffer: torch.nn.Parameter, meta: ParameterMeta) -> typing.Callable[[tuple, tuple], None]:
    def hook(grad_inputs, grad_outputs):  # noqa
        if buffer.grad is not None:
            if not meta.auto_grad_accumulation:
                raise RuntimeError(f"Unexpected grad for parameter {meta.tensor_name}")
            accumulate_gradient(buffer, buffer.grad)
            buffer.grad = None

    return hook


class Stage(StageBase):
    _is_restored: bool
    _training: bool | None = None
    # TODO: Handle all buffer sharing in multi_stage
    _weight_buffer_shared_with: list["Stage"]
    _is_tied_weight_copy: bool
    _accumulators: list

    def setup(  # noqa
        self,
        *,
        distributed: Distributed,
        weight_shard: torch.Tensor | None,
        grad_shard: torch.Tensor | None,
        weight_buffer: torch.Tensor | None,
        grad_buffer: torch.Tensor | None,
        mode: StageMode = StageMode.training,
        is_tied_weight_copy: bool = False,
        weight_buffer_shared_with: list["Stage"],
    ) -> None:
        super().setup(
            distributed=distributed,
            weight_shard=weight_shard,
            grad_shard=grad_shard,
            weight_buffer=weight_buffer,
            grad_buffer=grad_buffer,
            mode=mode,
        )
        self._is_tied_weight_copy = is_tied_weight_copy
        if self._mode.support_forward:
            self._weight_buffer_shared_with = [x for x in weight_buffer_shared_with if x is not self]
            self.invalidate_buffer()
        else:
            Assert.empty(weight_buffer_shared_with)

        if self._mode.support_backward:
            self._accumulators = []
            with torch.enable_grad():
                for buffer, meta in zip(self._parameter_buffers, self._parameter_metas):
                    # We want to replace the grad accumulation function with ours, but pytorch won't let us do that.
                    # Instead, we let a trivial accumulation run its course (sets .grad),
                    # then run the actual accumulation.
                    # We get the AccumulateGrad object through the grad function of a dummy tensor.
                    accumulator = buffer.expand_as(buffer).grad_fn.next_functions[0][0]
                    accumulator.register_hook(_accumulate_grad_hook(buffer, meta))
                    # We keep a pointer to the AccumulateGrad object, otherwise it will be garbage collected.
                    self._accumulators.append(accumulator)

    def forward_meta(self, input_: TensorMeta, kwargs: dict) -> TensorMeta:
        # Store the meta inputs and outputs, for debugging only.
        self._meta_inputs, self._meta_outputs = [], []
        # TODO: use layer.forward_meta
        for layer in self._layers:
            self._meta_inputs.append(input_)
            input_ = layer(
                input_,
                kwargs,
                losses={},
            )
            self._meta_outputs.append(input_)
        return input_

    def forward(
        self, input_: torch.Tensor, kwargs: dict, losses: dict[str, list[torch.Tensor]], metrics: dict | None = None
    ) -> tuple[torch.Tensor | None, tuple[torch.Tensor | None, torch.Tensor | None]]:
        assert self._is_restored
        assert self._mode.support_forward
        output = input_
        for i, layer in enumerate(self._layers):
            self._log_layer_backward(output, kwargs, i)
            output = layer(
                output,
                kwargs,
                losses,
                metrics,
            )
            self._log_layer_forward(output, kwargs, i)
        return None if output is None else output.detach(), (input_, output)

    def backward(
        self, output_grad: torch.Tensor, grad_context: tuple[torch.Tensor, torch.Tensor], metrics: dict | None = None
    ) -> torch.Tensor:  # noqa
        # TODO: This context format wastes memory.
        # TODO: Allow non-autograd layers.
        assert self._mode.support_backward
        input_, output = grad_context
        output.backward(output_grad)
        return input_.grad

    def restore_parameters(self) -> None:
        assert self._is_setup
        assert self._mode.support_forward
        # TODO: Allow partial FSDP
        if not self._is_restored:
            triton_copy(self._weight_shard, self._weight_buffer_local_shard)
            if self._fsdp_size > 1:
                self._reconstruct_from_shard(self._weight_buffer_local_shard, self._weight_buffer)
            self._is_restored = True
            for stage in self._weight_buffer_shared_with:
                stage.invalidate_buffer()

    def reset_gradients(self) -> None:
        # TODO: Allow re-allocating the gradient every time.
        # TODO: Autograd will always increment gradient instead of setting the value (less efficient)
        #   Can this (and op below) be avoided? (Probably needs messing with autograd)
        #   Solution: set a zero_grad flag on parameter, then adjust backward fn to set or accumulate depending on flag.
        #     Then we can also avoid explicitly setting to zero.
        #     Logic implemented for linear and ln, missing embedding.
        assert self._is_setup
        assert self._mode.support_backward
        # assert self._is_restored
        for buffer in self._parameter_buffers:
            assert buffer.grad is None
            buffer.param_grad_is_zero = True

    def reduce_gradients(self, accumulate=False) -> None:
        # Just need to reduce the buffer, then copy (add) to actual grads.
        # Works fine as is but does not allow communication overlap by itself.
        # Reduction should only be done once per step, after the full backward pass is done for the stage.
        # TODO: Allow partial FSDP
        assert self._is_restored
        assert self._mode.support_backward
        for buffer, meta in zip(self._parameter_buffers, self._parameter_metas):
            if buffer.param_grad_is_zero:  # noqa
                assert self.is_tied_weight_copy or meta.allow_no_grad, meta
                triton_fill(buffer.grad_buffer, 0)  # noqa
        if self._sequence_parallel_grads is not None and self._distributed.tensor_group:
            all_reduce(self._sequence_parallel_grads, group=self._distributed.tensor_group)
        if self._fsdp_size > 1:
            out = self._grad_shard if self._config.full_precision_gradients else self._grad_buffer_local_shard
            if accumulate:
                out = torch.empty_like(out)
            reduce_scatter_tensor(
                out,
                self._grad_buffer,
                group=self._fsdp_group,
                op=ReduceOp.AVG,
            )
            if accumulate:
                triton_add(self._grad_shard, out, self._grad_shard)
            elif not self._config.full_precision_gradients:
                triton_copy(self._grad_buffer_local_shard, self._grad_shard)
        else:
            triton_copy(self._grad_buffer_local_shard, self._grad_shard)
        if self._config.debug_param_gradients:
            log_tensor(
                "Reduced gradient shard",
                self._grad_shard,
                level=self._config.debug_param_gradients,
                global_=False,
            )
        if self._config.debug_all_param_gradients:
            self.log_shard(
                name="gradient",
                shard=self._grad_shard,
                level=self._config.debug_all_param_gradients,
            )

    @property
    def is_tied_weight_copy(self) -> bool:
        return self._is_tied_weight_copy

    def train(self, mode: bool = True) -> None:
        if mode:
            assert self._mode.support_backward
        if self._training != mode:
            for layer in self._layers:
                layer.train(mode)
            self._training = mode

    def invalidate_buffer(self) -> None:
        # Buffer is no longer valid (Updated weights or overwritten by other stage)
        assert self._mode.support_forward
        self._is_restored = False

    def _log_layer_forward(self, output: torch.Tensor, kwargs: dict[str, typing.Any], i: int) -> None:
        if (
            self._config.debug_tensor_parallel
            and self._distributed.tensor_group is not None
            and not self._meta_outputs[i].is_tensor_parallel
        ):
            check_parallel_match(output, self._distributed.tensor_group, f"layer {self._layer_range[i]} fw")
        if self._config.debug_layer_outputs:
            name = f"layer {self._layer_range[i]} fw"
            if (nmb := kwargs.get("num_micro_batches", 1)) > 1:
                name = f"{name}, mb={kwargs.get('micro_batch',0)}/{nmb}"
            if (nms := kwargs.get("num_micro_sequences", 1)) > 1:
                name = f"{name}, ms={kwargs.get('micro_sequence',0)}/{nms}"

            log_distributed_tensor(
                name,
                output,
                level=self._config.debug_layer_outputs,
                distributed=self._distributed,
                global_=self._config.debug_global_tensors,
                meta=self._meta_outputs[i],
            )
        if self._config.debug_activation_memory:
            log_pipeline_parallel_main_rank(lambda: log_memory_usage(f"layer {self._layer_range[i]} fw", str))

    def _log_layer_backward(self, input_: torch.Tensor, kwargs: dict[str, typing.Any], i: int) -> None:
        if not input_.requires_grad:
            return
        if (
            self._config.debug_tensor_parallel
            and self._distributed.tensor_group is not None
            and not self._meta_inputs[i].is_tensor_parallel
        ):
            input_.register_hook(
                lambda grad: check_parallel_match(
                    grad, self._distributed.tensor_group, f"layer {self._layer_range[i]} bw"
                )
            )
        if self._config.debug_layer_gradients:
            name = f"layer {self._layer_range[i]} bw"
            if (nmb := kwargs.get("num_micro_batches", 1)) > 1:
                name = f"{name}, mb={kwargs.get('micro_batch',0)}/{nmb}"
            if (nms := kwargs.get("num_micro_sequences", 1)) > 1:
                name = f"{name}, ms={kwargs.get('micro_sequence',0)}/{nms}"
            log_distributed_grad(
                name,
                input_,
                level=self._config.debug_layer_gradients,
                distributed=self._distributed,
                grad_fn=lambda grad: grad / self._fsdp_size,
                global_=self._config.debug_global_tensors,
                meta=self._meta_inputs[i],
            )
        if self._config.debug_activation_memory:
            input_.register_hook(
                lambda grad: log_pipeline_parallel_main_rank(
                    lambda: log_memory_usage(f"layer {self._layer_range[i]} bw", str)
                )
            )
