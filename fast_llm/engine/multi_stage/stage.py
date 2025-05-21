import collections
import logging
import typing

import torch

from fast_llm.core.distributed import check_parallel_match
from fast_llm.engine.config_utils.run import log_pipeline_parallel_main_rank
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import StageMode
from fast_llm.engine.multi_stage.stage_base import StageBase
from fast_llm.logging import log_distributed_grad, log_distributed_tensor, log_memory_usage, log_tensor
from fast_llm.tensor import ParameterMeta, TensorMeta, accumulate_gradient
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    pass

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
        weight_shards: list[torch.Tensor | None] | None = None,
        grad_shards: list[torch.Tensor | None] | None = None,
        weight_buffers: list[torch.Tensor | None] | None = None,
        grad_buffers: list[torch.Tensor | None] | None = None,
        mode: StageMode = StageMode.training,
        is_tied_weight_copy: bool = False,
        weight_buffer_shared_with: collections.abc.Sequence["Stage"] = (),
    ) -> None:
        super().setup(
            distributed=distributed,
            weight_shards=weight_shards,
            grad_shards=grad_shards,
            weight_buffers=weight_buffers,
            grad_buffers=grad_buffers,
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
                for meta in self._parameter_metas:
                    buffer = self.get_parameter_buffer(meta.tensor_name)
                    if not buffer.requires_grad:
                        continue
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
        self,
        input_: torch.Tensor,
        kwargs: dict,
        losses: dict[str, list[torch.Tensor]] | None = None,
        metrics: dict | None = None,
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

            # TODO: very slow and memory consuming, only use for debugging for now
            # TODO: decide if and how we want to return
            #       HF transformer style details from forward properly
            if "output_hidden_states" in kwargs and kwargs["output_hidden_states"]:
                # Last layer does not provide output
                if output is not None:
                    meta = self._meta_outputs[i]
                    output_global, _ = meta.local_to_global(output.detach(), distributed=self._distributed)
                    kwargs["hidden_states"][self._layer_range[i]] = {
                        "layer_type": type(layer).__name__,
                        "tensor": output_global,
                    }
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
            for fsdp in self._fsdps:
                fsdp.restore_parameters()
            self._is_restored = True
            for stage in self._weight_buffer_shared_with:
                stage.invalidate_buffer()

    def reset_gradients(self) -> None:
        # TODO: Allow re-allocating the gradient every time.
        assert self._is_setup
        assert self._mode.support_backward
        for fsdp in self._fsdps:
            fsdp.reset_gradients()

    def reduce_gradients(self, accumulate=False) -> None:
        # Reduce the buffer, then copy (add) to actual grads.
        # Run in a separate cuda stream to allow communication overlap.
        # TODO: Allow partial FSDP
        assert self._is_restored
        assert self._mode.support_backward
        for fsdp in self._fsdps:
            fsdp.reduce_gradients(self._distributed, accumulate, self._is_tied_weight_copy)
            if self._config.debug_param_gradients:
                log_tensor(
                    "Reduced gradient shard",
                    fsdp.grad_shard,
                    level=self._config.debug_param_gradients,
                    global_=False,
                )
            if self._config.debug_all_param_gradients and fsdp.requires_grad:
                fsdp.log_shard(
                    name="gradient",
                    shard=fsdp.grad_shard,
                    distributed=self._distributed,
                    level=self._config.debug_all_param_gradients,
                    global_=self._config.debug_global_tensors,
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
        # TODO: Frozen weights fsdps may not be invalidated on weight update.
        for fsdp in self._fsdps:
            fsdp.invalidate_buffer()

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
            if (nms := kwargs.get("micro_batch_splits", 1)) > 1:
                name = f"{name}, ms={kwargs.get('micro_batch_split',0)}/{nms}"

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
            if (nms := kwargs.get("micro_batch_splits", 1)) > 1:
                name = f"{name}, ms={kwargs.get('micro_batch_split',0)}/{nms}"
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
