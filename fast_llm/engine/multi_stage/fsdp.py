import typing

import torch
from torch._C._distributed_c10d import ReduceOp
from torch.distributed import all_reduce, reduce_scatter_tensor

from fast_llm.core.distributed import ProcessGroup
from fast_llm.core.ops import gather_op
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.tensor_space import TensorDim
from fast_llm.engine.distributed.config import DistributedDim
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import SHARD_PAD_TO_MULTIPLE, ShardName, StageMode
from fast_llm.functional.triton.pointwise import triton_add, triton_copy
from fast_llm.logging import log_distributed_tensor
from fast_llm.tensor import ParameterMeta, SafeTensorSlice, TensorMeta
from fast_llm.utils import Assert, clamp, padded_cumsum


class FSDP:
    _is_setup: bool = False
    _is_restored: bool = False
    _fsdp_group: ProcessGroup

    _weight_shard: torch.Tensor
    _grad_shard: torch.Tensor
    _weight_buffer: torch.Tensor
    _grad_buffer: torch.Tensor
    _sequence_parallel_grads: torch.Tensor
    _weight_buffer_local_shard: torch.Tensor
    _grad_buffer_local_shard: torch.Tensor
    _parameter_buffers: dict[str, torch.nn.Parameter]

    def __init__(
        self,
        name: str,
        parameter_metas: list[ParameterMeta],
        fsdp_dim: DistributedDim,
        training_dtype: DataType,
        gradient_buffer_dtype: DataType,
        optimization_dtype: DataType,
    ):
        self._name = name
        self._parameter_metas = {parameter_meta.tensor_name: parameter_meta for parameter_meta in parameter_metas}
        self._fsdp_dim = fsdp_dim
        self._training_dtype = training_dtype
        self._gradient_buffer_dtype = gradient_buffer_dtype
        self._optimization_dtype = optimization_dtype

        self._requires_grad = any(parameter_meta.requires_grad for parameter_meta in self._parameter_metas.values())

        parameter_sizes = [meta.numel() for meta in self._parameter_metas.values()]
        self._parameter_count = sum(parameter_sizes)
        parameter_offsets = padded_cumsum(parameter_sizes).tolist()

        # The index range of the parameters in the buffer.
        self._parameter_begins_in_buffer = {
            parameter_meta.tensor_name: offset
            for parameter_meta, offset in zip(parameter_metas, parameter_offsets[:-1])
        }
        self._parameter_ends_in_buffer = {
            parameter_meta.tensor_name: offset
            for parameter_meta, offset in zip(parameter_metas, parameter_offsets[1:])
        }

        # Shard properties
        # We pad the stage so that each shard has the same size
        #   and is a multiple of SHARD_PAD_TO_MULTIPLE (for data alignment)
        self._global_pad = -self._parameter_count % (self._fsdp_dim.size * SHARD_PAD_TO_MULTIPLE)
        self._shard_size = (self._parameter_count + self._global_pad) // self._fsdp_dim.size
        # Typically the padding is all on the last shard, but in some cases it can overflow to other shards.
        self._shard_pad = min(
            max(self._global_pad - self._shard_size * (self._fsdp_dim.size - self._fsdp_dim.rank - 1), 0),
            self._shard_size,
        )

        # TODO: Use parallel_dim property instead?
        weight_shard_dim = TensorDim("weight_shard", (self._parameter_count + self._global_pad) // self._fsdp_dim.size)
        grad_shard_dim = TensorDim("grad_shard", weight_shard_dim.size if self._requires_grad else 0)

        self._weight_shard_meta = TensorMeta.from_dims(
            (weight_shard_dim,),
            tensor_name=f"{self._name}_weight_shard",
            dtype=self._optimization_dtype.torch,
        )
        # TODO: Distinguish grad and optimizer shard?
        self._grad_shard_meta = TensorMeta.from_dims(
            (grad_shard_dim,),
            tensor_name=f"{self._name}_grad_shard",
            dtype=self._optimization_dtype.torch,
        )
        self._weight_buffer_meta = TensorMeta.from_dims(
            (TensorDim("weight_buffer", weight_shard_dim.size * self._fsdp_dim.size),),
            tensor_name=f"{self._name}_weight_buffer",
            dtype=self._training_dtype.torch,
        )
        self._grad_buffer_meta = TensorMeta.from_dims(
            (TensorDim("grad_buffer", weight_shard_dim.size * self._fsdp_dim.size if self._requires_grad else 0),),
            tensor_name=f"{self._name}_grad_buffer",
            dtype=self._gradient_buffer_dtype.torch,
        )

    @property
    def parameter_names(self) -> list[str]:
        return list(self._parameter_metas)

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def weight_shard_meta(self) -> TensorMeta:
        return self._weight_shard_meta

    @property
    def grad_shard_meta(self) -> TensorMeta:
        return self._grad_shard_meta

    @property
    def weight_buffer_meta(self) -> TensorMeta:
        return self._weight_buffer_meta

    @property
    def grad_buffer_meta(self) -> TensorMeta:
        return self._grad_buffer_meta

    @property
    def weight_shard(self) -> torch.Tensor:
        # TODO: Avoid this method (needed for tied weights broadcast)
        assert self._is_setup
        assert self._mode.support_forward
        return self._weight_shard

    @property
    def grad_shard(self) -> torch.Tensor:
        # TODO: Avoid this method (needed for tied weights reduce)
        assert self._is_setup
        assert self._mode.support_backward
        return self._grad_shard

    def __contains__(self, parameter_name: str) -> bool:
        return parameter_name in self.parameter_names

    def get_parameter_meta(self, parameter_name: str) -> ParameterMeta:
        return self._parameter_metas[parameter_name]

    def get_parameter_buffer(self, parameter_name: str) -> torch.nn.Parameter:
        assert self._is_setup
        assert self._mode.support_forward
        return self._parameter_buffers[parameter_name]

    def get_parameter_begin_in_buffer(self, parameter_name: str) -> int:
        return self._parameter_begins_in_buffer[parameter_name]

    def get_parameter_end_in_buffer(self, parameter_name: str) -> int:
        return self._parameter_ends_in_buffer[parameter_name]

    def setup(
        self,
        mode: StageMode,
        fsdp_group: ProcessGroup,
        weight_shard: torch.Tensor | None,
        grad_shard: torch.Tensor | None,
        weight_buffer: torch.Tensor | None,
        grad_buffer: torch.Tensor | None,
        sequence_tensor_parallel: bool,
        device: torch.device | None,
    ) -> None:
        assert not self._is_setup
        self._is_setup = True
        self._fsdp_group = fsdp_group
        self._mode = mode

        # Validate and set the shards and buffers
        if self._mode.on_device:
            self._weight_shard = (
                torch.empty_like(self._weight_shard_meta, device=device)
                if weight_shard is None
                else self._weight_shard_meta.validate(weight_shard)
            )
        else:
            Assert.none(weight_shard)
        if self._mode.support_forward:
            self._weight_buffer = (
                torch.empty_like(self._weight_buffer_meta, device=device)
                if weight_buffer is None
                else self._weight_buffer_meta.validate(weight_buffer)
            )
            # Pre-compute the local shard for restore ops.
            self._weight_buffer_local_shard = self._weight_buffer[
                self._fsdp_dim.rank * self._shard_size : (self._fsdp_dim.rank + 1) * self._shard_size
            ]
        else:
            Assert.none(weight_buffer)

        if self._mode.support_backward:
            self._grad_shard = (
                torch.empty_like(self._grad_shard_meta, device=device)
                if grad_shard is None
                else self._grad_shard_meta.validate(grad_shard)
            )
            self._grad_buffer = (
                torch.empty_like(self._grad_buffer_meta, device=device)
                if grad_buffer is None
                else self._grad_buffer_meta.validate(grad_buffer)
            )
            # Pre-compute the local shard for reduce ops.
            self._grad_buffer_local_shard = self._grad_buffer[
                self._fsdp_dim.rank * self._shard_size : (self._fsdp_dim.rank + 1) * self._shard_size
            ]
            # Pre-compute the sequence-parallel grads.
            sp_indices = [i for i, meta in enumerate(self._parameter_metas.values()) if meta.sequence_tensor_parallel]
            if sp_indices and sequence_tensor_parallel:
                Assert.eq(sp_indices, list(range(sp_indices[0], sp_indices[-1] + 1)))
                sp_indices = [
                    i for i, meta in enumerate(self._parameter_metas.values()) if meta.sequence_tensor_parallel
                ]
                sp_begin, sp_end = (
                    list(self._parameter_begins_in_buffer.values())[sp_indices[0]],
                    list(self._parameter_ends_in_buffer.values())[sp_indices[-1]],
                )
            else:
                sp_begin, sp_end = 0, 0
            self._sequence_parallel_grads = self._grad_buffer[sp_begin:sp_end] if sp_end > sp_begin else None

        else:
            Assert.none(grad_shard)
            Assert.none(grad_buffer)

        if self._mode.support_forward:
            # Precompute the buffer slice for each parameter.
            # Use `.data` to hide the restore ops from autograd.
            self._parameter_buffers = {}
            for parameter_name in self._parameter_metas:
                parameter_buffer = torch.nn.Parameter(
                    self._get_parameter_in_buffer(self._weight_buffer.data, parameter_name),
                    requires_grad=self._mode.support_backward and self._parameter_metas[parameter_name].requires_grad,
                )
                if self._mode.support_backward and self._requires_grad:
                    parameter_buffer.grad_buffer = self._get_parameter_in_buffer(
                        self._grad_buffer.data, parameter_name
                    )
                self._parameter_buffers[parameter_name] = parameter_buffer

    def reset_shard_pad(self, shard: torch.Tensor, shard_name: str) -> int:
        assert self._is_setup
        assert self._mode.on_device
        # TODO: Needed?
        # Prevent nans with the padded values
        # Also ensures a correct parameter count in loading context.
        shard_meta = self._weight_shard_meta if shard_name == ShardName.weights else self._grad_shard_meta
        shard_meta.validate(shard)
        if self._shard_pad > 0:
            shard[-self._shard_pad :].zero_()
            return self._shard_pad
        return 0

    def split_buffer(self, buffer: torch.Tensor) -> dict[str, torch.Tensor]:
        # Split a buffer into appropriately shaped parameters.
        return {name: self._get_parameter_in_buffer(buffer, name) for name in self._parameter_metas}

    def _get_parameter_in_buffer(self, buffer: torch.Tensor, name: str) -> torch.Tensor:
        return buffer[self.get_parameter_begin_in_buffer(name) : self.get_parameter_end_in_buffer(name)].view(
            self._parameter_metas[name].shape
        )

    def split_shard(self, shard: torch.Tensor) -> dict[str, torch.Tensor]:
        # Split a shard into flat (possibly empty) parameter slices.
        return {
            name: shard[
                self.index_buffer_to_shard(self.get_parameter_begin_in_buffer(name)) : self.index_buffer_to_shard(
                    self.get_parameter_end_in_buffer(name)
                )
            ]
            for name in self._parameter_metas
        }

    def index_buffer_to_shard(self, index: int, rank: int | None = None) -> int:
        shard_begin = (self._fsdp_dim.rank if rank is None else rank) * self._shard_size
        return clamp(index - shard_begin, 0, self._shard_size - self._shard_pad)

    def _index_buffer_to_param(self, index: int, parameter_name: str) -> int:
        return clamp(
            index - self.get_parameter_begin_in_buffer(parameter_name),
            0,
            self._parameter_metas[parameter_name].numel(),
        )

    def reconstruct_from_shard(self, local_shard: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return gather_op(local_shard, group=self._fsdp_group, dim=0, out=out)

    def import_state_tensor(
        self, parameter_name: str, shard: torch.Tensor, tensor: torch.Tensor | SafeTensorSlice
    ) -> int:
        """
        Given a global parameter tensor, set the associated slice of a local parameter shard.
        Return the size of the local slice.
        """
        Assert.eq(shard.shape, (self._shard_size,))
        tensor_shard = self.parameter_global_to_shard(tensor, parameter_name)
        begin, end = self._parameter_range_in_shard(parameter_name)
        Assert.eq(tensor_shard.numel(), end - begin)
        shard[begin:end].copy_(tensor_shard)
        return end - begin

    def export_shard(
        self, shard: torch.Tensor, distributed: Distributed, data_type: DataType | None = None
    ) -> typing.Generator[tuple[str, torch.Tensor], None, None]:
        if data_type is not None:
            shard = shard.to(dtype=data_type.torch)
        tensors = self.split_buffer(self.reconstruct_from_shard(shard))
        for name, meta in self._parameter_metas.items():
            yield name, meta.local_to_global(tensors[name], distributed=distributed)[0]

    def log_shard(self, name, shard, *, distributed: Distributed, level, global_: bool) -> None:
        # if global_ is None:
        #    global_ = self._config.debug_global_tensors
        parameters = self.split_buffer(self.reconstruct_from_shard(shard)) if global_ else self.split_shard(shard)
        for parameter_name, parameter in parameters.items():
            log_distributed_tensor(
                name,
                parameter,
                level=level,
                distributed=distributed,
                global_=global_,
                duplicate_groups=(distributed.data_group,),
                meta=self.get_parameter_meta(parameter_name),
            )

    def restore_parameters(self) -> None:
        assert self._is_setup
        assert self._mode.support_forward
        # TODO: Allow partial FSDP
        if not self._is_restored:
            triton_copy(self._weight_shard, self._weight_buffer_local_shard)
            if self._fsdp_dim.size > 1:
                self.reconstruct_from_shard(self._weight_buffer_local_shard, self._weight_buffer)
            self._is_restored = True

    def reset_gradients(self) -> None:
        # TODO: Allow re-allocating the gradient every time.
        assert self._is_setup
        assert self._mode.support_backward
        if not self._requires_grad:
            return
        for buffer in self._parameter_buffers.values():
            assert buffer.grad is None
            buffer.param_grad_is_zero = True

    def reduce_gradients(
        self, distributed: Distributed, accumulate: bool = False, allow_no_grad: bool = False
    ) -> None:
        # Reduce the buffer, then copy (add) to actual grads.
        # Run in a separate cuda stream to allow communication overlap.
        # TODO: Allow partial FSDP
        assert self._is_restored
        assert self._mode.support_backward
        if not self._requires_grad:
            return
        for buffer, meta in zip(self._parameter_buffers.values(), self._parameter_metas.values()):
            if buffer.param_grad_is_zero:  # noqa
                assert allow_no_grad or meta.allow_no_grad, meta
                triton_fill(buffer.grad_buffer, 0)  # noqa
        if self._sequence_parallel_grads is not None and distributed.tensor_group:
            all_reduce(self._sequence_parallel_grads, group=distributed.tensor_group)
        if self._fsdp_dim.size > 1:
            full_precision_gradients = self._grad_buffer_local_shard.dtype == self._grad_shard.dtype
            out = self._grad_shard if full_precision_gradients else self._grad_buffer_local_shard
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
            elif not full_precision_gradients:
                triton_copy(self._grad_buffer_local_shard, self._grad_shard)
        else:
            triton_copy(self._grad_buffer_local_shard, self._grad_shard)

    def _parameter_range_in_shard(self, parameter_name: str) -> tuple[int, int]:
        begin = self.index_buffer_to_shard(self.get_parameter_begin_in_buffer(parameter_name))
        end = self.index_buffer_to_shard(self.get_parameter_end_in_buffer(parameter_name))
        return begin, end

    def invalidate_buffer(self) -> None:
        # Buffer is no longer valid (Updated weights or overwritten by other stage)
        assert self._mode.support_forward
        self._is_restored = False

    def parameter_global_to_shard(
        self, global_param: torch.Tensor | SafeTensorSlice, parameter_name: str
    ) -> torch.Tensor:
        shard_param = self.get_parameter_meta(parameter_name).global_to_local(global_param).flatten()
        if self._fsdp_dim.size > 1:
            shard_param = shard_param[
                self._index_buffer_to_param(
                    self._fsdp_dim.rank * self._shard_size, parameter_name
                ) : self._index_buffer_to_param((self._fsdp_dim.rank + 1) * self._shard_size, parameter_name)
            ]
        return shard_param

    def _get_parameter_shard_indices_in_full_weight(self, parameter_name: str, device: torch.device) -> torch.Tensor:
        """
        Create an index array for the global parameter, where each entry corresponds to the index
        where it is located in the shard if it exists, or -1 if it's not in the shard.
        Used to determine the location of each entry in a different distributed configuration.
        """
        parameter_meta = self.get_parameter_meta(parameter_name)

        # Create an empty index for the global parameter.
        index = torch.full(
            parameter_meta.global_shape,
            -1,
            dtype=torch.int64,
            device=device,
        )
        # Set the shard slice of the global parameter to corresponding indices of the parameter slice of the shard
        begin, end = self._parameter_range_in_shard(parameter_name)
        self.parameter_global_to_shard(index, parameter_name).copy_(
            torch.arange(begin, end, dtype=torch.int64, device=device)
        )
        return index

    def copy_shard_overlaps(
        self,
        loaded_fsdp: "FSDP",
        shards: dict[str, torch.Tensor],
        loaded_shards: dict[str, torch.Tensor],
        counter: torch.Tensor,
        device: torch.device,
    ) -> None:
        """
        See MultiStage._load_partial.
        TODO: Not intended to work with frozen weights, need to enforce.
        """
        Assert.eq(set(shards), set(loaded_shards))
        index_overlap = [name for name in loaded_fsdp._parameter_metas if name in self._parameter_metas]
        for name in index_overlap:
            overlap_index_map = self.parameter_global_to_shard(
                loaded_fsdp._get_parameter_shard_indices_in_full_weight(name, device), name
            )
            overlap_mask = overlap_index_map >= 0
            overlap_index_map_masked = overlap_index_map[overlap_mask]
            overlap_count = overlap_mask.sum()
            begin, end = self._parameter_range_in_shard(name)

            for shard_name, shard in shards.items():
                # Shards can be empty (frozen weights)
                if shard.numel() == 0:
                    continue
                if loaded_shards[shard_name].numel() == 0:
                    shard[begin:end][overlap_mask] = 0
                    counter += overlap_count
                    continue
                shard[begin:end][overlap_mask] = loaded_shards[shard_name][overlap_index_map_masked]
                counter += overlap_count
