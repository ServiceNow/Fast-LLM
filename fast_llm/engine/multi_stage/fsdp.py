import dataclasses
import math
import typing

import torch
from torch._C._distributed_c10d import ReduceOp
from torch.distributed import all_reduce, reduce_scatter_tensor

from fast_llm.core.distributed import ProcessGroup
from fast_llm.core.ops import gather_op
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDim, DistributedDimNames
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import SHARD_PAD_TO_MULTIPLE, ShardName, StageMode
from fast_llm.functional.triton.pointwise import triton_add, triton_copy, triton_fill
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
        distributed_config: DistributedConfig,
        full_precision_gradient_buffer: bool = False,
        full_precision_shards: bool = True,
        is_tied_weight_copy: bool = False,
    ):
        self._name = name
        self._parameter_metas = {parameter_meta.tensor_name: parameter_meta for parameter_meta in parameter_metas}
        self._distributed_config = distributed_config
        self._fsdp_dim = self._distributed_config.get_distributed_dim(DistributedDimNames.data)
        self._is_tied_weight_copy = is_tied_weight_copy
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
        weight_shard_dim = TensorDim("weight_shard", self._shard_size)
        grad_shard_dim = TensorDim("grad_shard", self._shard_size if self._requires_grad else 0)

        self._weight_shard_meta = TensorMeta.from_dims(
            (weight_shard_dim,),
            tensor_name=f"{self._name}_weight_shard",
            dtype=(
                self._distributed_config.optimization_dtype
                if full_precision_shards
                else self._distributed_config.training_dtype
            ).torch,
        )
        # TODO: Distinguish grad and optimizer shard?
        self._grad_shard_meta = TensorMeta.from_dims(
            (grad_shard_dim,),
            tensor_name=f"{self._name}_grad_shard",
            dtype=(
                self._distributed_config.optimization_dtype
                if full_precision_shards
                else self._distributed_config.training_dtype
            ).torch,
        )
        self._weight_buffer_meta = TensorMeta.from_dims(
            (TensorDim("weight_buffer", weight_shard_dim.size * self._fsdp_dim.size),),
            tensor_name=f"{self._name}_weight_buffer",
            dtype=self._distributed_config.training_dtype.torch,
        )
        self._grad_buffer_meta = TensorMeta.from_dims(
            (TensorDim("grad_buffer", weight_shard_dim.size * self._fsdp_dim.size if self._requires_grad else 0),),
            tensor_name=f"{self._name}_grad_buffer",
            dtype=(
                self._distributed_config.optimization_dtype
                if full_precision_gradient_buffer
                else self._distributed_config.training_dtype
            ).torch,
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
        begin, end = self._get_parameter_range_in_shard(parameter_name)
        Assert.eq(tensor_shard.numel(), end - begin)
        shard[begin:end].copy_(tensor_shard)
        return end - begin

    def export_shard(
        self, shard: torch.Tensor, data_type: DataType | None = None
    ) -> typing.Generator[tuple[str, torch.Tensor], None, None]:
        if data_type is not None:
            shard = shard.to(dtype=data_type.torch)
        tensors = self.split_buffer(self.reconstruct_from_shard(shard))
        for name, meta in self._parameter_metas.items():
            yield name, meta.local_to_global(tensors[name])[0]

    def log_shard(self, name, shard, *, distributed: Distributed, level, global_: bool) -> None:
        # if global_ is None:
        #    global_ = self._config.debug_global_tensors
        parameters = self.split_buffer(self.reconstruct_from_shard(shard)) if global_ else self.split_shard(shard)
        for parameter_name, parameter in parameters.items():
            meta = self.get_parameter_meta(parameter_name)
            log_distributed_tensor(
                name,
                parameter,
                level=level,
                global_=global_,
                # Assuming all tensors are either duplicated of parallel in the TP direction.
                duplicate_groups=(
                    distributed.data_group,
                    distributed.tensor_group,
                ),
                meta=meta,
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

    def _get_parameter_range_in_shard(self, parameter_name: str) -> tuple[int, int]:
        begin = self.index_buffer_to_shard(self.get_parameter_begin_in_buffer(parameter_name))
        end = self.index_buffer_to_shard(self.get_parameter_end_in_buffer(parameter_name))
        return begin, end

    def get_parameter_size_in_shard(self, parameter_name: str, shard_name: str = ShardName.weights) -> int:
        if not self._requires_grad and shard_name != ShardName.weights:
            return 0
        begin, end = self._get_parameter_range_in_shard(parameter_name)
        return end - begin

    def invalidate_buffer(self) -> None:
        # Buffer is no longer valid (Updated weights or overwritten by other stage)
        assert self._mode.support_forward
        self._is_restored = False

    def parameter_global_to_shard(
        self,
        global_param: torch.Tensor | SafeTensorSlice,
        parameter_name: str,
        *,
        _parameter_meta: TensorMeta | None = None,
    ) -> torch.Tensor:
        if _parameter_meta is None:
            # Used with reduced tensor-parallel in `copy_shard_overlaps`
            _parameter_meta = self._parameter_metas[parameter_name]
        # This may copy the data.
        shard_param = _parameter_meta.global_to_local(global_param).flatten()
        if self._fsdp_dim.size > 1:
            shard_param = shard_param[
                self._index_buffer_to_param(
                    self._fsdp_dim.rank * self._shard_size, parameter_name
                ) : self._index_buffer_to_param((self._fsdp_dim.rank + 1) * self._shard_size, parameter_name)
            ]
        return shard_param

    def _get_parameter_shard_indices_in_full_weight(
        self, parameter_name: str, device: torch.device, parameter_meta: TensorMeta
    ) -> torch.Tensor:
        """
        Create an index array for the global parameter, where each entry corresponds to the index
        where it is located in the shard if it exists, or -1 if it's not in the shard.
        Used to determine the location of each entry in a different distributed configuration.
        """
        # Set the shard slice of the global parameter to corresponding indices of the parameter slice of the shard
        begin, end = self._get_parameter_range_in_shard(parameter_name)

        # Create an empty local index to hold the local shard indices.
        buffer_index = torch.full_like(parameter_meta, -1, dtype=torch.int64, device=device)

        # Copy the shard indices at their respective positions in the buffer index.
        buffer_index.flatten()[
            self._index_buffer_to_param(
                self._fsdp_dim.rank * self._shard_size, parameter_name
            ) : self._index_buffer_to_param((self._fsdp_dim.rank + 1) * self._shard_size, parameter_name)
        ].copy_(torch.arange(begin, end, dtype=torch.int64, device=device))

        # Create a global index from the local one.
        return parameter_meta.local_to_global_partial(buffer_index, -1)

    def copy_shard_overlaps(
        self,
        loaded_fsdp: typing.Self,
        shards: dict[str, torch.Tensor] | None,
        loaded_shards: dict[str, torch.Tensor] | None,
    ) -> dict[tuple[str, str], int]:
        """
        See MultiStage._load_partial.
        """
        if shards is not None:
            Assert.eq(set(shards), set(loaded_shards))
        index_overlap = [name for name in loaded_fsdp._parameter_metas if name in self._parameter_metas]
        counter = {}

        self_tensor_dim = self._distributed_config.get_distributed_dim(DistributedDimNames.tensor)
        loaded_tensor_dim = loaded_fsdp._distributed_config.get_distributed_dim(DistributedDimNames.tensor)

        # The shared tensor-parallel part (usually the smallest of the two) can be safely ignored.
        if (shared_tp := math.gcd(self_tensor_dim.size, loaded_tensor_dim.size)) > 1:
            self_tensor_dim, self_new_size, self_shared_rank = _reduce_tensor_parallel_size(self_tensor_dim, shared_tp)
            loaded_tensor_dim, loaded_new_size, loaded_shared_rank = _reduce_tensor_parallel_size(
                loaded_tensor_dim, shared_tp
            )

            if self_shared_rank != loaded_shared_rank:
                # Disjoint tensor-parallel slices, no possible overlap.
                #   (Duplicated parameters will be loaded from the new rank 0 which prevents unnecessary file loading).
                return counter

        for parameter_name in index_overlap:
            self_meta = self._parameter_metas[parameter_name]
            loaded_meta = loaded_fsdp._parameter_metas[parameter_name]

            if shared_tp > 1:
                self_meta = self_meta.replace_tensor_parallel_dim(self_tensor_dim)
                loaded_meta = loaded_meta.replace_tensor_parallel_dim(loaded_tensor_dim)

            if not loaded_meta.is_tensor_parallel and loaded_tensor_dim.rank != 0:
                # Loaded parameter is tensor-parallel duplicate, ignore.
                continue

            if self_meta.tensor_parallel_size == loaded_meta.tensor_parallel_size == 1:
                self._copy_shard_overlaps(loaded_fsdp, shards, loaded_shards, parameter_name, counter)
            else:
                self._copy_tensor_parallel_shard_overlaps(
                    loaded_fsdp, shards, loaded_shards, parameter_name, counter, self_meta, loaded_meta
                )

        return counter

    def _copy_shard_overlaps(
        self,
        loaded_fsdp: typing.Self,
        shards: dict[str, torch.Tensor] | None,
        loaded_shards: dict[str, torch.Tensor] | None,
        parameter_name: str,
        counter: dict[tuple[str, str], int],
    ):
        # Common case: the overlap is a contiguous slice of the shards.

        # Find the slice of the parameter contained in each shard.
        self_shard_begin_in_buffer = self._fsdp_dim.rank * self._shard_size
        self_shard_end_in_buffer = (self._fsdp_dim.rank + 1) * self._shard_size
        self_shard_begin_in_param = self._index_buffer_to_param(self_shard_begin_in_buffer, parameter_name)
        self_shard_end_in_param = self._index_buffer_to_param(self_shard_end_in_buffer, parameter_name)
        loaded_shard_begin_in_buffer = loaded_fsdp._fsdp_dim.rank * loaded_fsdp._shard_size
        loaded_shard_end_in_buffer = (loaded_fsdp._fsdp_dim.rank + 1) * loaded_fsdp._shard_size
        loaded_shard_begin_in_param = loaded_fsdp._index_buffer_to_param(loaded_shard_begin_in_buffer, parameter_name)
        loaded_shard_end_in_param = loaded_fsdp._index_buffer_to_param(loaded_shard_end_in_buffer, parameter_name)

        # Calculate the overap.
        overlap_begin_in_param = max(self_shard_begin_in_param, loaded_shard_begin_in_param)
        overlap_end_in_param = min(self_shard_end_in_param, loaded_shard_end_in_param)

        if (overlap_size := overlap_end_in_param - overlap_begin_in_param) <= 0:
            return

        # Map the overlap back to the shards.
        overlap_begin_in_self_shard = (
            self._parameter_begins_in_buffer[parameter_name] + overlap_begin_in_param - self_shard_begin_in_buffer
        )
        overlap_begin_in_loaded_shard = (
            loaded_fsdp._parameter_begins_in_buffer[parameter_name]
            + overlap_begin_in_param
            - loaded_shard_begin_in_buffer
        )

        if shards is None:
            # Dry run.
            counter[(parameter_name, "")] = overlap_size
            return

        for shard_name, shard in shards.items():
            # Shards can be empty (frozen weights)
            if shard.numel() == 0:
                continue
            counter[(parameter_name, shard_name)] = overlap_size

            # Copy the overlap.
            shard[overlap_begin_in_self_shard : overlap_begin_in_self_shard + overlap_size] = (
                loaded_shards[shard_name][overlap_begin_in_loaded_shard : overlap_begin_in_loaded_shard + overlap_size]
                if loaded_shards[shard_name].numel() > 0
                else 0
            )

    def _copy_tensor_parallel_shard_overlaps(
        self,
        loaded_fsdp: typing.Self,
        shards: dict[str, torch.Tensor] | None,
        loaded_shards: dict[str, torch.Tensor] | None,
        parameter_name: str,
        counter: dict[tuple[str, str], int],
        self_meta: TensorMeta,
        loaded_meta: TensorMeta,
    ):

        self_begin, self_end = self._get_parameter_range_in_shard(parameter_name)
        loaded_begin, loaded_end = loaded_fsdp._get_parameter_range_in_shard(parameter_name)
        if self_begin >= self_end or loaded_begin >= loaded_end:
            # Parameter is not present in both shards, no overlap.
            return

        # Tensor-parallel case: the overlap cannot be represented as a slice.
        if shards is None:
            # Dry run. Since we only need to know if there can be overlap,
            #   we skip the slow computation and return a dummy value.
            counter[(parameter_name, "")] = 1
            return

        device = next(iter(shards.values())).device
        # Create an array that associates each entry in the `parameter_name` slice of `shard`
        #   to the index of the same parameter entry in `loaded_shard`, or -1 if not present.
        overlap_index_map = self.parameter_global_to_shard(
            loaded_fsdp._get_parameter_shard_indices_in_full_weight(parameter_name, device, loaded_meta),
            parameter_name,
            _parameter_meta=self_meta,
        )
        # Create a mask to exclude the missing entries.
        overlap_mask = overlap_index_map >= 0
        overlap_index_map_masked = overlap_index_map[overlap_mask]
        overlap_size = overlap_mask.sum().item()
        if overlap_size == 0:
            return
        begin, end = self._get_parameter_range_in_shard(parameter_name)

        for shard_name, shard in shards.items():
            # Shards can be empty (frozen weights)
            if shard.numel() == 0:
                continue
            counter[(parameter_name, shard_name)] = overlap_size
            # Masked copy of the overlap index map.
            shard[begin:end][overlap_mask] = (
                loaded_shards[shard_name][overlap_index_map_masked] if loaded_shards[shard_name].numel() > 0 else 0
            )


def _reduce_tensor_parallel_size(distributed_dim: DistributedDim, shared_size: int):
    new_size = distributed_dim.size // shared_size
    shared_rank = distributed_dim.rank // new_size
    new_dim = dataclasses.replace(
        distributed_dim,
        size=new_size,
        rank=distributed_dim.rank % new_size,
        global_ranks=distributed_dim.global_ranks[shared_size * shared_rank : shared_size * (shared_rank + 1)],
    )
    return new_dim, new_size, shared_rank
