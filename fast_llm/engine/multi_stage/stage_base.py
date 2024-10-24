import logging
import typing

import torch
import torch._dynamo  # noqa

from fast_llm.core.distributed import ProcessGroup, check_parallel_match
from fast_llm.core.ops import gather_op
from fast_llm.engine.base_model.base_model import BaseModel
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.tensor_space import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import SHARD_PAD_TO_MULTIPLE, StageConfig, StageMode
from fast_llm.engine.optimizer.config import ParamGroup
from fast_llm.logging import log_distributed_tensor, log_generator
from fast_llm.tensor import ParameterMeta, SafeTensorSlice, TensorMeta
from fast_llm.utils import Assert, clamp, div, padded_cumsum

logger = logging.getLogger(__name__)


class StageBase:
    _meta_inputs: list[TensorMeta]
    _meta_outputs: list[TensorMeta]
    _distributed: Distributed
    _fsdp_group: ProcessGroup
    _mode: StageMode

    _weight_shard: torch.Tensor
    _grad_shard: torch.Tensor
    _weight_buffer: torch.Tensor
    _grad_buffer: torch.Tensor
    _sequence_parallel_grads: torch.Tensor
    _weight_buffer_local_shard: torch.Tensor
    _grad_buffer_local_shard: torch.Tensor
    _parameter_buffers: list[torch.nn.Parameter]

    def __init__(
        self,
        *,
        base_model: BaseModel,
        config: StageConfig,
        distributed_config: DistributedConfig,
        begin: int,
        end: int,
        index: int,
    ):
        self._config = config.validate()
        self._distributed_config = distributed_config.validate()

        Assert.in_range(begin, 0, end)
        Assert.leq(end, len(base_model))

        self._fsdp_rank = self._distributed_config.data_rank
        self._fsdp_size = self._distributed_config.data_parallel
        self._is_setup = False
        self._index = index

        self._layers = [torch.compile(layer) if self._config.compile_all else layer for layer in base_model[begin:end]]
        self._layer_range = list(range(begin, end))

        self._parameter_metas: list[ParameterMeta] = self._get_parameter_metas()

        self._parameter_index = {param.tensor_name: i for i, param in enumerate(self._parameter_metas)}

        parameter_sizes = [meta.numel() for meta in self._parameter_metas]
        self._parameter_count = sum(parameter_sizes)
        parameter_offsets = padded_cumsum(parameter_sizes).tolist()

        # The index range of the parameters in the buffer.
        self._parameter_begins_in_buffer = parameter_offsets[:-1]
        self._parameter_ends_in_buffer = parameter_offsets[1:]

        # Shard properties
        # We pad the stage so that each shard has the same size
        #   and is a multiple of SHARD_PAD_TO_MULTIPLE (for data alignment)
        self._global_pad = -self._parameter_count % (self._fsdp_size * SHARD_PAD_TO_MULTIPLE)
        self._shard_size = (self._parameter_count + self._global_pad) // self._fsdp_size
        # Typically the padding is all on the last shard, but in some cases it can overflow to other shards.
        self._shard_pad = min(
            max(self._global_pad - self._shard_size * (self._fsdp_size - self._fsdp_rank - 1), 0), self._shard_size
        )

        # TODO: Use parallel_dim property instead?
        shard_dim = TensorDim("flat_shard", (self._parameter_count + self._global_pad) // self._fsdp_size)
        buffer_dim = TensorDim("flat_buffer", shard_dim.size * self._fsdp_size)

        self._weight_shard_meta = TensorMeta.from_dims(
            (shard_dim,),
            tensor_name=f"stage_{self._index}_weight_shard",
            dtype=self._distributed_config.optimization_dtype.torch,
        )
        self._grad_shard_meta = TensorMeta.from_dims(
            (shard_dim,),
            tensor_name=f"stage_{self._index}_grad_shard",
            dtype=self._distributed_config.optimization_dtype.torch,
        )
        self._weight_buffer_meta = TensorMeta.from_dims(
            (buffer_dim,),
            tensor_name=f"stage_{self._index}_weight_buffer",
            dtype=self._distributed_config.training_dtype.torch,
        )
        self._grad_buffer_meta = TensorMeta.from_dims(
            (buffer_dim,),
            tensor_name=f"stage_{self._index}_weight_buffer",
            dtype=(
                self._distributed_config.optimization_dtype
                if self._config.full_precision_gradients
                else self._distributed_config.training_dtype
            ).torch,
        )

    @property
    def mode(self):
        assert self._is_setup
        return self._mode

    @property
    def index(self):
        return self._index

    @property
    def weight_shard_meta(self):
        return self._weight_shard_meta

    @property
    def grad_shard_meta(self):
        return self._grad_shard_meta

    @property
    def weight_buffer_meta(self):
        return self._weight_buffer_meta

    @property
    def grad_buffer_meta(self):
        return self._grad_buffer_meta

    @property
    def weight_shard(self):
        # TODO: Avoid this method (needed for tied weights broadcast)
        assert self._is_setup
        assert self._mode.support_forward
        return self._weight_shard

    @property
    def grad_shard(self):
        # TODO: Avoid this method (needed for tied weights reduce)
        assert self._is_setup
        assert self._mode.support_backward
        return self._grad_shard

    @property
    def parameter_count(self):
        return self._parameter_count

    @property
    def parameter_names(self):
        return list(self._parameter_index)

    def get_parameter_meta(self, parameter_name: str):
        return self._parameter_metas[self._parameter_index[parameter_name]]

    def get_parameter_buffer(self, meta: ParameterMeta):
        assert self._is_setup
        assert self._mode.support_forward
        return self._parameter_buffers[self._parameter_index[meta.tensor_name]]

    def setup(
        self,
        *,
        distributed: Distributed,
        weight_shard: torch.Tensor | None,
        grad_shard: torch.Tensor | None,
        weight_buffer: torch.Tensor | None,
        grad_buffer: torch.Tensor | None,
        mode: StageMode = StageMode.training,
    ):
        assert not self._is_setup
        assert distributed.config is self._distributed_config
        self._is_setup = True
        self._distributed = distributed
        self._fsdp_group = self._distributed.data_group
        self._mode = mode

        # Validate and set the shards and buffers
        if self._mode.on_device:
            self._weight_shard = self._weight_shard_meta.validate(weight_shard)
        else:
            Assert.none(weight_shard)
        if self._mode.support_forward:
            self._weight_buffer = self._weight_buffer_meta.validate(weight_buffer)
            # Pre-compute the local shard for restore ops.
            self._weight_buffer_local_shard = self._weight_buffer[
                self._fsdp_rank * self._shard_size : (self._fsdp_rank + 1) * self._shard_size
            ]
        else:
            Assert.none(weight_buffer)

        if self._mode.support_backward:
            self._grad_shard = self._grad_shard_meta.validate(grad_shard)
            self._grad_buffer = self._grad_buffer_meta.validate(grad_buffer)
            # Pre-compute the local shard for reduce ops.
            self._grad_buffer_local_shard = self._grad_buffer[
                self._fsdp_rank * self._shard_size : (self._fsdp_rank + 1) * self._shard_size
            ]
            # Pre-compute the sequence-parallel grads.
            sp_indices = [i for i, meta in enumerate(self._parameter_metas) if meta.sequence_tensor_parallel]
            if sp_indices and self._distributed_config.sequence_tensor_parallel:
                Assert.eq(sp_indices, list(range(sp_indices[0], sp_indices[-1] + 1)))
                sp_begin, sp_end = (
                    self._parameter_begins_in_buffer[sp_indices[0]],
                    self._parameter_ends_in_buffer[sp_indices[-1]],
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
            self._parameter_buffers = []
            for weight_buffer, grad_buffer, meta in zip(
                self._split_buffer(self._weight_buffer.data),
                self._split_buffer(self._grad_buffer if self._mode.support_backward else self._grad_buffer_meta),
                self._parameter_metas,
            ):
                parameter_buffer = torch.nn.Parameter(weight_buffer, requires_grad=self._mode.support_backward)
                if self._mode.support_backward:
                    parameter_buffer.grad_buffer = grad_buffer
                # TODO: This is only needed for Megatron initialization
                self._parameter_buffers.append(parameter_buffer)

            # Replace the parameter definitions in each module with the actual parameter buffers.
            def _replace(module: torch.nn.Module):
                nonlocal i
                for key in module._parameters:  # noqa
                    meta = typing.cast(ParameterMeta, module._parameters[key])  # noqa
                    module._parameters[key] = self._parameter_buffers[self._parameter_index[meta.tensor_name]]  # noqa
                    i += 1

            i = 0
            for layer in self._layers:
                layer.apply(_replace)

            Assert.eq(i, len(self._parameter_metas))

    def initialize_weights(self):
        # TODO: Avoid all the _on_device checks
        assert self._is_setup
        with torch.no_grad():
            if self._config.debug_param_init:
                log_generator("CPU generator before reset", torch.random.default_generator)
                log_generator("PP init generator before reset", self._distributed.pp_init_generator)
                log_generator("TP init generator before reset", self._distributed.tp_init_generator)

            index = range(len(self._parameter_metas))
            if self._distributed_config.reproducible_init:
                # Ensure a reproducible ordering.
                index = sorted(index, key=lambda j: self._parameter_metas[j].tensor_name)
            weight_shard_split = self._split_shard(
                self._weight_shard if self._mode.on_device else self._weight_shard_meta
            )

            for i in index:
                parameter = weight_shard_split[i]
                meta = self._parameter_metas[i]
                # Multi-gpu init may be different because of TP or FSDP (different shape), or PP (not on device)
                global_shape = meta.global_shape

                if self._distributed_config.reproducible_init and (
                    global_shape.numel() != parameter.numel() or not self._mode.on_device
                ):
                    # Initialize all global weights on every gpu, then select the appropriate slice if applicable.
                    global_param = parameter.new_empty(global_shape, device=self._distributed.device)
                    meta.init_parameter(global_param, distributed=self._distributed)
                    if self._mode.on_device:
                        parameter.copy_(self._parameter_global_to_shard(global_param, i))
                elif self._mode.on_device:
                    meta.init_parameter(parameter, self._distributed)

            if self.mode.on_device:
                self.reset_shard_pad(self._weight_shard)

        if self._config.debug_param_init:
            log_generator("CPU generator after reset", torch.random.default_generator)
            log_generator("PP init generator after reset", self._distributed.pp_init_generator)
            log_generator("TP init generator after reset", self._distributed.tp_init_generator)
            if self._mode.on_device:
                self.log_shard(
                    name="param",
                    shard=self._weight_shard,
                    level=self._config.debug_param_init,
                )

    def reset_shard_pad(self, shard: torch.Tensor):
        assert self._is_setup
        assert self._mode.on_device
        # TODO: Needed?
        # Prevent nans with the padded values
        # Also ensures a correct parameter count in loading context.
        self._weight_shard_meta.validate(shard)
        if self._shard_pad > 0:
            shard[-self._shard_pad :].zero_()
            return self._shard_pad
        return 0

    def log_shard(self, name, shard, *, level, global_=None):
        if global_ is None:
            global_ = self._config.debug_global_tensors
        parameters = self._split_buffer(self._reconstruct_from_shard(shard)) if global_ else self._split_shard(shard)
        for parameter, meta in zip(parameters, self._parameter_metas):
            log_distributed_tensor(
                name,
                parameter,
                level=level,
                distributed=self._distributed,
                global_=global_,
                duplicate_groups=(self._distributed.data_group,),
                meta=meta,
            )

    def get_param_groups(
        self, optimizer_state_shards: dict[str, torch.Tensor], param_group_cls: type[ParamGroup]
    ) -> tuple[list[ParamGroup], list[torch.Tensor]]:
        # TODO: Separate model-specific code.
        # TODO: verify optimizer states
        assert self._is_setup
        assert self._mode.support_training

        # Get the weight slices and group by optimizer parameters, merging consecutive slices.
        grouped_parameter_slices = {}
        for meta in self._parameter_metas:
            # If needed, chunk the parameter on the first dimension.
            chunk_size = div(meta.numel(), len(meta.lr_scale))
            param_index = self._parameter_index[meta.tensor_name]
            buffer_begin = self._parameter_begins_in_buffer[param_index]
            for i, lr_scale in enumerate(meta.lr_scale):
                begin = self._index_buffer_to_shard(buffer_begin + i * chunk_size)
                end = self._index_buffer_to_shard(buffer_begin + (i + 1) * chunk_size)
                if lr_scale == 0 or begin == end:
                    continue
                optimizer_params = (meta.param_weight_decay, lr_scale)
                if optimizer_params in grouped_parameter_slices:
                    last_slice = grouped_parameter_slices[optimizer_params][-1]
                    if begin == last_slice.stop:
                        grouped_parameter_slices[optimizer_params][-1] = slice(last_slice.start, end)
                        continue
                else:
                    grouped_parameter_slices[optimizer_params] = []
                grouped_parameter_slices[optimizer_params].append(slice(begin, end))

        param_groups = [
            param_group_cls(
                name=f"wd_{weight_decay}_lr_scale_{lr_scale}",  # noqa
                params=[self._weight_shard[slice_] for slice_ in slices],  # noqa
                grads=[self._grad_shard[slice_] for slice_ in slices],  # noqa
                **{  # noqa
                    name: [optimizer_state[slice_] for slice_ in slices]
                    for name, optimizer_state in optimizer_state_shards.items()
                },
                weight_decay=None if weight_decay else 0.0,  # noqa
                lr_scale=lr_scale,  # noqa
            )
            for (weight_decay, lr_scale), slices in grouped_parameter_slices.items()
        ]

        # Get the weight slices to use for grad norm computation, merging consecutive slices.
        grad_norm_indices = (
            list(range(len(self._parameter_metas)))
            if self._distributed_config.tensor_rank == 0
            else [i for i, meta in enumerate(self._parameter_metas) if meta.is_tensor_parallel]
        )
        grads_norm_slices = []
        for i in grad_norm_indices:
            begin, end = self._parameter_range_in_shard(i)
            if len(grads_norm_slices) < 0 and begin == grads_norm_slices[-1].stop:
                grads_norm_slices[-1] = slice(grads_norm_slices[-1].start, end)
            else:
                grads_norm_slices.append(slice(begin, end))
        grads_for_norm = [self._grad_shard[slice_] for slice_ in grads_norm_slices]

        return param_groups, grads_for_norm

    def check_tensor_parallel_synchronization(self):
        # TODO: Option to check the optimizer state.
        for name, shard in zip(("grad", "weight"), (self.grad_shard, self.weight_shard)):
            for meta, shard_slice in zip(self._parameter_metas, self._split_shard(shard)):
                if shard_slice.numel() > 0 and not meta.is_tensor_parallel:
                    check_parallel_match(shard_slice, self._distributed.tensor_group, f"{name} {meta.tensor_name}")

    def _get_parameter_shard_indices_in_full_weight(self, name: str, device: torch.device):
        """
        Create an index array for the global parameter, where each entry corresponds to the index
        where it is located in the shard if it exists, or -1 if it's not in the shard.
        Used to determine the location of each entry in a different distributed configuration.
        """
        Assert.incl(name, self._parameter_index)
        parameter_index = self._parameter_index[name]
        parameter_meta = self._parameter_metas[parameter_index]

        # Create an empty index for the global parameter.
        index = torch.full(
            parameter_meta.global_shape,
            -1,
            dtype=torch.int64,
            device=device,
        )
        # Set the shard slice of the global parameter to corresponding indices of the parameter slice of the shard
        begin, end = self._parameter_range_in_shard(parameter_index)
        self._parameter_global_to_shard(index, parameter_index).copy_(
            torch.arange(begin, end, dtype=torch.int64, device=device)
        )
        return index

    def _copy_shard_overlaps(
        self,
        loaded_stage: "StageBase",
        shards: list[torch.Tensor],
        loaded_shards: list[torch.Tensor],
        counter: torch.Tensor,
    ):
        """
        See MultiStage._load_partial.
        """
        index_overlap = [name for name in loaded_stage._parameter_index if name in self._parameter_index]
        for name in index_overlap:
            self_index = self._parameter_index[name]
            overlap_index_map = self._parameter_global_to_shard(
                loaded_stage._get_parameter_shard_indices_in_full_weight(name, self._distributed.device), self_index
            )
            overlap_mask = overlap_index_map >= 0
            overlap_index_map_masked = overlap_index_map[overlap_mask]
            overlap_count = overlap_mask.sum()
            begin, end = self._parameter_range_in_shard(self_index)

            for shard, loaded_shard in zip(shards, loaded_shards):
                shard[begin:end][overlap_mask] = loaded_shard[overlap_index_map_masked]
                counter += overlap_count

    def import_state_tensor(self, parameter_name: str, shard: torch.Tensor, tensor: torch.Tensor | SafeTensorSlice):
        """
        Given a global parameter tensor, set the associated slice of a local parameter shard.
        Return the size of the local slice.
        """
        Assert.eq(shard.shape, (self._shard_size,))
        parameter_index = self._parameter_index[parameter_name]
        tensor_shard = self._parameter_global_to_shard(tensor, parameter_index)
        begin, end = self._parameter_range_in_shard(parameter_index)
        Assert.eq(tensor_shard.numel(), end - begin)
        shard[begin:end].copy_(tensor_shard)
        return end - begin

    def _export_shard(self, shard: torch.Tensor, data_type: DataType | None = None):
        if data_type is not None:
            shard = shard.to(dtype=data_type.torch)
        tensors = self._split_buffer(self._reconstruct_from_shard(shard))
        for name, param_index in self._parameter_index.items():
            yield name, self._parameter_metas[param_index].local_to_global(
                tensors[param_index], distributed=self._distributed
            )[0]

    def _parameter_range_in_shard(self, param_index: int):
        begin = self._index_buffer_to_shard(self._parameter_begins_in_buffer[param_index])
        end = self._index_buffer_to_shard(self._parameter_ends_in_buffer[param_index])
        return begin, end

    def _parameter_global_to_shard(self, global_param: torch.Tensor | SafeTensorSlice, param_index: int):
        shard_param = self._parameter_metas[param_index].global_to_local(global_param).flatten()
        if self._fsdp_size > 1:
            shard_param = shard_param[
                self._index_buffer_to_param(
                    self._fsdp_rank * self._shard_size, param_index
                ) : self._index_buffer_to_param((self._fsdp_rank + 1) * self._shard_size, param_index)
            ]
        return shard_param

    def _get_parameter_metas(self):
        # Get all the stage parameters,
        # then separate the parameters with and without weight decay,
        # and squeeze the non-tensor parallel and sequence parallel ones in the middle.
        # This allows running the optimizer, grad norm and sequence_parallel reduction on contiguous buffers.
        parameter_metas: list[ParameterMeta] = []
        for layer in self._layers:
            for name, meta in layer.named_parameters():
                Assert.custom(isinstance, meta, ParameterMeta)
                Assert.eq(meta.dtype, self._distributed_config.optimization_dtype.torch)
                parameter_metas.append(meta)

        reorder_index = sorted(
            range(len(parameter_metas)),
            key=lambda i: (
                parameter_metas[i].param_weight_decay,
                parameter_metas[i].param_weight_decay == parameter_metas[i].is_tensor_parallel,
                parameter_metas[i].param_weight_decay != parameter_metas[i].sequence_tensor_parallel,
            ),
        )
        reordered_metas = [parameter_metas[i] for i in reorder_index]

        return reordered_metas

    def _index_buffer_to_shard(self, index: int, rank: int | None = None):
        shard_begin = (self._fsdp_rank if rank is None else rank) * self._shard_size
        return clamp(index - shard_begin, 0, self._shard_size - self._shard_pad)

    def _index_buffer_to_param(self, index: int, param_index: int):
        return clamp(
            index - self._parameter_begins_in_buffer[param_index], 0, self._parameter_metas[param_index].numel()
        )

    def _reconstruct_from_shard(self, local_shard, out=None):
        return gather_op(local_shard, group=self._fsdp_group, dim=0, out=out)

    def _split_buffer(self, buffer):
        # Split a buffer into appropriately shaped parameters.
        return [
            buffer[begin:end].view(meta.shape)
            for begin, end, meta in zip(
                self._parameter_begins_in_buffer,
                self._parameter_ends_in_buffer,
                self._parameter_metas,
            )
        ]

    def _split_shard(self, shard):
        # Split a shard into flat (possibly empty) parameter slices.
        return [
            shard[self._index_buffer_to_shard(begin) : self._index_buffer_to_shard(end)]
            for begin, end, meta in zip(
                self._parameter_begins_in_buffer,
                self._parameter_ends_in_buffer,
                self._parameter_metas,
            )
        ]
