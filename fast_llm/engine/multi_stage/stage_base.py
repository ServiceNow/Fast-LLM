import logging
import typing

import torch
import torch._dynamo  # noqa

from fast_llm.config import Configurable
from fast_llm.core.distributed import check_parallel_match
from fast_llm.engine.base_model.base_model import BaseModel, Layer
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import ShardName, StageConfig, StageMode
from fast_llm.engine.multi_stage.fsdp import FSDP
from fast_llm.engine.optimizer.config import ParamGroup
from fast_llm.logging import log_generator
from fast_llm.tensor import ParameterMeta, SafeTensorSlice
from fast_llm.utils import Assert, div

logger = logging.getLogger(__name__)


class StageBase(Configurable[StageConfig]):
    config_class: typing.ClassVar[type[StageConfig]] = StageConfig
    _distributed: Distributed
    _mode: StageMode

    def __init__(
        self,
        *,
        config: StageConfig,
        base_model: BaseModel | list[Layer],
        distributed_config: DistributedConfig,
        begin: int,
        end: int,
        index: int,
    ):
        super().__init__(config)
        self._distributed_config = distributed_config.validate()
        Assert.in_range(begin, 0, end)
        Assert.leq(end, len(base_model))

        self._fsdp_rank = self._distributed_config.data_rank
        self._fsdp_size = self._distributed_config.data_parallel
        self._is_setup = False
        self._index = index

        self._layers = [torch.compile(layer) if self._config.compile_all else layer for layer in base_model[begin:end]]
        self._layer_range = list(range(begin, end))

        parameter_metas, frozen_metas = self._get_parameter_metas()
        self._parameter_metas = parameter_metas + frozen_metas
        self._fsdps = []
        gradient_buffer_dtype = (
            self._distributed_config.optimization_dtype
            if self._config.full_precision_gradients
            else self._distributed_config.training_dtype
        )
        if parameter_metas:
            self._fsdps.append(
                FSDP(
                    f"stage_{self._index}",
                    parameter_metas,
                    self._distributed_config.get_distributed_dim(DistributedDimNames.data),
                    training_dtype=self._distributed_config.training_dtype,
                    gradient_buffer_dtype=gradient_buffer_dtype,
                    optimization_dtype=self._distributed_config.optimization_dtype,
                )
            )
        if frozen_metas:
            self._fsdps.append(
                FSDP(
                    f"stage_{self._index}_frozen",
                    frozen_metas,
                    self._distributed_config.get_distributed_dim(DistributedDimNames.data),
                    training_dtype=self._distributed_config.training_dtype,
                    gradient_buffer_dtype=gradient_buffer_dtype,
                    optimization_dtype=(
                        self._distributed_config.optimization_dtype
                        if self._config.store_frozen_weights_in_optimization_precision
                        else self._distributed_config.training_dtype.torch
                    ),
                )
            )
        # TODO: Separate fsdp for tied weights?
        self._fsdp_index = {name: i for i, fsdp in enumerate(self._fsdps) for name in fsdp.parameter_names}

    @property
    def requires_grad(self):
        return any(fsdp.requires_grad for fsdp in self._fsdps)

    @property
    def mode(self) -> StageMode:
        assert self._is_setup
        return self._mode

    @property
    def index(self) -> int:
        return self._index

    @property
    def fsdps(self) -> list[FSDP]:
        return self._fsdps

    @property
    def parameter_count(self) -> int:
        return sum(fsdp.parameter_count for fsdp in self._fsdps)

    @property
    def parameter_names(self) -> list[str]:
        return sum((fsdp.parameter_names for fsdp in self._fsdps), [])

    def get_parameter_meta(self, parameter_name: str) -> ParameterMeta:
        return self._fsdps[self._fsdp_index[parameter_name]].get_parameter_meta(parameter_name)

    def get_parameter_buffer(self, parameter_name: str) -> torch.nn.Parameter:
        assert self._is_setup
        return self._fsdps[self._fsdp_index[parameter_name]].get_parameter_buffer(parameter_name)

    def setup(
        self,
        *,
        distributed: Distributed,
        weight_shards: list[torch.Tensor | None] | None,
        grad_shards: list[torch.Tensor | None] | None,
        weight_buffers: list[torch.Tensor | None] | None,
        grad_buffers: list[torch.Tensor | None] | None,
        mode: StageMode = StageMode.training,
    ) -> None:
        assert not self._is_setup
        distributed.check_config(self._distributed_config)
        self._mode = mode
        self._is_setup = True
        self._distributed = distributed

        if weight_shards is None:
            weight_shards = [None for _ in self._fsdps]
        if grad_shards is None:
            grad_shards = [None for _ in self._fsdps]
        if weight_buffers is None:
            weight_buffers = [None for _ in self._fsdps]
        if grad_buffers is None:
            grad_buffers = [None for _ in self._fsdps]

        for fsdp, weight_shard, grad_shard, weight_buffer, grad_buffer in zip(
            self._fsdps, weight_shards, grad_shards, weight_buffers, grad_buffers, strict=True
        ):
            fsdp.setup(
                mode=mode,
                fsdp_group=self._distributed.data_group,
                weight_shard=weight_shard,
                grad_shard=grad_shard,
                weight_buffer=weight_buffer,
                grad_buffer=grad_buffer,
                sequence_tensor_parallel=self._distributed_config.sequence_tensor_parallel,
                device=self._distributed.device,
            )

        if self._mode.support_forward:
            # Replace the parameter definitions in each module with the actual parameter buffers.
            def _replace(module: torch.nn.Module):
                nonlocal i
                for key in module._parameters:
                    meta = typing.cast(ParameterMeta, module._parameters[key])
                    module._parameters[key] = self.get_parameter_buffer(meta.tensor_name)
                    i += 1

            i = 0
            for layer in self._layers:
                layer.apply(_replace)

            Assert.eq(i, len(self._parameter_metas))

    def initialize_weights(self) -> None:
        # TODO: Avoid all the _on_device checks
        assert self._is_setup
        with torch.no_grad():
            if self._config.debug_param_init:
                log_generator("CPU generator before reset", torch.random.default_generator)
                log_generator("PP init generator before reset", self._distributed.pp_init_generator)
                log_generator("TP init generator before reset", self._distributed.tp_init_generator)

            # Ensure a reproducible ordering.
            metas = (
                sorted(self._parameter_metas, key=lambda parameter_meta: parameter_meta.tensor_name)
                if self._distributed_config.reproducible_init
                else self._parameter_metas
            )
            weight_shards_split = [
                fsdp.split_shard(fsdp.weight_shard if self._mode.on_device else fsdp.weight_shard_meta)
                for fsdp in self._fsdps
            ]

            for meta in metas:
                fsdp = self._fsdps[fsdp_index := self._fsdp_index[meta.tensor_name]]
                parameter = weight_shards_split[fsdp_index][meta.tensor_name]
                # Multi-gpu init may be different because of TP or FSDP (different shape), or PP (not on device)
                global_shape = meta.global_shape

                if self._distributed_config.reproducible_init and (
                    global_shape.numel() != parameter.numel() or not self._mode.on_device
                ):
                    # Initialize all global weights on every gpu, then select the appropriate slice if applicable.
                    global_param = parameter.new_empty(global_shape, device=self._distributed.device)
                    meta.init_parameter(global_param, distributed=self._distributed)
                    if self._mode.on_device:
                        parameter.copy_(fsdp.parameter_global_to_shard(global_param, meta.tensor_name))
                elif self._mode.on_device:
                    meta.init_parameter(parameter, self._distributed)

            if self.mode.on_device:
                fsdp.reset_shard_pad(fsdp.weight_shard, ShardName.weights)

        if self._config.debug_param_init:
            log_generator("CPU generator after reset", torch.random.default_generator)
            log_generator("PP init generator after reset", self._distributed.pp_init_generator)
            log_generator("TP init generator after reset", self._distributed.tp_init_generator)
            if self._mode.on_device:
                fsdp.log_shard(
                    name="param",
                    shard=fsdp.weight_shard,
                    distributed=self._distributed,
                    level=self._config.debug_param_init,
                    global_=self._config.debug_global_tensors,
                )

    # def reset_shard_pad(self, shard: torch.Tensor) -> int:
    #    assert self._is_setup
    #    assert self._mode.on_device
    #    return sum(fsdp.reset_shard_pad(shard) for fsdp in self._fsdps)

    def get_param_groups(
        self, optimizer_state_shards: dict[str, tuple[torch.Tensor]], param_group_cls: type[ParamGroup]
    ) -> tuple[list[ParamGroup], list[torch.Tensor]]:
        # TODO: Separate model-specific code.
        # TODO: verify optimizer states
        assert self._is_setup
        assert self._mode.support_training
        assert all(len(state_shards) == len(self._fsdps) for state_shards in optimizer_state_shards.values())

        # Get the weight slices and group by optimizer parameters, merging consecutive slices.
        grouped_parameter_slices = {}
        param_groups = []
        for i, fsdp in enumerate(self._fsdps):
            if not fsdp.requires_grad:
                continue
            for parameter_name in fsdp.parameter_names:
                # If needed, chunk the parameter on the first dimension.
                parameter_meta = fsdp.get_parameter_meta(parameter_name)
                if not parameter_meta.requires_grad:
                    continue
                chunk_size = div(parameter_meta.numel(), len(parameter_meta.lr_scale))
                buffer_begin = fsdp.get_parameter_begin_in_buffer(parameter_meta.tensor_name)
                for i, lr_scale in enumerate(parameter_meta.lr_scale):
                    begin = fsdp.index_buffer_to_shard(buffer_begin + i * chunk_size)
                    end = fsdp.index_buffer_to_shard(buffer_begin + (i + 1) * chunk_size)
                    if lr_scale == 0 or begin == end:
                        continue
                    optimizer_params = (parameter_meta.param_weight_decay, lr_scale)
                    if optimizer_params in grouped_parameter_slices:
                        last_slice = grouped_parameter_slices[optimizer_params][-1]
                        if begin == last_slice.stop:
                            grouped_parameter_slices[optimizer_params][-1] = slice(last_slice.start, end)
                            continue
                    else:
                        grouped_parameter_slices[optimizer_params] = []
                    grouped_parameter_slices[optimizer_params].append(slice(begin, end))

            param_groups += [
                param_group_cls(
                    name=f"wd_{weight_decay}_lr_scale_{lr_scale}",  # noqa
                    params=[fsdp.weight_shard[slice_] for slice_ in slices],  # noqa
                    grads=[fsdp.grad_shard[slice_] for slice_ in slices],  # noqa
                    **{  # noqa
                        name: [optimizer_state[i][slice_] for slice_ in slices]
                        for name, optimizer_state in optimizer_state_shards.items()
                    },
                    weight_decay=None if weight_decay else 0.0,  # noqa
                    lr_scale=lr_scale,  # noqa
                )
                for (weight_decay, lr_scale), slices in grouped_parameter_slices.items()
            ]

        # Get the weight slices to use for grad norm computation, merging consecutive slices.
        grads_for_norm = []
        for fsdp in self._fsdps:
            grad_norm_names = (
                fsdp.parameter_names
                if self._distributed_config.tensor_rank == 0
                else [name for name in fsdp.parameter_names if fsdp.get_parameter_meta(name).is_tensor_parallel]
            )
            grads_norm_slices = []
            for name in grad_norm_names:
                begin, end = fsdp._parameter_range_in_shard(name)
                if len(grads_norm_slices) < 0 and begin == grads_norm_slices[-1].stop:
                    grads_norm_slices[-1] = slice(grads_norm_slices[-1].start, end)
                else:
                    grads_norm_slices.append(slice(begin, end))
            grads_for_norm += [fsdp.grad_shard[slice_] for slice_ in grads_norm_slices]

        return param_groups, grads_for_norm

    def check_tensor_parallel_synchronization(self) -> None:
        # TODO: Option to check the optimizer state.
        for fsdp in self._fsdps:
            for shard_name, shard in zip(("grad", "weight"), (fsdp.grad_shard, fsdp.weight_shard)):
                for parameter_name, shard_slice in fsdp.split_shard(shard).items():
                    if shard_slice.numel() > 0 and not fsdp.get_parameter_meta(parameter_name).is_tensor_parallel:
                        check_parallel_match(
                            shard_slice, self._distributed.tensor_group, f"{shard_name} {parameter_name}"
                        )

    def import_state_tensor(
        self, parameter_name: str, shards: tuple[torch.Tensor], tensor: torch.Tensor | SafeTensorSlice
    ) -> int:
        """
        Given a global parameter tensor, set the associated slice of a local parameter shard.
        Return the size of the local slice.
        TODO: Doesn't work
        """
        fsdp_index = self._fsdp_index[parameter_name]
        return self._fsdps[fsdp_index].import_state_tensor(parameter_name, shards[fsdp_index], tensor)

    def _export_shard(
        self, shards: tuple[torch.Tensor], data_type: DataType | None = None
    ) -> typing.Generator[tuple[str, torch.Tensor], None, None]:
        # TODO: Doesn't work
        for fsdp, shard in zip(self._fsdps, shards, strict=True):
            yield from fsdp.export_shard(shard, self._distributed, data_type)

    def _get_parameter_metas(self) -> tuple[list[ParameterMeta], list[ParameterMeta]]:
        # Get all the stage parameters,
        # then separate the parameters with and without weight decay,
        # and squeeze the non-tensor parallel and sequence parallel ones in the middle.
        # This allows running the optimizer, grad norm and sequence_parallel reduction on contiguous buffers.
        parameter_metas: list[ParameterMeta] = []
        frozen_metas: list[ParameterMeta] = []
        meta: ParameterMeta
        for layer in self._layers:
            for name, meta in layer.named_parameters():
                Assert.custom(isinstance, meta, ParameterMeta)
                Assert.eq(meta.dtype, self._distributed_config.optimization_dtype.torch)
                if meta.requires_grad:
                    parameter_metas.append(meta)
                else:
                    frozen_metas.append(meta)

        return self._reorder_parameter_metas(parameter_metas), self._reorder_parameter_metas(frozen_metas)

    @classmethod
    def _reorder_parameter_metas(cls, parameter_metas):
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
