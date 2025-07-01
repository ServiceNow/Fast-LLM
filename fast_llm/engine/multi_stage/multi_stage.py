import abc
import dataclasses
import logging
import typing
import warnings

import numpy as np
import torch
from torch._C._distributed_c10d import ProcessGroup

from fast_llm.config import Configurable
from fast_llm.engine.base_model.base_model import BaseModel
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.run import log_main_rank, log_model_parallel_main_rank
from fast_llm.engine.config_utils.tensor_space import TensorDim
from fast_llm.engine.distributed.config import DistributedDim, DistributedDimNames, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, ShardName, StageMode
from fast_llm.engine.multi_stage.fsdp import FSDP
from fast_llm.engine.multi_stage.stage import Stage
from fast_llm.engine.optimizer.config import ParamGroup
from fast_llm.tensor import ParameterMeta, SafeTensorSlice, TensorMeta
from fast_llm.utils import Assert, get_unique

logger = logging.getLogger(__name__)


class MultiStageModel[ConfigType: FastLLMModelConfig](Configurable[ConfigType]):
    config_class: typing.ClassVar[type[FastLLMModelConfig]] = FastLLMModelConfig
    base_model_class: typing.ClassVar[type[BaseModel]] = BaseModel
    _is_setup: bool = False
    _flat_shard: torch.Tensor
    _shards: dict[str, torch.Tensor]
    _shard_names: tuple[str, ...]
    _weight_buffers: tuple[torch.Tensor, ...]
    _grad_buffers: tuple[torch.Tensor, ...]
    _distributed: Distributed
    _mode: StageMode

    def __init__(
        self,
        config: FastLLMModelConfig,
        *,
        # TODO: No longer needed in __init__, move to setup?
        optimizer_state_names: tuple[str, ...] = (),
        verbose: bool = True,
        # A filter to create only a subset of the stages. Used for model conversion.
        stage_filter: set | None = None,
    ):
        super().__init__(config)
        self._base_model = self.base_model_class(self._config.base_model, self._config.distributed)
        self._training = None
        self._verbose = verbose
        self._stage_filter = stage_filter
        self._optimizer_state_names = optimizer_state_names
        self._all_shard_names = (ShardName.weights, *optimizer_state_names, ShardName.grads)

        stage_splits = self._split_into_stages()
        self._num_stages = len(stage_splits) - 1

        if self._verbose:
            log_main_rank(lambda: f"  Splitting the model into {self._num_stages} stages...")
        Assert.geq(
            self._num_stages,
            self._config.distributed.pipeline_parallel * self._config.multi_stage.stages_per_pipeline_stage,
        )

        # Create the stages.
        self._stages = [
            Stage(
                config=self._config.multi_stage,
                base_model=self._base_model,
                distributed_config=self._config.distributed,
                begin=stage_splits[i],
                end=stage_splits[i + 1],
                index=i,
            )
            for i in (range(self._num_stages))
        ]

        if self._verbose:
            log_main_rank(lambda: f"  Total parameters: {sum(stage_.parameter_count for stage_ in self._stages):,} ")

        # Keep track of which stage each parameter belongs to.
        self._parameter_stages: dict[str, int] = {}
        for stage_index, stage in enumerate(self._stages):
            for parameter_name in stage.parameter_names:
                assert parameter_name not in self._parameter_stages
                self._parameter_stages[parameter_name] = stage_index

        # Determine which stages belong to this pipeline rank.
        self._stage_pipeline_ranks = {
            stage_index: (stage_index // self._config.multi_stage.stages_per_pipeline_stage)
            % self._config.distributed.pipeline_parallel
            for stage_index in (range(self._num_stages))
        }
        self._stages_owned = {
            stage_index: self._stages[stage_index]
            for stage_index, stage_rank in self._stage_pipeline_ranks.items()
            if stage_rank == self._config.distributed.pipeline_rank
        }

        # Set up tied weights.
        self._tied_parameters = self._get_tied_parameters(stage_splits[1:])
        self._tied_weight_main_stages_on_device = {
            stage_index: self._stages[stage_index]
            for stage_index in sorted(
                {
                    tied_parameter.main_stage
                    for tied_parameter in self._tied_parameters.values()
                    if tied_parameter.on_device
                }
            )
        }

        # Keep track of which stages are on this device (owned or tied weight copy)
        # Apply the stage filter if applicable.
        self._stages_on_device = {
            stage_index: self._stages[stage_index]
            for stage_index in sorted(set(self._stages_owned) | set(self._tied_weight_main_stages_on_device))
            if self._stage_filter is None or stage_index in self._stage_filter
        }

        # Pre-compute shard specs.
        self._stage_shard_indices = {
            stage_index: shard_index for shard_index, stage_index in enumerate(self._stages_on_device)
        }

        self._fsdp_weight_shard_sizes = [
            [fsdp.weight_shard_meta.numel() for fsdp in stage.fsdps] for stage in self._stages_on_device.values()
        ]
        self._stage_weight_shard_sizes = [sum(shard_sizes) for shard_sizes in self._fsdp_weight_shard_sizes]
        self._weight_shard_size = sum(self._stage_weight_shard_sizes)

        self._fsdp_grad_shard_sizes = [
            [fsdp.grad_shard_meta.numel() for fsdp in stage.fsdps] for stage in self._stages_on_device.values()
        ]
        self._stage_grad_shard_sizes = [sum(shard_sizes) for shard_sizes in self._fsdp_grad_shard_sizes]
        self._grad_shard_size = sum(self._stage_grad_shard_sizes)

        # TODO: Support non-unique data type.
        self._shard_dtype = get_unique(
            [fsdp.weight_shard_meta.dtype for stage in self._stages_on_device.values() for fsdp in stage.fsdps]
        )
        # self._shard_names = (ShardName.weights,) + self._optimizer_state_names + (ShardName.grads,)
        self._weight_shard_meta = TensorMeta.from_dims(
            (TensorDim("weight_shard", self._weight_shard_size),),
            tensor_name=f"multi_stage_weight_shard",
            dtype=self._shard_dtype,
        )
        self._grad_shard_meta = TensorMeta.from_dims(
            (TensorDim("grad_shard", self._grad_shard_size),),
            tensor_name=f"multi_stage_grad_shard",
            dtype=self._shard_dtype,
        )

        # Pre-compute buffer specs.
        # TODO: Reduce code duplication.
        self._weight_buffer_contents, self._weight_buffer_indices = self._get_buffer_placement(
            self._config.multi_stage.num_weight_buffers
        )
        if self._verbose:
            log_model_parallel_main_rank(f"Weight buffer placement:\n{self._weight_buffer_indices}")
        # TODO: Let stages worry about their buffer splitting?
        self._fsdp_weight_buffer_sizes = [
            [fsdp.weight_buffer_meta.numel() for fsdp in stage.fsdps] for stage in self._stages
        ]
        self._stage_weight_buffer_sizes = [sum(buffer_sizes) for buffer_sizes in self._fsdp_weight_buffer_sizes]
        self._weight_buffer_sizes = [
            max(self._stage_weight_buffer_sizes[stage_index] for stage_index in contents)
            for contents in self._weight_buffer_contents
        ]
        weight_buffer_dtype = get_unique(
            [fsdp.weight_buffer_meta.dtype for stage in self._stages for fsdp in stage.fsdps]
        )
        self._weight_buffer_meta = TensorMeta.from_dims(
            (TensorDim("weight_buffer", sum(self._weight_buffer_sizes)),),
            tensor_name=f"multi_stage_weight_buffer",
            dtype=weight_buffer_dtype,
        )

        self._grad_buffer_contents, self._grad_buffer_indices = self._get_buffer_placement(
            self._config.multi_stage.num_grad_buffers
        )
        if self._verbose:
            log_model_parallel_main_rank(f"Grad buffer placement:\n{self._grad_buffer_indices}")
        self._fsdp_grad_buffer_sizes = [
            [fsdp.grad_buffer_meta.numel() for fsdp in stage.fsdps] for stage in self._stages
        ]
        self._stage_grad_buffer_sizes = [sum(buffer_sizes) for buffer_sizes in self._fsdp_grad_buffer_sizes]
        self._grad_buffer_sizes = [
            max(self._stage_grad_buffer_sizes[stage_index] for stage_index in contents)
            for contents in self._grad_buffer_contents
        ]

        grad_buffer_dtype = get_unique([fsdp.grad_buffer_meta.dtype for stage in self._stages for fsdp in stage.fsdps])
        self._grad_buffer_meta = TensorMeta.from_dims(
            (TensorDim("grad_buffer", sum(self._grad_buffer_sizes)),),
            tensor_name=f"multi_stage_grad_buffer",
            dtype=grad_buffer_dtype,
        )

        if self._grad_buffer_meta.dtype == torch.bfloat16:
            warnings.warn(
                "Bfloat16 gradient accumulation and reduction is not recommended. (use --full_precision_gradients=1)"
            )

    def setup(self, distributed: Distributed | None = None, mode: StageMode = StageMode.training) -> None:
        # TODO: More checks?
        stage: Stage
        assert not self._is_setup
        self._is_setup = True
        if distributed is None:
            distributed = Distributed(self._config.distributed)
        else:
            distributed.check_config(self._config.distributed)
        self._distributed = distributed
        self._mode = mode
        self._base_model.setup(distributed)

        allocated = 0

        # Allocate and split shards and buffers.
        if self._mode.support_forward:
            self._weight_buffers, mem = self._allocate_buffers(
                self._weight_buffer_meta, self._weight_buffer_sizes, "weight"
            )
            allocated += mem
        if self._mode.support_backward:
            self._grad_buffers, mem = self._allocate_buffers(self._grad_buffer_meta, self._grad_buffer_sizes, "grad")
            allocated += mem

        self._shard_names = ()
        if self._mode.on_device:
            self._shard_names += (ShardName.weights,)
            if self._mode.support_training:
                self._shard_names += self._optimizer_state_names
            if self._mode.support_backward:
                self._shard_names += (ShardName.grads,)

        if self._mode.on_device:
            allocated += self._allocate_shards()

        if self._verbose:
            log_model_parallel_main_rank(f"Total allocated: {allocated / 2 ** 20:,.2f} MiB")

        # Setup the tied parameter process groups
        for tied_parameter in self._tied_parameters.values():
            tied_parameter.setup(self._distributed)

        # Setup the layer shards and buffers.
        self._setup_stages()

        self.train(self._mode.support_backward)

    @abc.abstractmethod
    def get_tflops(self, phase: PhaseType, elapsed_time_per_iteration, batch_size, sequence_length) -> tuple[int, int]:
        # TODO: Do in model, automate/generalize, get other stats
        pass

    def _allocate_buffers(
        self, buffer_meta: TensorMeta, sizes: list[int], name: str
    ) -> tuple[tuple[torch.Tensor, ...], int]:
        mem = buffer_meta.memory_usage
        if self._verbose:
            log_model_parallel_main_rank(f">>> Allocating {len(sizes)} {name} buffers ({mem / 2 ** 20:,.2f} MiB)")
        return torch.empty_like(buffer_meta, device=self._distributed.device).split(sizes), mem

    def _allocate_shards(self) -> int:
        shard_sizes = [
            self._weight_shard_size if shard_name == ShardName.weights else self._grad_shard_size
            for shard_name in self._shard_names
        ]
        full_shards_meta = TensorMeta.from_dims(
            (TensorDim("", sum(shard_sizes)),),
            tensor_name=f"",
            dtype=self._shard_dtype,
        )
        mem = full_shards_meta.memory_usage
        if self._verbose:
            log_model_parallel_main_rank(f">>> Allocating {len(self._shard_names)} shards ({mem / 2 ** 20:,.2f} MiB)")
        self._flat_shard = torch.empty_like(full_shards_meta, device=self._distributed.device)

        self._shards = {
            shard_name: shard
            for shard_name, shard in zip(self._shard_names, self._flat_shard.split(shard_sizes), strict=True)
        }
        return mem

    def _setup_stages(self) -> None:
        for stage_index, stage in enumerate(self._stages):
            shard_index = self._stage_shard_indices.get(stage_index)
            weight_buffer_index = self._weight_buffer_indices.get(stage_index)
            grad_buffer_index = self._grad_buffer_indices.get(stage_index)
            stage_weight_buffers = (
                self._weight_buffers[weight_buffer_index][
                    : self._stage_weight_buffer_sizes[stage_index]
                ].split(  # noqa
                    self._fsdp_weight_buffer_sizes[stage_index]
                )
                if self._mode.support_forward and weight_buffer_index is not None
                else None
            )
            stage_grad_buffers = (
                self._grad_buffers[grad_buffer_index][: self._stage_grad_buffer_sizes[stage_index]].split(  # noqa
                    self._fsdp_grad_buffer_sizes[stage_index]
                )
                if self._mode.support_backward and grad_buffer_index is not None
                else None
            )
            stage_weight_shards = (
                self._shards[ShardName.weights]
                .split(self._stage_weight_shard_sizes)[shard_index]
                .split(self._fsdp_weight_shard_sizes[shard_index])
                if self._mode.on_device and shard_index is not None
                else None  # noqa
            )
            stage_grad_shards = (
                self._shards[ShardName.grads]
                .split(self._stage_grad_shard_sizes)[shard_index]
                .split(self._fsdp_grad_shard_sizes[shard_index])
                if self._mode.support_backward and shard_index is not None
                else None
            )
            weight_buffer_shared_with = (
                [self._stages[i] for i in self._weight_buffer_contents[weight_buffer_index]]
                if self._mode.support_forward and weight_buffer_index is not None
                else []
            )
            stage.setup(
                distributed=self._distributed,
                weight_shards=stage_weight_shards,
                grad_shards=stage_grad_shards,
                weight_buffers=stage_weight_buffers,
                grad_buffers=stage_grad_buffers,
                mode=self._mode if stage_index in self._stages_on_device else StageMode.off_device,
                is_tied_weight_copy=stage_index in self._stages_on_device and stage_index not in self._stages_owned,
                weight_buffer_shared_with=weight_buffer_shared_with,
            )

    def get_param_groups(
        self, param_group_cls: type[ParamGroup] = ParamGroup
    ) -> tuple[list[ParamGroup], list[torch.Tensor]]:
        assert self._is_setup
        assert self._mode.support_training
        # Setup the optimizer param groups.
        optimizer_shards_split = {
            shard_name: self._shards[shard_name].split(
                self._stage_weight_shard_sizes if shard_name == ShardName.weights else self._stage_grad_shard_sizes
            )
            for shard_name in self._optimizer_state_names
        }
        param_groups, grads_for_norm = [], []
        for stage_index, stage in self._stages_on_device.items():
            shard_index = self._stage_shard_indices.get(stage_index)
            stage_optimizer_shards = {
                shard_name: shard_split[shard_index].split(
                    (
                        self._fsdp_weight_shard_sizes
                        if shard_name == ShardName.weights
                        else self._fsdp_grad_shard_sizes
                    )[shard_index]
                )
                for shard_name, shard_split in optimizer_shards_split.items()
            }
            stage_param_groups, stage_grads_for_norm = stage.get_param_groups(
                stage_optimizer_shards,
                param_group_cls=param_group_cls,
            )
            param_groups.extend(stage_param_groups)
            # Exclude contribution from tied weight copies.
            if stage_index in self._stages_owned:
                grads_for_norm.extend(stage_grads_for_norm)

        return param_groups, grads_for_norm

    def get_shard_meta(self, name: str) -> TensorMeta:
        if name not in self._all_shard_names:
            raise KeyError(f"Unknown shard name {name}")
        return self._weight_shard_meta if name == ShardName.weights else self._grad_shard_meta

    def get_shard(self, name: str) -> torch.Tensor:
        assert self._is_setup
        if name not in self._shard_names:
            raise KeyError(f"Unknown shard name {name}")
        return self._shards[name]

    @property
    def is_setup(self) -> bool:
        return self._is_setup

    @property
    def support_forward(self) -> bool:
        assert self._is_setup
        return self._mode.support_forward and self._stage_filter is None

    @property
    def support_backward(self) -> bool:
        assert self._is_setup
        return self._mode.support_backward and self._stage_filter is None

    @property
    def support_training(self) -> bool:
        assert self._is_setup
        return self._mode.support_training and self._stage_filter is None

    @property
    def base_model(self) -> BaseModel:
        return self._base_model

    @property
    def stages(self) -> list[Stage]:
        return self._stages

    @property
    def stages_on_device(self) -> dict[int, Stage]:
        return self._stages_on_device

    @property
    def tied_parameters(self) -> dict[str, "TiedParameter"]:
        return self._tied_parameters

    @property
    def weight_buffer_indices(self) -> dict[int, int]:
        return self._weight_buffer_indices

    @property
    def grad_buffer_indices(self) -> dict[int, int]:
        return self._grad_buffer_indices

    @property
    def state_shard_names(self) -> tuple[str, ...]:
        return self._shard_names[:-1] if self._mode.support_backward else self._shard_names

    def _get_stage_shard_sizes(self, shard_name: str) -> list[int]:
        return self._stage_weight_shard_sizes if shard_name == ShardName.weights else self._stage_grad_shard_sizes

    def _get_fsdp_shard_sizes(self, shard_name: str) -> list[list[int]]:
        return self._fsdp_weight_shard_sizes if shard_name == ShardName.weights else self._fsdp_grad_shard_sizes

    @property
    def parameter_names(self) -> list[str]:
        return list(self._parameter_stages)

    def get_parameter_stage(self, parameter_name: str) -> Stage:
        return self._stages[self._parameter_stages[parameter_name]]

    def is_parameter_on_device(self, parameter_name: str) -> bool:
        return self._parameter_stages[parameter_name] in self._stages_on_device

    @property
    def distributed(self) -> Distributed:
        assert self._is_setup
        return self._distributed

    @property
    def stages_fsdp_parameters(self) -> typing.Generator[tuple[Stage, FSDP, str, ParameterMeta], None, None]:
        for stage in self._stages:
            for fsdp in stage.fsdps:
                for parameter_name in fsdp.parameter_names:
                    yield stage, fsdp, parameter_name, stage.get_parameter_meta(parameter_name)

    def invalidate_buffers(self) -> None:
        for stage in self._stages_on_device.values():
            stage.invalidate_buffer()

    def train(self, mode: bool = True) -> None:
        if self._training != mode:
            for stage in self._stages_on_device.values():
                stage.train(mode)
            self._training = mode

    def get_state_tensor_iterator(
        self, shard_names: tuple[str, ...], data_type: DataType | None = None
    ) -> typing.Generator[tuple[str, str, torch.Tensor], None, None]:
        for shard_name in shard_names:
            shard_split = self._shards[shard_name].split(self._stage_weight_shard_sizes, 0)
            for shard_index, (stage, shard) in enumerate(
                zip(self._stages_on_device.values(), shard_split, strict=True)
            ):
                for name, tensor in stage._export_shard(
                    shard.split(self._fsdp_weight_shard_sizes[shard_index]), data_type=data_type
                ):  # noqa
                    yield name, shard_name, tensor

    def import_state_tensor(self, parameter_name: str, shard_name: str, tensor: torch.Tensor | SafeTensorSlice):
        """
        Given a global parameter tensor, set the associated slice of a local parameter shard.
        Return the size of the local slice.
        """
        if not self.is_parameter_on_device(parameter_name):
            # Parameter is not on device, nothing to do.
            return 0
        shard_index = self._stage_shard_indices[self._parameter_stages[parameter_name]]
        stage_shards = (
            self._shards[shard_name]
            .split(self._stage_weight_shard_sizes, 0)[shard_index]
            .split(self._fsdp_weight_shard_sizes[shard_index])
        )
        return self.get_parameter_stage(parameter_name).import_state_tensor(parameter_name, stage_shards, tensor)

    def split_shards_by_fsdp(
        self, shards: dict[str, torch.Tensor]
    ) -> typing.Generator[tuple[Stage, FSDP, dict[str, torch.Tensor]], None, None]:
        stage_shards = {
            shard_name: shard.split(self._get_stage_shard_sizes(shard_name)) for shard_name, shard in shards.items()
        }
        for shard_index, stage in enumerate(self.stages_on_device.values()):
            fsdp_shards = {
                shard_name: stage_shards_[shard_index].split(self._get_fsdp_shard_sizes(shard_name)[shard_index])
                for shard_name, stage_shards_ in stage_shards.items()
            }
            for fsdp_index, fsdp in enumerate(stage.fsdps):
                yield stage, fsdp, {
                    shard_name: fsdp_shards_[fsdp_index] for shard_name, fsdp_shards_ in fsdp_shards.items()
                }

    def _split_into_stages(self) -> list[int]:
        # Create stages (greedy split, could do better).
        stage_splits = [0]
        layer_counter, last_counter = 0, 0
        for i, layer in enumerate(self._base_model):
            layer_counter += layer.layer_count  # noqa
            if (
                layer_counter >= last_counter + self._config.multi_stage.layers_per_stage
                or i == len(self._base_model) - 1
            ):
                stage_splits.append(i + 1)
                last_counter = layer_counter
        return stage_splits

    def _get_buffer_placement(self, num_shared_buffers: int | None) -> tuple[list[set[int]], dict[int, int]]:
        num_shared_buffers = num_shared_buffers or self._num_stages
        buffer_contents: list[set[int]] = [set() for _ in range(num_shared_buffers)]
        local_stage_index = 0
        for stage_index, stage in self._stages_on_device.items():
            if stage_index in self._tied_weight_main_stages_on_device:
                buffer_contents.append({stage_index})
            else:
                buffer_contents[local_stage_index % num_shared_buffers].add(stage_index)
                local_stage_index += 1
        buffer_contents = [contents for contents in buffer_contents if contents]
        buffer_indices = {
            stage_index: buffer_index
            for buffer_index, buffer_contents_ in enumerate(buffer_contents)
            for stage_index in buffer_contents_
        }
        return buffer_contents, buffer_indices

    def _get_tied_parameters(self, stage_ends) -> dict[str, "TiedParameter"]:
        tied_parameters = {}
        for name, (meta, layer_indexes) in self._base_model.get_tied_weights().items():
            Assert.eq(list(layer_indexes), sorted(layer_indexes))
            Assert.incl(meta, list(self._base_model[layer_indexes[0]].parameters()))
            stage_indexes = sorted({np.searchsorted(stage_ends, i, side="right").item() for i in layer_indexes})
            all_ranks = {self._stage_pipeline_ranks[stage_index] for stage_index in stage_indexes}

            tied_parameters[name] = TiedParameter(
                name=name,
                meta=meta,
                all_ranks=all_ranks,
                on_device=self._config.distributed.pipeline_rank in all_ranks,
                main_stage=stage_indexes[0],
            )
        return tied_parameters


@dataclasses.dataclass
class TiedParameter:
    name: str
    # Parameter definition.
    meta: ParameterMeta
    # Whether the local rank is involved at all.
    on_device: bool
    # Process group for reduction.
    group: ProcessGroup | None = dataclasses.field(init=False)
    all_ranks: set[int]
    # The index of the main stage.
    main_stage: int

    def setup(self, distributed: Distributed) -> None:
        assert not hasattr(self, "group")
        # Setup the tied parameter process groups
        if len(self.all_ranks) > 1 and self.on_device:
            # TODO: Create a group def first?
            pipeline_ranks = distributed.config.get_distributed_dim(DistributedDimNames.pipeline).global_ranks
            self.group = distributed.add_group(
                DistributedDim(
                    name=self.name + "_tied_weight",
                    size=len(self.all_ranks),
                    rank=sorted(self.all_ranks).index(distributed.config.pipeline_rank),
                    global_ranks=tuple(pipeline_ranks[rank] for rank in self.all_ranks),
                )
            )
        else:
            self.group = None
