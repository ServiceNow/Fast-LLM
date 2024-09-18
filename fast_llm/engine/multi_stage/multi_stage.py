import dataclasses
import logging
import warnings

import numpy as np
import torch
from torch._C._distributed_c10d import ProcessGroup

from fast_llm.distributed import Distributed, DistributedConfig, DistributedDim, DistributedDimNames
from fast_llm.engine.base_model.base_model import BaseModel
from fast_llm.engine.multi_stage.config import MultiStageConfig, StageMode
from fast_llm.engine.multi_stage.stage import Stage
from fast_llm.engine.optimizer.config import ParamGroup
from fast_llm.run import log_main_rank, log_model_parallel_main_rank
from fast_llm.tensor import ParameterMeta, TensorDim, TensorMeta
from fast_llm.utils import Assert, get_unique

logger = logging.getLogger(__name__)


class MultiStageModel:
    _is_setup: bool = False
    _state_shard: torch.Tensor
    _weight_shard: torch.Tensor
    _grad_shard: torch.Tensor
    _optimizer_shard: torch.Tensor
    _distributed: Distributed
    _mode: StageMode

    def __init__(
        self,
        *,
        base_model: BaseModel,
        multi_stage_config: MultiStageConfig,
        distributed_config: DistributedConfig,
        optimizer_state_names: tuple[str, ...],
        verbose: bool = True,
        # A filter to create only a subset of the stages. Used for model conversion.
        stage_filter: set | None = None,
    ):
        super().__init__()
        self._multi_stage_config = multi_stage_config
        self._distributed_config = distributed_config
        self._base_model = base_model
        self._training = None
        self._verbose = verbose
        self._stage_filter = stage_filter

        stage_splits = self._split_into_stages()
        self._num_stages = len(stage_splits) - 1

        if self._verbose:
            log_main_rank(lambda: f"  Splitting the model into {self._num_stages} stages...")
        Assert.geq(
            self._num_stages,
            self._distributed_config.pipeline_parallel * self._multi_stage_config.stages_per_pipeline_stage,
        )

        # Create the stages.
        self._stages = [
            Stage(
                base_model=self._base_model,
                config=self._multi_stage_config,
                distributed_config=self._distributed_config,
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
            stage_index: (stage_index // self._multi_stage_config.stages_per_pipeline_stage)
            % self._distributed_config.pipeline_parallel
            for stage_index in (range(self._num_stages))
        }
        self._stages_owned = {
            stage_index: self._stages[stage_index]
            for stage_index, stage_rank in self._stage_pipeline_ranks.items()
            if stage_rank == self._distributed_config.pipeline_rank
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
        self._stage_shard_sizes = [stage.weight_shard_meta.numel() for stage in self._stages_on_device.values()]
        stage_shard_dtype = get_unique([stage.weight_shard_meta.dtype for stage in self._stages_on_device.values()])

        self._state_shard_names = ("weights",) + optimizer_state_names
        self._num_shards = len(self._state_shard_names) + 1

        shard_dim = TensorDim("flat_shard", sum(self._stage_shard_sizes))
        self._weight_shard_meta = TensorMeta.from_dims(
            (shard_dim,),
            tensor_name=f"multi_stage_weight_shard",
            dtype=stage_shard_dtype,
        )
        self._state_shard_meta = TensorMeta.from_dims(
            (TensorDim("state_shards", self._num_shards - 1), shard_dim),
            tensor_name=f"multi_stage_state_shard",
            dtype=stage_shard_dtype,
        )
        self._full_shards_meta = TensorMeta.from_dims(
            (TensorDim("shards", self._num_shards), shard_dim),
            tensor_name=f"multi_stage_state_shard",
            dtype=stage_shard_dtype,
        )

        # contents: buffer_index -> set[stage_index]
        # indices: stage_index -> buffer_index

        # Pre-compute buffer specs.
        # TODO: Reduce code duplication.
        self._weight_buffer_contents, self._weight_buffer_indices = self._get_buffer_placement(
            self._multi_stage_config.num_weight_buffers
        )
        if self._verbose:
            log_model_parallel_main_rank(f"Weight buffer placement:\n{self._weight_buffer_indices}")
        self._weight_buffer_sizes = [
            max(self._stages[stage_index].weight_buffer_meta.numel() for stage_index in contents)
            for contents in self._weight_buffer_contents
        ]
        weight_buffer_dtype = get_unique([stage.weight_buffer_meta.dtype for stage in self._stages])
        self._weight_buffer_meta = TensorMeta.from_dims(
            (TensorDim("weight_buffer", sum(self._weight_buffer_sizes)),),
            tensor_name=f"multi_stage_weight_buffer",
            dtype=weight_buffer_dtype,
        )

        self._grad_buffer_contents, self._grad_buffer_indices = self._get_buffer_placement(
            self._multi_stage_config.num_grad_buffers
        )
        if self._verbose:
            log_model_parallel_main_rank(f"Grad buffer placement:\n{self._grad_buffer_indices}")
        self._grad_buffer_sizes = [
            max(self._stages[stage_index].grad_buffer_meta.numel() for stage_index in contents)
            for contents in self._grad_buffer_contents
        ]
        grad_buffer_dtype = get_unique([stage.grad_buffer_meta.dtype for stage in self._stages])
        self._grad_buffer_meta = TensorMeta.from_dims(
            (TensorDim("grad_buffer", sum(self._grad_buffer_sizes)),),
            tensor_name=f"multi_stage_grad_buffer",
            dtype=grad_buffer_dtype,
        )

        if self._grad_buffer_meta.dtype == torch.bfloat16:
            warnings.warn(
                "Bfloat16 gradient accumulation and reduction is not recommended. (use --full_precision_gradients=1)"
            )

    def setup(self, distributed: Distributed, mode: StageMode = StageMode.training):
        # TODO: More checks?
        stage: Stage
        assert distributed.config is self._distributed_config
        assert not self._is_setup
        self._is_setup = True
        self._distributed = distributed
        self._mode = mode

        self._base_model.setup(distributed)

        allocated = 0

        # Allocate and split shards and buffers.
        if self._mode.support_forward:
            allocated += (mem := self._weight_buffer_meta.memory_usage)
            if self._verbose:
                log_model_parallel_main_rank(
                    f">>> Allocating {len(self._weight_buffer_sizes)} weight buffers ({mem / 2 ** 20:,.2f} MiB)"
                )
            weight_buffers = torch.empty_like(self._weight_buffer_meta, device=self._distributed.device).split(
                self._weight_buffer_sizes
            )
        if self._mode.support_backward:
            allocated += (mem := self._grad_buffer_meta.memory_usage)
            if self._verbose:
                log_model_parallel_main_rank(
                    f">>> Allocating {len(self._grad_buffer_sizes)} grad buffers ({mem / 2 ** 20:,.2f} MiB)"
                )
            grad_buffers = torch.empty_like(self._grad_buffer_meta, device=self._distributed.device).split(
                self._grad_buffer_sizes
            )

        if self._mode.on_device:
            num_shards = (
                self._full_shards_meta.size(0)
                if self._mode.support_training
                else 2 if self._mode.support_backward else 1
            )
            allocated += (mem := num_shards * self._full_shards_meta.memory_usage // self._full_shards_meta.size(0))
            if self._verbose:
                log_model_parallel_main_rank(
                    f">>> Allocating {self._num_shards} x {len(self._stage_shard_sizes)}"
                    f" shards ({mem / 2 ** 20:,.2f} MiB)"
                )
            shards = torch.empty_like(self._full_shards_meta[:num_shards], device=self._distributed.device)
            if self._verbose:
                log_model_parallel_main_rank(f"Total allocated: {allocated / 2 ** 20:,.2f} MiB")
            self._weight_shard = shards[0]
            weight_shard_split = self._weight_shard.split(self._stage_shard_sizes)
            if self._mode.support_backward:
                self._state_shard = shards[:-1]
                if self._mode.support_training:
                    self._optimizer_shard = shards[1:-1]
                self._grad_shard = shards[-1]
                grad_shard_split = self._grad_shard.split(self._stage_shard_sizes)
            else:
                self._state_shard = shards

        # Setup the tied parameter process groups
        for tied_parameter in self._tied_parameters.values():
            tied_parameter.setup(self._distributed)

        # Setup the layer shards and buffers.
        for stage_index, stage in enumerate(self._stages):
            shard_index = self._stage_shard_indices.get(stage_index)
            weight_buffer_index = self._weight_buffer_indices.get(stage_index)
            grad_buffer_index = self._grad_buffer_indices.get(stage_index)
            weight_buffer = (
                weight_buffers[weight_buffer_index][: stage.weight_buffer_meta.numel()]  # noqa
                if self._mode.support_forward and weight_buffer_index is not None
                else None
            )
            grad_buffer = (
                grad_buffers[grad_buffer_index][: stage.grad_buffer_meta.numel()]  # noqa
                if self._mode.support_backward and grad_buffer_index is not None
                else None
            )
            weight_shard = (
                weight_shard_split[shard_index] if self._mode.on_device and shard_index is not None else None  # noqa
            )
            grad_shard = (
                grad_shard_split[shard_index]  # noqa
                if self._mode.support_backward and shard_index is not None
                else None
            )
            weight_buffer_shared_with = (
                [self._stages[i] for i in self._weight_buffer_contents[weight_buffer_index]]
                if self._mode.support_forward and weight_buffer_index is not None
                else []
            )
            stage.setup(
                distributed=distributed,
                weight_shard=weight_shard,
                grad_shard=grad_shard,
                weight_buffer=weight_buffer,
                grad_buffer=grad_buffer,
                mode=self._mode if stage_index in self._stages_on_device else StageMode.off_device,
                is_tied_weight_copy=stage_index in self._stages_on_device and stage_index not in self._stages_owned,
                weight_buffer_shared_with=weight_buffer_shared_with,
            )

        self.train(self._mode.support_backward)

    def get_param_groups(self, param_group_cls: type[ParamGroup] = ParamGroup):
        assert self._is_setup
        assert self._mode.support_training
        # Setup the optimizer param groups.
        optimizer_shards_split = [shard.split(self._stage_shard_sizes) for shard in self._optimizer_shard.unbind()]
        param_groups, grads_for_norm = [], []
        for stage_index, stage in self._stages_on_device.items():
            stage_optimizer_shards = {
                name: shard_split[self._stage_shard_indices[stage_index]]
                for name, shard_split in zip(self._state_shard_names[1:], optimizer_shards_split)
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

    @property
    def support_forward(self):
        assert self._is_setup
        return self._mode.support_forward and self._stage_filter is None

    @property
    def support_backward(self):
        assert self._is_setup
        return self._mode.support_backward and self._stage_filter is None

    @property
    def support_training(self):
        assert self._is_setup
        return self._mode.support_training and self._stage_filter is None

    @property
    def base_model(self):
        return self._base_model

    @property
    def stages(self):
        return self._stages

    @property
    def multi_stage_config(self):
        return self._multi_stage_config

    @property
    def tied_parameters(self):
        return self._tied_parameters

    @property
    def weight_buffer_indices(self):
        return self._weight_buffer_indices

    @property
    def grad_buffer_indices(self):
        return self._grad_buffer_indices

    def invalidate_buffers(self):
        for stage in self._stages_on_device.values():
            stage.invalidate_buffer()

    def train(self, mode: bool = True):
        if self._training != mode:
            for stage in self._stages_on_device.values():
                stage.train(mode)
            self._training = mode

    def _split_into_stages(self):
        # Create stages (greedy split, could do better).
        stage_splits = [0]
        layer_counter, last_counter = 0, 0
        for i, layer in enumerate(self._base_model):
            layer_counter += layer.layer_count  # noqa
            if (
                layer_counter >= last_counter + self._multi_stage_config.layers_per_stage
                or i == len(self._base_model) - 1
            ):
                stage_splits.append(i + 1)
                last_counter = layer_counter
        return stage_splits

    def _get_buffer_placement(self, num_shared_buffers: int | None):
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

    def _get_tied_parameters(self, stage_ends):
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
                on_device=self._distributed_config.pipeline_rank in all_ranks,
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

    def setup(self, distributed: Distributed):
        assert not hasattr(self, "group")
        # Setup the tied parameter process groups
        if len(self.all_ranks) > 1 and self.on_device:
            # TODO: Create a group def first?
            self.group = distributed.add_group(
                DistributedDim(
                    name=self.name + "_tied_weight",
                    size=len(self.all_ranks),
                    rank=sorted(self.all_ranks).index(distributed.config.pipeline_rank),
                    id_=f"{distributed.config.data_rank}_x_{distributed.config.tensor_rank}",
                    parent=DistributedDimNames.pipeline,
                )
            )
        else:
            self.group = None
