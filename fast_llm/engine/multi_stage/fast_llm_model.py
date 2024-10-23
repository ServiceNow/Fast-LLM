import json
import logging
import math
import pathlib
import typing

import safetensors.torch
import torch
import yaml

from fast_llm.core.distributed import all_reduce, broadcast
from fast_llm.engine.base_model.base_model import BaseModel
from fast_llm.engine.config_utils.checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointFormat,
    CheckpointLoadConfig,
    CheckpointSaveConfig,
    ModelConfigType,
)
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.checkpoint import StateDictSaver
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, StageMode
from fast_llm.engine.multi_stage.conversion import ModelConverter, TrivialConverter
from fast_llm.engine.multi_stage.multi_stage import MultiStageModel
from fast_llm.engine.multi_stage.stage import Stage
from fast_llm.functional.triton.pointwise import triton_fill
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


def _export_safetensors_metadata(metadata):
    """
    Safetensor only accepts string entries, so we convert to string explicitly.
    We use yaml rather than json because json requires explicit quotation marks on strings, which breaks things.
    (ex. "format": "pt" becomes '"pt"' which breaks huggingface models.)
    We avoid using safe_dump for scalars because it adds junk ("\n...\n") at the end of the string
    (decoding is unaffected.)
    """
    return {
        key: str(value) if isinstance(value, (str, int, float, bool)) else yaml.safe_dump(value)
        for key, value in metadata.items()
    }


def _import_safetensors_metadata(metadata):
    return {key: yaml.safe_load(value) for key, value in metadata.items()}


class FastLLMModel(MultiStageModel):
    _is_setup: bool = False
    _is_loaded: bool = False
    _distributed: Distributed
    config_class: typing.ClassVar[type[FastLLMModelConfig]] = FastLLMModelConfig
    base_model_class: typing.ClassVar[type[BaseModel]] = BaseModel

    def __init__(
        self,
        config: FastLLMModelConfig,
        *,
        optimizer_state_names: tuple[str, ...] = (),
        verbose: bool = True,
        # A filter to create only a subset of the stages. Used for model conversion.
        stage_filter: set | None = None,
    ):
        self._base_model_config = config.base_model
        self._fast_llm_config = config
        super().__init__(
            base_model=self.base_model_class(config.base_model, config.distributed),
            multi_stage_config=config.multi_stage,
            distributed_config=config.distributed,
            optimizer_state_names=optimizer_state_names,
            verbose=verbose,
            stage_filter=stage_filter,
        )

    @property
    def fast_llm_config(self):
        return self._fast_llm_config

    @property
    def distributed(self):
        return self._distributed

    def save_checkpoint(
        self,
        checkpoint_config: CheckpointSaveConfig,
        metadata: dict | None = None,
    ):
        # TODO: Handle barriers, ok file, mkdir, etc. here

        num_shards = len(self._state_shard_names) if checkpoint_config.optimizer_state else 1
        metadata = {
            "checkpoint_type": CheckpointFormat.distributed.value,
            "checkpoint_version": str(CHECKPOINT_VERSION),
            "fast_llm_config": self._fast_llm_config.to_serialized(),
            "state_shard_names": list(self._state_shard_names[:num_shards]),
            "metadata": {} if metadata is None else metadata,
        }

        # TODO: Simplify branching.
        if checkpoint_config.format == CheckpointFormat.external:
            # TODO: Support optimizer?
            assert not checkpoint_config.optimizer_state
            converter_class = self._base_model_config.get_converter_class(checkpoint_config.model_type)
            exported_config = converter_class.export_config(self._base_model_config)
            converter_class.save_config(checkpoint_config.path, exported_config)
            self._save_state_dict(
                checkpoint_config,
                converter_class(self._base_model_config),
                {
                    "fast_llm_metadata": metadata,
                    "model_config": exported_config,
                    "format": "pt",
                },
            )
        elif checkpoint_config.format == CheckpointFormat.state_dict:
            self._save_state_dict(checkpoint_config, TrivialConverter(), metadata)
        elif checkpoint_config.format == CheckpointFormat.distributed:
            if self._distributed_config.rank == 0:
                yaml.safe_dump(metadata, (checkpoint_config.path / "metadata.yaml").open("w"))
            safetensors.torch.save_file(
                tensors={"state_shard": self._state_shard[:num_shards]},
                filename=checkpoint_config.path / f"rank_{self._distributed_config.rank}.safetensors",
                metadata=_export_safetensors_metadata(metadata),
            )
        else:
            raise NotImplementedError(checkpoint_config.format)

    def _save_state_dict(self, checkpoint_config: CheckpointSaveConfig, converter: ModelConverter, metadata: dict):
        with StateDictSaver(
            checkpoint_config,
            distributed=self._distributed,
            metadata=metadata,
            base_file_name=converter.base_file_name,
        ) as context:
            # The tensor mapping may not be one-to-one. `convert_state_dict` pops all tensors from
            #   `fast_llm_state_dict` that are ready for conversion,
            #   and return a dict containing the converted tensors(s).
            #   If converting a tensor requires another one that is not yet available (e.g. for concatenation),
            #   it will remain in `fast_llm_state_dict` until that tensor is available.
            fast_llm_state_dict = {}
            for i, shard_name in enumerate(
                self._state_shard_names if checkpoint_config.optimizer_state else self._state_shard_names[:1]
            ):
                shard_split = self._state_shard[i].split(self._stage_shard_sizes, 0)
                for stage, shard in zip(self._stages_on_device.values(), shard_split):
                    for name, tensor in stage._export_shard(shard, data_type=checkpoint_config.data_type):  # noqa
                        assert name not in fast_llm_state_dict
                        fast_llm_state_dict[(name, shard_name)] = tensor
                        for exported_name, exported_tensor in converter.convert_state_dict(
                            fast_llm_state_dict, True
                        ).items():
                            context.add_tensor(exported_name, exported_tensor)
            assert not fast_llm_state_dict, list(fast_llm_state_dict)

    def load_pretrained_checkpoint(self, pretrained_config: CheckpointLoadConfig):
        if pretrained_config.format == CheckpointFormat.distributed:
            # TODO: Check if same format.
            self._load_distributed_checkpoint(pretrained_config)
        elif pretrained_config.format == CheckpointFormat.state_dict:
            self._load_state_dict_checkpoint(pretrained_config)
        elif pretrained_config.format == CheckpointFormat.external:
            self._import_checkpoint(pretrained_config)
        else:
            raise NotImplementedError(pretrained_config.format)

    def load_distributed_checkpoint_same_format(self, directory: pathlib.Path):
        # TODO: Handle barriers, ok file, etc. here
        # TODO: More safety checks
        # TODO: Integrate to load_checkpoint.
        pretrained_config = CheckpointLoadConfig(path=directory, format=CheckpointFormat.distributed)
        metadata = self.config_class.load_pretrained_metadata(pretrained_config)
        with self._LoadContext(self, safe=False, load_optimizer=True, reset_pads=False) as context:
            Assert.eq(
                metadata["state_shard_names"][: context.num_shards],
                list(self._state_shard_names[: context.num_shards]),
            )
            with safetensors.safe_open(
                directory / f"rank_{self._distributed_config.rank}.safetensors",
                framework="pt",
                device=str(self._distributed.device),
            ) as f:
                # TODO: Does this copy twice?
                self._state_shard[: context.num_shards].copy_(f.get_slice("state_shard")[: context.num_shards])
        return metadata["metadata"]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_config: CheckpointLoadConfig,
        default_config: FastLLMModelConfig = None,
        *,
        config_updates: dict[str | tuple[str, ...], typing.Any] | None = None,
        optimizer_state_names: tuple[str, ...] | None = None,
        setup: bool = True,
        mode: StageMode = StageMode.training,
        use_cpu: bool = False,
        stage_filter: set | None = None,
    ):
        metadata = cls.config_class.load_pretrained_metadata(pretrained_config)
        config = cls.config_class.from_metadata(pretrained_config, metadata, default_config, config_updates)
        if mode.support_training:
            if "state_shard_names" in metadata:
                if optimizer_state_names is None:
                    optimizer_state_names = metadata["state_shard_names"][1:]
                else:
                    Assert.eq(optimizer_state_names, metadata["state_shard_names"][1:])
            elif optimizer_state_names is None:
                raise ValueError("`optimizer_state_names` is required")
        else:
            assert optimizer_state_names is None
            optimizer_state_names = ()

        model = cls(
            config,
            optimizer_state_names=tuple(optimizer_state_names),
            stage_filter=stage_filter,
        )

        if setup:
            model.setup(Distributed(config.distributed, use_cpu=use_cpu), mode=mode)

            if mode.on_device:
                if pretrained_config.model_weights:
                    model.load_pretrained_checkpoint(pretrained_config)
                else:
                    model.initialize_weights()
        return model

    def initialize_weights(self):
        assert self._is_setup
        with self._LoadContext(self, safe=False, load_optimizer=False, reset_pads=True):
            assert self._is_setup
            for stage in self._stages:
                stage.initialize_weights()
            for name, tied_parameter in self._tied_parameters.items():
                if tied_parameter.group is not None:
                    broadcast(self._stages[tied_parameter.main_stage].weight_shard, 0, tied_parameter.group)

    def _reset_shard_pads(self, optimizer: bool = False):
        counter = 0
        for shard in self._state_shard if optimizer else self._state_shard[:1]:
            shard_split = shard.split(self._stage_shard_sizes, 0)
            for stage, stage_shard in zip(self._stages_on_device.values(), shard_split):
                counter += stage.reset_shard_pad(stage_shard)
        return counter

    class _LoadContext:
        # TODO: Improve
        def __init__(self, model: "FastLLMModel", *, safe: bool, load_optimizer: bool, reset_pads: bool):
            assert model._is_setup
            self.multi_stage = model
            self.safe = safe
            self.load_optimizer = load_optimizer
            self.num_shards = len(self.multi_stage._state_shard_names) if self.load_optimizer else 1
            self.self_shard = self.multi_stage._state_shard[: self.num_shards]
            self.reset_pads = reset_pads
            self.shard_names = self.multi_stage._state_shard_names[: self.num_shards]

        def __enter__(self):
            if self.safe:
                self.loaded = 0
                self.loaded_parameters = {}
                # Track the number of loaded entries.
                # Use nan to mark non-loaded entries.
                triton_fill(self.self_shard, math.nan)
                if self.reset_pads:
                    self.loaded += self.multi_stage._reset_shard_pads(self.load_optimizer)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not exc_type:
                if self.safe:
                    self._validate()
                if not self.load_optimizer:
                    triton_fill(self.multi_stage._state_shard[1:], 0.0)
                if self.multi_stage._mode.support_forward:
                    self.multi_stage.invalidate_buffers()
                self.multi_stage._is_loaded = True

        def _validate(self):
            errors = []
            self._check_counter(errors)
            self._check_missing(errors)
            if self.loaded_parameters:
                self._check_parameters(errors)
            if errors:
                for error in errors:
                    logger.error(error)
                raise RuntimeError("Model loading validation failed. See logs for details.")
            logger.info(f"{self.loaded:,} state entries loaded successfully")

        def _check_counter(self, errors: list[str]):
            to_load = self.self_shard.numel()
            if self.loaded != to_load:
                # Ensure the right amount of weights is loaded.
                errors.append(f"Loaded a total of {self.loaded:,}, state entries, expected {to_load:,}")

        def _check_missing(self, errors: list[str]):
            # Ensure the loaded weights have a 1-1 mapping by looking for nans.
            missing = self.self_shard.new_zeros([], dtype=torch.int64)
            # Count nans in slices of 100M parameters to limit memory usage.
            # TODO: Find better solution (triton kernel?)
            for shard_slice in self.self_shard.flatten().split(100000000):
                missing += shard_slice.isnan().sum()
            local_missing = missing.item()
            if self.multi_stage._distributed.world_group is not None:
                all_reduce(missing, group=self.multi_stage._distributed.world_group)
            global_missing = missing.item()
            if global_missing:
                errors.append(
                    f"{global_missing:,} state entries failed to load or corrupted (local={local_missing:,})."
                )
                # Determine where the missing values are coming from.
                global_total, local_total = 0, 0
                for shard_name, shard_ in zip(self.shard_names, self.self_shard):
                    shard_split = shard_.split(self.multi_stage._stage_shard_sizes, 0)
                    for stage, shard in zip(self.multi_stage._stages_on_device.values(), shard_split):
                        buffer = stage._reconstruct_from_shard(shard)
                        for i, parameter in enumerate(stage._split_buffer(buffer)):
                            missing_for_param = parameter.isnan().sum().item()
                            if missing_for_param > 0:
                                global_total += missing_for_param
                                local_values = stage._split_shard(shard)[i]
                                local_missing_for_param = local_values.isnan().sum().item()
                                local_total += local_missing_for_param
                                errors.append(
                                    f"{missing_for_param:,} values missing out of {parameter.numel():,} for parameter {stage.parameter_names[i]} in stage {stage.index}, shard {shard_name}"
                                    f" (locally {local_missing_for_param:,} out of {local_values.numel():,})"
                                )
                        missing_for_pad = buffer[-stage._global_pad :].isnan().sum().item()
                        if missing_for_pad > 0:
                            global_total += missing_for_pad
                            local_missing_for_pad = (
                                shard[-stage._shard_pad :].isnan().sum().item() if stage._shard_pad > 0 else 0
                            )
                            local_total += local_missing_for_pad
                            errors.append(
                                f"{missing_for_pad:,} values missing out of {stage._global_pad:,} for padding in stage {stage.index}, shard {shard_name}"
                                f" (locally {local_missing_for_pad:,} out of {stage._shard_pad:,})"
                            )
                if global_total != global_missing:
                    errors.append(
                        f"Incorrect global breakdown of missing state entries (expected {global_missing:,}, got {global_total:,})"
                    )
                if local_total != local_missing:
                    errors.append(
                        f"Incorrect local breakdown of missing state entries (expected {local_missing:,}, got {local_total:,})"
                    )

        def _check_parameters(self, errors: list[str]):
            loaded_shard_names = set(self.loaded_parameters)
            shard_names = set(self.shard_names)
            if loaded_shard_names != shard_names:
                errors.append(f"Incorrect loaded shards: {loaded_shard_names}!={shard_names}")
            for shard_name in shard_names & loaded_shard_names:
                counter_per_parameter = {
                    parameter_name: self.loaded_parameters[shard_name].pop(parameter_name, None)
                    for parameter_name in self.multi_stage._parameter_stages
                }
                for parameter_name, count in self.loaded_parameters[shard_name].items():
                    errors.append(
                        f'Loaded unknown parameter "{parameter_name}" for shard "{shard_name}" (count={count})'
                    )
                for parameter_name, counter in counter_per_parameter.items():
                    if self.multi_stage._parameter_stages[parameter_name] in self.multi_stage._stages_on_device:
                        if counter is None:
                            errors.append(f'Missing parameter "{parameter_name}" for shard "{shard_name}"')
                    elif counter is not None and counter > 0:
                        errors.append(f'Loaded off-device parameter : "{parameter_name}" for shard "{shard_name}"')
                distributed = self.multi_stage._distributed
                if distributed.world_group is not None:
                    counter_tensor = torch.tensor(
                        [counter or 0 for counter in counter_per_parameter.values()], dtype=torch.int64
                    ).to(distributed.device)
                    all_reduce(counter_tensor, group=distributed.world_group)
                    counter_per_parameter = {
                        parameter_name: counter
                        for parameter_name, counter in zip(counter_per_parameter, counter_tensor.tolist())
                    }
                for parameter_name, counter in counter_per_parameter.items():
                    parameter_size = (
                        self.multi_stage._stages[self.multi_stage._parameter_stages[parameter_name]]
                        .get_parameter_meta(parameter_name)
                        .global_shape.numel()
                    )
                    if counter != parameter_size:
                        errors.append(
                            f'Global counter mismatch for parameter "{parameter_name}" and shard "{shard_name}": {counter} != {parameter_size}'
                        )

        def load_state_unsafe(self, state: torch.Tensor, loaded_model: "MultiStageModel"):
            """
            Load a state from a checkpoint saved in another distributed configuration.
            """
            self_stage: Stage
            loaded_stage: Stage
            loaded_model._state_shard_meta.validate(state)
            multi_stage = self.multi_stage

            # TODO: Improve num shard selection.
            self_shard_split = multi_stage._state_shard[: state.size(0)].split(multi_stage._stage_shard_sizes, 1)
            loaded_shard_split = state.split(loaded_model._stage_shard_sizes, 1)

            counter = torch.zeros(1, dtype=torch.int64, device=multi_stage._distributed.device)
            for loaded_shard_index, loaded_stage in enumerate(loaded_model._stages_on_device.values()):
                loaded_shards = loaded_shard_split[loaded_shard_index].to(multi_stage._distributed.device).unbind(0)
                for self_shard_index, self_stage in enumerate(multi_stage._stages_on_device.values()):
                    self_stage._copy_shard_overlaps(  # noqa
                        loaded_stage,
                        self_shard_split[self_shard_index].unbind(0),
                        loaded_shards,
                        counter,
                    )
            self.loaded += counter.item()

        def import_state_tensor(self, shard_name: str, parameter_name: str, tensor: torch.Tensor | SafeTensorSlice):
            multi_stage = self.multi_stage
            stage_index = multi_stage._parameter_stages[parameter_name]
            if stage_index not in multi_stage._stage_shard_indices:
                # Tensor is not on this device.
                return 0
            stage_shard = multi_stage._state_shard[multi_stage._state_shard_names.index(shard_name)].split(
                multi_stage._stage_shard_sizes, 0
            )[multi_stage._stage_shard_indices[stage_index]]
            loaded = multi_stage._stages[stage_index]._import_state_tensor(stage_shard, parameter_name, tensor)  # noqa
            self.loaded += loaded
            if shard_name not in self.loaded_parameters:
                self.loaded_parameters[shard_name] = {}
            self.loaded_parameters[shard_name][parameter_name] = loaded

    def _load_distributed_checkpoint(self, pretrained_config: CheckpointLoadConfig):
        # TODO: More safety checks
        metadata = self.config_class.load_pretrained_metadata(pretrained_config)
        loaded_pretrained_config = pretrained_config.to_copy({"load_config": ModelConfigType.fast_llm})
        loaded_config = self.config_class.from_metadata(
            loaded_pretrained_config,
            metadata,
        )
        with self._LoadContext(
            self, safe=True, load_optimizer=pretrained_config.optimizer_state, reset_pads=True
        ) as context:
            Assert.eq(metadata["state_shard_names"][: context.num_shards], list(context.shard_names))

            for rank in range(loaded_config.distributed.world_size):
                loaded_multi_stage = self.__class__(
                    loaded_config.to_copy({("distributed", "rank"): rank}),
                    optimizer_state_names=context.shard_names[1:],
                    verbose=False,
                )
                path = pretrained_config.path / f"rank_{rank}.safetensors"
                logger.info(f"Loading from {path}")
                # TODO: skip shards without overlap.
                with safetensors.safe_open(path, framework="pt", device=str(self._distributed.device)) as f:
                    # TODO: Use self_shard
                    context.load_state_unsafe(f.get_slice("state_shard")[: context.num_shards], loaded_multi_stage)

        return metadata["metadata"]

    def _load_state_dict_checkpoint(self, pretrained_config: CheckpointLoadConfig):
        # TODO: Make into a special case of _import_state_tensor?
        # TODO: Verify more distributed configs.
        # TODO: More safety checks
        with self._LoadContext(
            self, safe=True, load_optimizer=pretrained_config.optimizer_state, reset_pads=True
        ) as context:
            index_path = pretrained_config.path / f"state_dict.safetensors.index.json"
            logger.info(f"Loading index from {index_path}")
            file_names = set(json.load(index_path.open("r"))["weight_map"].values())
            for file_name in file_names:
                logger.info(f"Loading from {pretrained_config.path / file_name}")
                with safetensors.safe_open(
                    pretrained_config.path / file_name,
                    framework="pt",
                    device=str(self._distributed.device),
                ) as f:
                    metadata = _import_safetensors_metadata(f.metadata())
                    Assert.eq(metadata["state_shard_names"][: context.num_shards], list(context.shard_names))
                    for key in f.keys():
                        parameter_name, shard_name = key.split("/", 1)
                        if shard_name in context.shard_names:
                            context.import_state_tensor(shard_name, parameter_name, f.get_slice(key))

        return metadata["metadata"]

    def _import_checkpoint(self, pretrained_config: CheckpointLoadConfig):
        # TODO: Support optimizer?
        assert not pretrained_config.optimizer_state
        # TODO: Verify more distributed configs.
        # TODO: Safety checks

        converter_class = self.base_model.architecture_cls().get_converter_class(pretrained_config.model_type)
        converter = converter_class.from_config(converter_class.load_config(pretrained_config.path))
        self._base_model_config.compare_architecture(converter.config, pretrained_config.compare_log_fn)

        state_dict = {}
        with self._LoadContext(
            self, safe=True, load_optimizer=pretrained_config.optimizer_state, reset_pads=True
        ) as context:
            for name, tensor in converter.load_weights(pretrained_config.path, self._distributed.device):
                assert name not in state_dict
                state_dict[(name, "weights")] = tensor
                for parameter_name, fast_llm_tensor in converter.convert_state_dict(state_dict, False).items():
                    context.import_state_tensor("weights", parameter_name, fast_llm_tensor)

            assert not state_dict, list(state_dict)
