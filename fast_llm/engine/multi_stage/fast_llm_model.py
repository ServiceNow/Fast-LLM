import logging
import math
import typing

import safetensors.torch
import torch
import yaml

from fast_llm.core.distributed import all_reduce, broadcast
from fast_llm.engine.base_model.base_model import BaseModel
from fast_llm.engine.checkpoint.config import (
    CHECKPOINT_VERSION,
    CheckpointFormat,
    CheckpointLoadConfig,
    CheckpointSaveConfig,
    ModelConfigType,
)
from fast_llm.engine.checkpoint.state_dict import StateDictConverter, StateDictSaver, TrivialConverter
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, StageMode
from fast_llm.engine.multi_stage.multi_stage import MultiStageModel
from fast_llm.functional.triton.pointwise import triton_fill
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
        config: CheckpointSaveConfig,
        metadata: dict | None = None,
    ):
        # TODO: Handle barriers, ok file, mkdir, etc. here

        num_shards = len(self._state_shard_names) if config.optimizer_state else 1
        metadata = {
            "checkpoint_type": CheckpointFormat.distributed.value,
            "checkpoint_version": str(CHECKPOINT_VERSION),
            "fast_llm_config": self._fast_llm_config.to_serialized(),
            "state_shard_names": list(self._state_shard_names[:num_shards]),
            "metadata": {} if metadata is None else metadata,
        }

        # TODO: Simplify branching.
        if config.format == CheckpointFormat.external:
            # TODO: Support optimizer?
            assert not config.optimizer_state
            converter_class = self._base_model_config.get_converter_class(config.model_type)
            exported_config = converter_class.export_config(self._base_model_config)
            converter_class.save_config(config.path, exported_config)
            self._save_state_dict(
                config,
                converter_class(self._base_model_config),
                {
                    "fast_llm_metadata": metadata,
                    "model_config": exported_config,
                    "format": "pt",
                },
            )
        elif config.format == CheckpointFormat.state_dict:
            self._save_state_dict(config, TrivialConverter(), metadata)
        elif config.format == CheckpointFormat.distributed:
            if self._distributed_config.rank == 0:
                yaml.safe_dump(metadata, (config.path / "metadata.yaml").open("w"))
            safetensors.torch.save_file(
                tensors={"state_shard": self._state_shard[:num_shards]},
                filename=config.path / f"rank_{self._distributed_config.rank}.safetensors",
                metadata=_export_safetensors_metadata(metadata),
            )
        else:
            raise NotImplementedError(config.format)

    def _save_state_dict(self, config: CheckpointSaveConfig, converter: StateDictConverter, metadata: dict):
        with StateDictSaver(
            config,
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
            for parameter_name, shard_name, tensor in self.get_state_tensor_iterator(
                self._state_shard_names if config.optimizer_state else self._state_shard_names[:1], config.data_type
            ):
                if shard_name not in fast_llm_state_dict:
                    fast_llm_state_dict[shard_name] = {}
                shard_state_dict = fast_llm_state_dict[shard_name]
                assert parameter_name not in shard_state_dict
                shard_state_dict[parameter_name] = tensor
                for exported_name, exported_tensor in converter.convert_state_dict(shard_state_dict, True).items():
                    context.add_tensor(converter.get_key(exported_name, shard_name), exported_tensor)
            for shard_name, shard_state_dict in fast_llm_state_dict.items():
                assert not shard_state_dict, (shard_name, list(fast_llm_state_dict))

    def load_checkpoint(self, config: CheckpointLoadConfig):
        # TODO: Simplify branching.
        # TODO: Test with more distributed configs.
        # TODO: Safety checks
        # TODO: Handle barriers, ok file, etc. here
        metadata = self.config_class.load_pretrained_metadata(config)
        if config.format == CheckpointFormat.distributed:
            # TODO: Check if same format.
            self._load_distributed_checkpoint(config, metadata)
        elif config.format == CheckpointFormat.state_dict:
            self._load_state_dict(config, TrivialConverter())
        elif config.format == CheckpointFormat.external:
            # TODO: Support optimizer.
            assert not config.optimizer_state
            converter_class = self.base_model.architecture_cls().get_converter_class(config.model_type)
            converter = converter_class.from_config(converter_class.load_config(config.path))
            self._base_model_config.compare_architecture(converter.config, config.compare_log_fn)
            self._load_state_dict(config, converter)
        else:
            raise NotImplementedError(config.format)
        return metadata.get("metadata")

    def _load_state_dict(self, config: CheckpointLoadConfig, converter: StateDictConverter):
        num_shards = len(self._state_shard_names) if config.optimizer_state else 1
        with self._SafeLoadContext(self, num_shards=num_shards) as context:
            state_dict = {}
            for parameter_name, shard_name, tensor in converter.load_weights(
                config.path, self._distributed.device, self._state_shard_names[:num_shards]
            ):
                if shard_name not in state_dict:
                    state_dict[shard_name] = {}
                shard_state_dict = state_dict[shard_name]
                assert parameter_name not in shard_state_dict
                shard_state_dict[parameter_name] = tensor
                for parameter_name, fast_llm_tensor in converter.convert_state_dict(shard_state_dict, False).items():
                    stage_index = self._parameter_stages[parameter_name]
                    if stage_index not in self._stage_shard_indices:
                        # Tensor is not on this device.
                        return 0
                    stage_shard = self._state_shard[self._state_shard_names.index(shard_name)].split(
                        self._stage_shard_sizes, 0
                    )[self._stage_shard_indices[stage_index]]
                    loaded = self._stages[stage_index]._import_state_tensor(
                        stage_shard, parameter_name, fast_llm_tensor
                    )  # noqa
                    context.mark_as_loaded(loaded, (parameter_name, shard_name))

            for shard_name, shard_state_dict in state_dict.items():
                assert not shard_state_dict, (shard_name, list(state_dict))

        self._finalize_load(reset_optimizer=not config.optimizer_state)

    def _load_distributed_checkpoint(self, config: CheckpointLoadConfig, metadata: dict):
        # TODO: More safety checks
        loaded_config_dict = config.to_copy({"load_config": ModelConfigType.fast_llm})
        loaded_config = self.config_class.from_metadata(loaded_config_dict, metadata)
        num_shards = self._num_state_shards if config.optimizer_state else 1
        Assert.eq(metadata["state_shard_names"][:num_shards], list(self._state_shard_names[:num_shards]))

        if (
            loaded_config.to_serialized(verbose=None) == self._fast_llm_config.to_serialized(verbose=None)
            and config.optimizer_state
        ):
            logger.info("Checkpoint format matches, using fast load")
            # TODO: Add version without optimizer state?
            with safetensors.safe_open(
                config.path / f"rank_{self._distributed_config.rank}.safetensors",
                framework="pt",
                device=str(self._distributed.device),
            ) as f:
                # TODO: Does this copy twice?
                self._state_shard[:num_shards].copy_(f.get_slice("state_shard")[:num_shards])
        else:
            logger.info("Checkpoint format doesn't match, using safe load")
            self._base_model_config.compare_architecture(loaded_config.base_model, config.compare_log_fn)
            with self._SafeLoadContext(self, num_shards=num_shards) as context:
                for rank in range(loaded_config.distributed.world_size):
                    loaded_model = self.__class__(
                        loaded_config.to_copy({("distributed", "rank"): rank}),
                        optimizer_state_names=self._state_shard_names[1:num_shards],
                        verbose=False,
                    )
                    path = config.path / f"rank_{rank}.safetensors"
                    logger.info(f"Loading from {path}")
                    # TODO: skip shards without overlap.
                    with safetensors.safe_open(path, framework="pt", device=str(self._distributed.device)) as f:
                        # TODO: Use self_shard
                        loaded_shard = f.get_slice("state_shard")[:num_shards]
                        loaded_model._state_shard_meta.validate(loaded_shard)

                        # TODO: Improve num shard selection.
                        self_shard_split = self._state_shard[: loaded_shard.size(0)].split(self._stage_shard_sizes, 1)
                        loaded_shard_split = loaded_shard.split(loaded_model._stage_shard_sizes, 1)

                        counter = torch.zeros(1, dtype=torch.int64, device=self._distributed.device)
                        for loaded_shard_index, loaded_stage in enumerate(loaded_model._stages_on_device.values()):
                            loaded_shards = (
                                loaded_shard_split[loaded_shard_index].to(self._distributed.device).unbind(0)
                            )
                            for self_shard_index, self_stage in enumerate(self._stages_on_device.values()):
                                self_stage._copy_shard_overlaps(  # noqa
                                    loaded_stage,
                                    self_shard_split[self_shard_index].unbind(0),
                                    loaded_shards,
                                    counter,
                                )
                        context.mark_as_loaded(counter.item())
        self._finalize_load(reset_optimizer=not config.optimizer_state)

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
                    model.load_checkpoint(pretrained_config)
                else:
                    model.initialize_weights()
        return model

    def initialize_weights(self):
        assert self._is_setup
        for stage in self._stages:
            stage.initialize_weights()
        for name, tied_parameter in self._tied_parameters.items():
            if tied_parameter.group is not None:
                broadcast(self._stages[tied_parameter.main_stage].weight_shard, 0, tied_parameter.group)
        self._finalize_load(reset_optimizer=True)

    def _finalize_load(self, reset_optimizer: bool = True):
        if reset_optimizer:
            triton_fill(self._state_shard[1:], 0.0)
        if self._mode.support_forward:
            self.invalidate_buffers()
        self._is_loaded = True

    class _SafeLoadContext:
        # TODO: Improve
        def __init__(self, model: "FastLLMModel", *, num_shards: int):
            self._model = model
            self._num_shards = num_shards
            self._self_shard = self._model._state_shard[: self._num_shards]

        def __enter__(self):
            self._loaded = 0
            self._loaded_parameters = {}
            # Track the number of loaded entries.
            # Use nan to mark non-loaded entries.
            triton_fill(self._self_shard, math.nan)
            # Reset and count shard pads
            for shard in self._model._state_shard[: self._num_shards]:
                shard_split = shard.split(self._model._stage_shard_sizes, 0)
                for stage, stage_shard in zip(self._model._stages_on_device.values(), shard_split):
                    self._loaded += stage.reset_shard_pad(stage_shard)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not exc_type:
                self._validate()

        def mark_as_loaded(self, count: int, parameter: tuple[str, str] | None = None):
            self._loaded += count
            if parameter is not None:
                parameter_name, shard_name = parameter
                if shard_name not in self._loaded_parameters:
                    self._loaded_parameters[shard_name] = {}
                Assert.not_incl(parameter_name, self._loaded_parameters[shard_name])
                self._loaded_parameters[shard_name][parameter_name] = count

        def _validate(self):
            errors = []
            self._check_counter(errors)
            self._check_missing(errors)
            if self._loaded_parameters:
                self._check_parameters(errors)
            if errors:
                for error in errors:
                    logger.error(error)
                raise RuntimeError("Model loading validation failed. See logs for details.")
            logger.info(f"{self._loaded:,} state entries loaded successfully")

        def _check_counter(self, errors: list[str]):
            to_load = self._self_shard.numel()
            if self._loaded != to_load:
                # Ensure the right amount of weights is loaded.
                errors.append(f"Loaded a total of {self._loaded:,}, state entries, expected {to_load:,}")

        def _check_missing(self, errors: list[str]):
            # Ensure the loaded weights have a 1-1 mapping by looking for nans.
            missing = self._self_shard.new_zeros([], dtype=torch.int64)
            # Count nans in slices of 100M parameters to limit memory usage.
            # TODO: Find better solution (triton kernel?)
            for shard_slice in self._self_shard.flatten().split(100000000):
                missing += shard_slice.isnan().sum()
            local_missing = missing.item()
            if self._model._distributed.world_group is not None:
                all_reduce(missing, group=self._model._distributed.world_group)
            global_missing = missing.item()
            if global_missing:
                errors.append(
                    f"{global_missing:,} state entries failed to load or corrupted (local={local_missing:,})."
                )
                # Determine where the missing values are coming from.
                global_total, local_total = 0, 0
                for shard_name, shard_ in zip(self._model._state_shard_names[: self._num_shards], self._self_shard):
                    shard_split = shard_.split(self._model._stage_shard_sizes, 0)
                    for stage, shard in zip(self._model._stages_on_device.values(), shard_split):
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
            loaded_shard_names = set(self._loaded_parameters)
            shard_names = set(self._model._state_shard_names[: self._num_shards])
            if loaded_shard_names != shard_names:
                errors.append(f"Incorrect loaded shards: {loaded_shard_names}!={shard_names}")
            for shard_name in shard_names & loaded_shard_names:
                counter_per_parameter = {
                    parameter_name: self._loaded_parameters[shard_name].pop(parameter_name, None)
                    for parameter_name in self._model._parameter_stages
                }
                for parameter_name, count in self._loaded_parameters[shard_name].items():
                    errors.append(
                        f'Loaded unknown parameter "{parameter_name}" for shard "{shard_name}" (count={count})'
                    )
                for parameter_name, counter in counter_per_parameter.items():
                    if self._model._parameter_stages[parameter_name] in self._model._stages_on_device:
                        if counter is None:
                            errors.append(f'Missing parameter "{parameter_name}" for shard "{shard_name}"')
                    elif counter is not None and counter > 0:
                        errors.append(f'Loaded off-device parameter : "{parameter_name}" for shard "{shard_name}"')
                distributed = self._model._distributed
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
                        self._model._stages[self._model._parameter_stages[parameter_name]]
                        .get_parameter_meta(parameter_name)
                        .global_shape.numel()
                    )
                    if counter != parameter_size:
                        errors.append(
                            f'Global counter mismatch for parameter "{parameter_name}" and shard "{shard_name}": {counter} != {parameter_size}'
                        )
