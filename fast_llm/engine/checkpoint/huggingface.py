import abc
import json
import pathlib
import shutil
import typing

import safetensors
import torch

from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, CheckpointSaveConfig, CheckpointSaveMetadataConfig
from fast_llm.engine.checkpoint.external import ExternalStateDictCheckpointHandler, WeightConverter, logger
from fast_llm.engine.multi_stage.config import CheckpointMetadata, FastLLMModelConfig
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert, safe_merge_dicts

if typing.TYPE_CHECKING:
    import transformers


class HuggingFaceBaseModelConverter:
    @classmethod
    @abc.abstractmethod
    def import_config(cls, config: dict) -> dict:
        pass

    @classmethod
    @abc.abstractmethod
    def export_config(cls, config: BaseModelConfig) -> dict:
        pass

    @classmethod
    @abc.abstractmethod
    def get_converters(cls, config: BaseModelConfig) -> list[WeightConverter]:
        pass


class HuggingfaceStateDictCheckpointHandler(ExternalStateDictCheckpointHandler, abc.ABC):
    architecture: typing.ClassVar[str]
    base_model_converter_class: typing.ClassVar[type[HuggingFaceBaseModelConverter]]

    @classmethod
    @abc.abstractmethod
    def get_transformers_configuration_class(cls) -> type["transformers.PretrainedConfig"]:
        pass

    @classmethod
    def get_model_files(cls) -> tuple[str | None, str | None, str | None]:
        return None, None, None

    @classmethod
    def _save_serialized_metadata(cls, config: CheckpointSaveMetadataConfig, metadata: dict, index: dict) -> None:
        config.path.mkdir(parents=True, exist_ok=True)
        path = config.path / f"{cls.base_file_name}.safetensors.index.json"
        logger.info(f"Saving index to {path}")
        # Save the index.
        json.dump(
            {"metadata": metadata, "weight_map": index},
            path.open("w"),
            indent=4,
        )

    def _serialize_metadata(self, config: CheckpointSaveMetadataConfig, metadata: CheckpointMetadata) -> dict:
        huggingface_config = self._export_config(self._model.config)
        self._save_config(config.path, huggingface_config)
        return {
            "fast_llm_metadata": metadata.to_dict(),
            "model_config": huggingface_config,
            "format": "pt",
        }

    def load(self, config: CheckpointLoadConfig) -> dict[str, typing.Any] | None:
        assert not config.optimizer_state
        metadata = self._model.config.load_metadata(config)
        self._model.config.base_model.compare_architecture(metadata.config.base_model, logger.warning)
        super().load(config)

    def save(self, config: CheckpointSaveConfig, metadata: CheckpointMetadata) -> None:
        super().save(config, metadata)
        # Copy the modeling files to the output directory
        modeling_file, configuration_file, generation_utils_file = self.get_model_files()
        if configuration_file is not None:
            shutil.copy(configuration_file, config.path)
        if modeling_file is not None:
            shutil.copy(modeling_file, config.path)
        if generation_utils_file is not None:
            shutil.copy(generation_utils_file, config.path)
            gen_config = pathlib.Path(generation_utils_file).parent / "generation_config.json"
            if gen_config.exists():
                shutil.copy(gen_config, config.path)

    @classmethod
    def get_huggingface_model_type(self) -> str:
        # We assume the two names match, but derived classes can make it different.
        return self.format.name

    @classmethod
    def _get_key(cls, parameter_name: str, shard_name: str) -> str:
        Assert.eq(shard_name, "weights")
        return parameter_name

    # Use custom config instead of relying on the transformers library
    @classmethod
    def _load_config(cls, directory: pathlib.Path | str) -> dict:
        config = cls.get_transformers_configuration_class().from_pretrained(directory).to_dict()
        Assert.eq(config["model_type"], cls.get_huggingface_model_type())
        return config

    @classmethod
    def _save_config(cls, directory: pathlib.Path | str, config: dict[str, typing.Any]) -> None:
        cls.get_transformers_configuration_class().from_dict(config).save_pretrained(directory)

    @classmethod
    def _export_config(cls, config: FastLLMModelConfig) -> dict[str, typing.Any]:
        return safe_merge_dicts(
            cls.base_model_converter_class.export_config(config.base_model),
            {
                "model_type": cls.get_huggingface_model_type(),
                "architecture": cls.architecture,
            },
        )

    @classmethod
    def _import_config(cls, config: dict[str, typing.Any]) -> FastLLMModelConfig:
        Assert.eq(config["model_type"], cls.get_huggingface_model_type())
        Assert.eq(config["architecture"], cls.architecture)
        return cls._model_class.from_dict({"base_model": cls.base_model_converter_class.import_config(config)})

    def _create_weight_converters(
        self,
    ) -> list[WeightConverter]:
        return self.base_model_converter_class.get_converters(self._model.config.base_model)

    def _load_weights(
        self, config: CheckpointLoadConfig, device
    ) -> typing.Iterator[tuple[str, str, torch.Tensor | SafeTensorSlice]]:
        import transformers

        Assert.eq(self.get_shard_names(config), ("weights",))
        if (config.path / transformers.utils.SAFE_WEIGHTS_NAME).is_file():
            paths = {config.path / transformers.utils.SAFE_WEIGHTS_NAME}
        elif (config.path / transformers.utils.SAFE_WEIGHTS_INDEX_NAME).is_file():
            logger.info(f"Loading index from {config.path / transformers.utils.SAFE_WEIGHTS_INDEX_NAME}")
            paths = {
                config.path / path
                for path in json.load((config.path / transformers.utils.SAFE_WEIGHTS_INDEX_NAME).open("r"))[
                    "weight_map"
                ].values()
            }
        elif (config.path / transformers.utils.WEIGHTS_NAME).is_file():
            # TODO: Prevent unsafe by default
            paths = {config.path / transformers.utils.WEIGHTS_NAME}
        elif (config.path / transformers.utils.WEIGHTS_INDEX_NAME).is_file():
            logger.info(f"Loading index from {config.path / transformers.utils.WEIGHTS_INDEX_NAME}")
            paths = {
                config.path / path
                for path in json.load((config.path / transformers.utils.WEIGHTS_INDEX_NAME).open("r"))[
                    "weight_map"
                ].values()
            }
        else:
            raise FileNotFoundError(f"No compatible checkpoint found in {config.path}")

        for path in paths:
            logger.info(f"Loading from {path}")
            if path.suffix == ".safetensors":
                with safetensors.safe_open(path, framework="pt", device=str(device)) as f:
                    for key in f.keys():
                        yield key, "weights", f.get_slice(key)
            elif path.suffix == ".bin":
                # TODO: Prevent unsafe by default
                yield from torch.load(path)
            else:
                raise NotImplementedError(f"Unknown file format for {path}")
