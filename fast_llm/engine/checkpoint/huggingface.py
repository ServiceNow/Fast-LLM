import abc
import json
import pathlib
import typing

import safetensors
import torch

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, CheckpointSaveMetadataConfig
from fast_llm.engine.checkpoint.external import (
    ConstantExportParamConverter,
    ExternalStateDictCheckpointHandler,
    ParamConverter,
    logger,
)
from fast_llm.engine.multi_stage.config import CheckpointMetadata
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert


class HuggingfaceStateDictCheckpointHandler(ExternalStateDictCheckpointHandler, abc.ABC):

    def _save_serialized_metadata(self, config: CheckpointSaveMetadataConfig, metadata: dict, index: dict) -> None:
        path = config.path / f"{self.base_file_name}.safetensors.index.json"
        logger.info(f"Saving index to {path}")
        # Save the index.
        json.dump(
            {"metadata": metadata, "weight_map": index},
            path.open("w"),
            indent=4,
        )

    def _serialize_metadata(self, config: CheckpointSaveMetadataConfig, metadata: CheckpointMetadata) -> dict:
        huggingface_config = self._export_config(self._model.config.base_model)
        self._save_config(config.path, huggingface_config)
        return {
            "fast_llm_metadata": metadata.to_dict(),
            "model_config": huggingface_config,
            "format": "pt",
        }

    def load(self, config: CheckpointLoadConfig, metadata: CheckpointMetadata) -> None:
        assert not config.optimizer_state
        self._model.config.base_model.compare_architecture(metadata.config.base_model, config.compare_log_fn)
        super().load(config, metadata)

    @classmethod
    def get_huggingface_model_type(self) -> str:
        # We assume the two names match, but derived classes can make it different.
        return self.format.name

    @classmethod
    def _get_key(cls, parameter_name: str, shard_name: str) -> str:
        Assert.eq(shard_name, "weights")
        return parameter_name

    @classmethod
    @abc.abstractmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return [
            ConstantExportParamConverter(
                export_names=(("model_type",),), export_value=cls.get_huggingface_model_type()
            )
        ]

    @classmethod
    def _load_config(cls, directory: pathlib.Path | str) -> dict:
        import transformers

        config = transformers.AutoConfig.from_pretrained(directory).to_dict()
        Assert.eq(config["model_type"], cls.get_huggingface_model_type())
        return config

    @classmethod
    def _save_config(cls, directory: pathlib.Path | str, config: dict[str, typing.Any]) -> None:
        import transformers

        transformers.CONFIG_MAPPING[config["model_type"]].from_dict(config).save_pretrained(directory)

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
