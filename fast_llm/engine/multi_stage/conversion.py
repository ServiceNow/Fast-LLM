import abc
import dataclasses
import json
import logging
import pathlib
import typing

import safetensors
import torch
import yaml

from fast_llm.engine.base_model.config import BaseModelArchitectureConfig, BaseModelConfig
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ParamConverter:
    fast_llm_name: tuple[str, ...] | None
    export_name: str | None

    def export_param(self, fast_llm_value):
        return fast_llm_value

    def import_param(self, export_value):
        return export_value


@dataclasses.dataclass
class ConstantImportParamConverter(ParamConverter):
    fast_llm_value: typing.Any

    def export_param(self, fast_llm_value):
        Assert.eq(fast_llm_value, self.fast_llm_value)

    def import_param(self, export_value):
        return self.fast_llm_value


@dataclasses.dataclass
class ConstantExportParamConverter(ParamConverter):
    export_value: typing.Any

    def export_param(self, fast_llm_value):
        return self.export_value

    def import_param(self, export_value):
        Assert.eq(export_value, self.export_value)


@dataclasses.dataclass
class IgnoreImportParamConverter(ParamConverter):
    ignore_export_value: typing.Any

    def export_param(self, fast_llm_value):
        pass

    def import_param(self, export_value):
        if export_value is not self.ignore_export_value:
            logger.warning(
                f"The configuration parameter `{self.export_name}={export_value}` is ignored during conversion."
                f" If you intend to use it in Fast-LLM, make sure to set it explicitly in the model configuration."
            )


@dataclasses.dataclass
class MappedConfigParamConverter(ParamConverter):
    fast_llm_value: typing.Callable
    export_value: typing.Callable

    def export_param(self, fast_llm_value):
        return self.export_value(fast_llm_value)

    def import_param(self, export_value):
        return self.fast_llm_value(export_value)


class WeightConverter:
    def __init__(
        self,
        fast_llm_name: str | tuple[str, ...],
        export_name: str | tuple[str, ...],
        config: BaseModelArchitectureConfig | None = None,
    ):
        self.fast_llm_name: tuple[str, ...] = (fast_llm_name,) if isinstance(fast_llm_name, str) else fast_llm_name
        self.export_name: tuple[str, ...] = (export_name,) if isinstance(export_name, str) else export_name
        self._config = config

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        return weight

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        return weight


class IgnoreWeightConverter(WeightConverter):
    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        raise RuntimeError(
            f"IgnoreWeightConverter should not be used for export: {self.fast_llm_name}, {self.export_name}"
        )

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        return ()


class CopyWeightConverter(WeightConverter):
    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        return weight[0], *[weight[0][:].clone() for _ in range(len(self.export_name) - 1)]

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        return weight[0], *[weight[0][:].clone() for _ in range(len(self.fast_llm_name) - 1)]


class SplitWeightConverter(WeightConverter):
    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (merged_weight,) = weight
        return tuple(merged_weight[:].chunk(len(self.export_name)))

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        return (torch.cat([weight_[:] for weight_ in weight]),)


class ModelConverter(abc.ABC):
    base_file_name: typing.ClassVar[str]

    @classmethod
    @abc.abstractmethod
    def get_key(cls, parameter_name: str, shard_name: str) -> str:
        pass

    @abc.abstractmethod
    def convert_state_dict(
        self, state_dict: dict[str, torch.Tensor | SafeTensorSlice], export: bool
    ) -> dict[str, torch.Tensor | SafeTensorSlice]:
        pass

    @abc.abstractmethod
    def load_weights(
        self,
        directory: pathlib.Path | str,
        device,
        shard_names: list[str],
    ) -> typing.Iterator[tuple[str, str, torch.Tensor | SafeTensorSlice]]:
        pass


def _import_safetensors_metadata(metadata):
    return {key: yaml.safe_load(value) for key, value in metadata.items()}


class TrivialConverter(ModelConverter):
    base_file_name = "state_dict"

    @classmethod
    def get_key(cls, parameter_name: str, shard_name: str) -> str:
        return f"{parameter_name}/{shard_name}"

    def convert_state_dict(
        self, state_dict: dict[str, torch.Tensor | SafeTensorSlice], export: bool
    ) -> dict[str, torch.Tensor | SafeTensorSlice]:
        out_state_dict = state_dict.copy()
        state_dict.clear()
        return out_state_dict

    def load_weights(
        self,
        directory: pathlib.Path | str,
        device,
        shard_names: list[str],
    ) -> typing.Iterator[tuple[str, str, torch.Tensor | SafeTensorSlice]]:
        index_path = directory / f"state_dict.safetensors.index.json"
        logger.info(f"Loading index from {index_path}")
        file_names = set(json.load(index_path.open("r"))["weight_map"].values())
        for file_name in file_names:
            logger.info(f"Loading from {directory / file_name}")
            with safetensors.safe_open(
                directory / file_name,
                framework="pt",
                device=str(device),
            ) as f:
                metadata = _import_safetensors_metadata(f.metadata())
                Assert.eq(metadata["state_shard_names"][: len(shard_names)], list(shard_names))
                for key in f.keys():
                    parameter_name, shard_name = key.split("/", 1)
                    if shard_name in shard_names:
                        yield parameter_name, shard_name, f.get_slice(key)

        # return metadata["metadata"]


class ExternalModelConverter(ModelConverter):
    base_file_name = "model"
    _base_model_cls: type[BaseModelConfig]
    _config_converters: list[ParamConverter]

    def __init__(self, config: BaseModelArchitectureConfig):
        self.config = config
        Assert.custom(isinstance, config, self._base_model_cls.architecture_cls)
        weight_converters = self._create_weight_converters()
        self._export_converters = {
            weight_converter.fast_llm_name[0]: weight_converter
            for weight_converter in weight_converters
            if weight_converter.fast_llm_name
        }
        self._import_converters = {
            weight_converter.export_name[0]: weight_converter for weight_converter in weight_converters
        }

    @classmethod
    @abc.abstractmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        pass

    @abc.abstractmethod
    def _create_weight_converters(self) -> list[WeightConverter]:
        pass

    @classmethod
    @abc.abstractmethod
    def load_config(cls, directory: pathlib.Path | str) -> dict[str, typing.Any]:
        pass

    @classmethod
    @abc.abstractmethod
    def save_config(cls, directory: pathlib.Path | str, config: dict[str, typing.Any]):
        pass

    @classmethod
    def export_config(cls, config: BaseModelArchitectureConfig) -> dict[str, typing.Any]:
        exported_config = {}
        for converter in cls._get_config_converters():
            value = converter.export_param(
                None
                if converter.fast_llm_name is None
                else cls._get_fast_llm_attribute(config, converter.fast_llm_name)  # Noqa
            )
            if converter.export_name is not None:
                exported_config[converter.export_name] = value

        return exported_config  # Noqa

    @classmethod
    def import_config(cls, config: dict[str, typing.Any], architecture_only: bool = False):  # noqa
        kwargs = {}
        for converter in cls._get_config_converters():
            value = converter.import_param(
                None
                if converter.export_name is None or converter.export_name not in config
                else config[converter.export_name]
            )
            if converter.fast_llm_name is not None:
                kwargs[converter.fast_llm_name] = value

        config_class = cls._base_model_cls.architecture_cls if architecture_only else cls._base_model_cls
        return config_class.from_dict({}, kwargs)

    @classmethod
    def from_config(cls, config: dict[str, typing.Any], architecture_only: bool = False):
        return cls(cls.import_config(config, architecture_only=architecture_only))

    def convert_state_dict(
        self, state_dict: dict[str, torch.Tensor | SafeTensorSlice], export: bool
    ) -> dict[str, torch.Tensor | SafeTensorSlice]:
        out_state_dict = {}
        weight_converters = self._export_converters if export else self._import_converters

        for state_dict_name in list(state_dict):
            try:
                if state_dict_name not in weight_converters:
                    continue
                weight_converter: WeightConverter = weight_converters[state_dict_name]
                in_names = weight_converter.fast_llm_name if export else weight_converter.export_name
                if not all(name in state_dict for name in in_names):
                    continue
                in_weights = tuple(state_dict.pop(name) for name in in_names)
                out_names = weight_converter.export_name if export else weight_converter.fast_llm_name
                out_weights = (
                    weight_converter.export_weight(in_weights)
                    if export
                    else weight_converter.import_weight(in_weights)
                )

                Assert.eq(len(out_names), len(out_weights))

                # Set the converted weights
                for name, weight in zip(out_names, out_weights):
                    assert name not in out_state_dict
                    out_state_dict[name] = weight

            except Exception as e:
                raise ValueError(f"Cannot convert `{state_dict_name}`: {e}")

        return out_state_dict

    @classmethod
    def _get_config_converters(cls) -> list[ParamConverter]:
        if not hasattr(cls, "_config_converters"):
            cls._config_converters = cls._create_config_converters()
        return cls._config_converters

    @staticmethod
    def _get_fast_llm_attribute(config: BaseModelArchitectureConfig, name: str | tuple[str, ...]):
        if isinstance(name, str):
            name = (name,)
        val = config
        for name_ in name:
            val = getattr(val, name_)
        return val


class AutoModelConverter(ExternalModelConverter, abc.ABC):
    converter_map: dict[str, type[ExternalModelConverter]]

    @classmethod
    def import_config(cls, config: dict[str, typing.Any], architecture_only: bool = False):
        return cls.converter_map[config["model_type"]].import_config(config, architecture_only)

    @classmethod
    def from_config(cls, config: dict[str, typing.Any], architecture_only: bool = False):
        return cls.converter_map[config["model_type"]].from_config(config, architecture_only)


class HuggingfaceModelConverter(ExternalModelConverter, abc.ABC):
    model_type: str | None = None

    @classmethod
    def get_key(cls, parameter_name: str, shard_name: str) -> str:
        Assert.eq(shard_name, "weights")
        return parameter_name

    @classmethod
    @abc.abstractmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return [ConstantExportParamConverter(None, "model_type", cls.model_type)]

    @classmethod
    def load_config(cls, directory: pathlib.Path | str):
        import transformers

        config = transformers.AutoConfig.from_pretrained(directory).to_dict()
        if cls.model_type is not None:
            Assert.eq(config["model_type"], cls.model_type)
        return config

    @classmethod
    def save_config(cls, directory: pathlib.Path | str, config: dict[str, typing.Any]):
        import transformers

        transformers.CONFIG_MAPPING[config["model_type"]].from_dict(config).save_pretrained(directory)

    def load_weights(
        self,
        directory: pathlib.Path | str,
        device,
        shard_names: list[str],
    ) -> typing.Iterator[tuple[str, str, torch.Tensor | SafeTensorSlice]]:
        import transformers

        Assert.eq(shard_names, ("weights",))
        if (directory / transformers.utils.SAFE_WEIGHTS_NAME).is_file():
            paths = {directory / transformers.utils.SAFE_WEIGHTS_NAME}
        elif (directory / transformers.utils.SAFE_WEIGHTS_INDEX_NAME).is_file():
            logger.info(f"Loading index from {directory / transformers.utils.SAFE_WEIGHTS_INDEX_NAME}")
            paths = {
                directory / path
                for path in json.load((directory / transformers.utils.SAFE_WEIGHTS_INDEX_NAME).open("r"))[
                    "weight_map"
                ].values()
            }
        elif (directory / transformers.utils.WEIGHTS_NAME).is_file():
            # TODO: Prevent unsafe by default
            paths = {directory / transformers.utils.WEIGHTS_NAME}
        elif (directory / transformers.utils.WEIGHTS_INDEX_NAME).is_file():
            logger.info(f"Loading index from {directory / transformers.utils.WEIGHTS_INDEX_NAME}")
            paths = {
                directory / path
                for path in json.load((directory / transformers.utils.WEIGHTS_INDEX_NAME).open("r"))[
                    "weight_map"
                ].values()
            }
        else:
            raise FileNotFoundError(f"No compatible checkpoint found in {directory}")

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
