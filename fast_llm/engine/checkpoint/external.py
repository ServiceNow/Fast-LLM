import abc
import dataclasses
import logging
import pathlib
import typing

import torch

from fast_llm import __version__
from fast_llm.config import MISSING
from fast_llm.engine.base_model.config import BaseModelArchitectureConfig
from fast_llm.engine.checkpoint.config import CheckpointLoadMetadataConfig
from fast_llm.engine.checkpoint.state_dict import StateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import CheckpointMetadata, FastLLMModelConfig
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert, get_nested_dict_value, set_nested_dict_value

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class ParamConverter(abc.ABC):
    fast_llm_names: tuple[tuple[str, ...], ...] = ()  # Array of fast-llm names, in nested (tuple) format.
    export_names: tuple[tuple[str, ...], ...] = ()  # Array of export names, in nested (tuple) format.

    @abc.abstractmethod
    def export_params(self, fast_llm_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        pass

    @abc.abstractmethod
    def import_params(self, export_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        pass


@dataclasses.dataclass(kw_only=True)
class RenameParamConverter(ParamConverter):

    def __post_init__(self) -> None:
        Assert.eq(len(self.fast_llm_names), 1)
        Assert.eq(len(self.export_names), 1)

    def export_params(self, fast_llm_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        return fast_llm_values

    def import_params(self, export_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        return export_values


# def __repr__(self):
#     return f"RenameParamConverter({'.'.join(self.fast_llm_names[0])} <--> {'.'.join(self.export_names[0])})"


@dataclasses.dataclass(kw_only=True)
class ConstantImportParamConverter(ParamConverter):
    fast_llm_value: typing.Any = MISSING

    def __post_init__(self):
        Assert.eq(len(self.fast_llm_names), 1)
        Assert.eq(len(self.export_names), 0)

    def export_params(self, fast_llm_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        Assert.eq(fast_llm_values[0], self.fast_llm_value)
        return ()

    def import_params(self, export_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        return (self.fast_llm_value,)


@dataclasses.dataclass(kw_only=True)
class ConstantExportParamConverter(ParamConverter):
    export_value: typing.Any = MISSING

    def __post_init__(self):
        Assert.eq(len(self.fast_llm_names), 0)
        Assert.eq(len(self.export_names), 1)

    def export_params(self, fast_llm_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        return (self.export_value,)

    def import_params(self, export_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        Assert.eq(export_values[0], self.export_value)
        return ()


@dataclasses.dataclass(kw_only=True)
class IgnoreImportParamConverter(ParamConverter):
    ignore_export_value: typing.Any = MISSING

    def __post_init__(self):
        Assert.eq(len(self.fast_llm_names), 0)
        Assert.eq(len(self.export_names), 1)

    def export_params(self, fast_llm_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        return (MISSING,)

    def import_params(self, export_values):
        if export_values[0] not in (self.ignore_export_value, MISSING):
            logger.warning(
                f"The configuration parameter `{self.export_names[0]}={export_values[0]}` is ignored during conversion."
                f" If you intend to use it in Fast-LLM, make sure to set it explicitly in the model configuration."
            )
        return ()


@dataclasses.dataclass(kw_only=True)
class MappedConfigParamConverter(ParamConverter):
    fast_llm_value: typing.Callable[[typing.Any], typing.Any] = lambda x: x
    export_value: typing.Callable[[typing.Any], typing.Any] = lambda x: x

    def __post_init__(self) -> None:
        Assert.eq(len(self.fast_llm_names), 1)
        Assert.eq(len(self.export_names), 1)

    def export_params(self, fast_llm_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        return (self.export_value(fast_llm_values[0]),)

    def import_params(self, export_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        return (self.fast_llm_value(export_values[0]),)


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


class IgnoreImportWeightConverter(WeightConverter):
    def __post_init__(self):
        Assert.eq(len(self.fast_llm_name), 0)
        Assert.gt(len(self.export_name), 0)

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        raise RuntimeError(
            f"IgnoreImportWeightConverter should not be used for export: {self.fast_llm_name}, {self.export_name}"
        )

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        return ()


class IgnoreExportWeightConverter(WeightConverter):
    def __post_init__(self):
        Assert.gt(len(self.fast_llm_name), 0)
        Assert.eq(len(self.export_name), 0)

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        return ()

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        raise RuntimeError(
            f"IgnoreExportWeightConverter should not be used for import: {self.fast_llm_name}, {self.export_name}"
        )


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


class ExternalStateDictCheckpointHandler(StateDictCheckpointHandler):
    _model_class: typing.ClassVar[FastLLMModelConfig]
    _config_converters: list[ParamConverter]

    def __init__(self, model: "FastLLMModel"):
        super().__init__(model)
        Assert.custom(
            isinstance,
            self._model.config.base_model,
            self._model_class.get_base_model_config_class().architecture_class,
        )
        weight_converters = self._create_weight_converters()
        self._export_converters = {
            weight_converter.fast_llm_name[0]: weight_converter
            for weight_converter in weight_converters
            if weight_converter.fast_llm_name
        }
        self._import_converters = {
            weight_converter.export_name[0]: weight_converter
            for weight_converter in weight_converters
            if weight_converter.export_name
        }

    @classmethod
    def load_metadata(cls, config: CheckpointLoadMetadataConfig) -> CheckpointMetadata:
        imported_model_config = cls._import_config(cls._load_config(config.path), True)
        return CheckpointMetadata(
            fast_llm_version=__version__,
            model=cls._model_class,
            format=config.format,
            config=cls._model_class.from_dict({"base_model": imported_model_config.to_serialized()}),
            shards=["weights"],
        )

    @classmethod
    @abc.abstractmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        pass

    @abc.abstractmethod
    def _create_weight_converters(self) -> list[WeightConverter]:
        pass

    @classmethod
    @abc.abstractmethod
    def _load_config(cls, directory: pathlib.Path | str) -> dict:
        pass

    @classmethod
    def _export_config(cls, config: BaseModelArchitectureConfig) -> dict[str, typing.Any]:
        # TODO v0.3: not used in this class
        exported_config = {}
        for converter in cls._get_config_converters():
            try:
                values = converter.export_params(
                    tuple(
                        cls._get_fast_llm_attribute(config, fast_llm_name)
                        for fast_llm_name in converter.fast_llm_names
                    )
                )
                for export_name, value in zip(converter.export_names, values, strict=True):
                    if value is not MISSING:
                        set_nested_dict_value(exported_config, export_name, value)
            except Exception as e:
                raise RuntimeError(f"Config conversion failed for converter {converter}", *e.args)

        return exported_config  # Noqa

    @classmethod
    def _import_config(
        cls, config: dict[str, typing.Any], architecture_only: bool = False
    ) -> BaseModelArchitectureConfig:  # noqa
        kwargs = {}
        for converter in cls._get_config_converters():
            try:
                values = ()
                for export_name in converter.export_names:
                    try:
                        value = get_nested_dict_value(config, export_name)
                    except KeyError:
                        value = MISSING
                    values = values + (value,)
                values = converter.import_params(values)
                for fast_llm_name, value in zip(converter.fast_llm_names, values, strict=True):
                    if value is MISSING:
                        # Missing values need to be handled in dedicated converters,
                        # because implicit / default values may not match.
                        # TODO: Different behavior from other uses of MISSING. Use different tag?
                        raise ValueError(f"Missing converted value for fast-llm parameter {fast_llm_name}")
                    if fast_llm_name in kwargs:
                        raise ValueError(f"Duplicate converted value for fast-llm parameter {fast_llm_name}")
                    kwargs[fast_llm_name] = value
            except Exception as e:
                raise RuntimeError(f"Config conversion failed for converter {converter}", *e.args)

        config_class = cls._model_class.get_base_model_config_class()
        if architecture_only:
            config_class = config_class.architecture_class
        return config_class.from_dict({}, kwargs)

    def _convert_state_dict(
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
    def _get_fast_llm_attribute(config: BaseModelArchitectureConfig, name: str | tuple[str, ...]) -> typing.Any:
        if isinstance(name, str):
            name = (name,)
        val = config
        for name_ in name:
            val = getattr(val, name_)
        return val


class AutoStateDictCheckpointHandler(ExternalStateDictCheckpointHandler, abc.ABC):
    handler_map: dict[str, type[ExternalStateDictCheckpointHandler]]

    @classmethod
    def get_handler_class(cls, format: str) -> type[ExternalStateDictCheckpointHandler]:
        if format in cls.handler_map:
            return cls.handler_map[format]
        elif format == "auto":
            return cls
        else:
            raise NotImplementedError(format)

    # TODO: load_metadata???

    @classmethod
    def _import_config(
        cls, config: dict[str, typing.Any], architecture_only: bool = False
    ) -> BaseModelArchitectureConfig:
        # TODO: ???
        return cls.handler_map[config["model_type"]]._import_config(config, architecture_only)
