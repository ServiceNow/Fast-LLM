import abc
import logging
import pathlib
import typing

import torch

from fast_llm import __version__
from fast_llm.config import Config
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.checkpoint.config import CheckpointLoadMetadataConfig
from fast_llm.engine.checkpoint.state_dict import StateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import CheckpointMetadata, FastLLMModelConfig
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class WeightConverter:
    def __init__(
        self,
        fast_llm_name: str | tuple[str, ...],
        export_name: str | tuple[str, ...],
        config: Config | None = None,
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

    def __init__(self, model: "FastLLMModel"):
        super().__init__(model)
        Assert.custom(
            isinstance,
            self._model.config.base_model,
            self._model_class.get_base_model_config_class(),
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
    def _load_metadata(cls, config: CheckpointLoadMetadataConfig) -> CheckpointMetadata:
        return CheckpointMetadata(
            fast_llm_version=__version__,
            model=cls._model_class,
            format=config.format,
            config=cls._import_config(cls._load_config(config.path)),
            shards=["weights"],
        )

    @abc.abstractmethod
    def _create_weight_converters(self) -> list[WeightConverter]:
        pass

    @classmethod
    @abc.abstractmethod
    def _load_config(cls, directory: pathlib.Path | str) -> dict:
        pass

    @classmethod
    @abc.abstractmethod
    def _export_config(cls, config: FastLLMModelConfig) -> dict[str, typing.Any]:
        # TODO: not used in this class
        pass

    @classmethod
    @abc.abstractmethod
    def _import_config(cls, config: dict[str, typing.Any]) -> FastLLMModelConfig:
        pass

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

    @staticmethod
    def _get_fast_llm_attribute(config: BaseModelConfig, name: str | tuple[str, ...]) -> typing.Any:
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
    def _import_config(cls, config: dict[str, typing.Any]) -> FastLLMModelConfig:
        # TODO: ???
        return cls.handler_map[config["model_type"]]._import_config(config)
