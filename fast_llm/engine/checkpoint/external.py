import abc
import dataclasses
import logging
import pathlib
import typing

import torch

from fast_llm import __version__
from fast_llm.config import Config, FieldHint, set_nested_dict_value
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.checkpoint.config import CheckpointLoadMetadataConfig
from fast_llm.engine.checkpoint.state_dict import StateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import CheckpointMetadata, FastLLMModelConfig
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


_MISSING = object()


def _get_nested(d: dict, path: tuple[str, ...], default=_MISSING):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            if default is _MISSING:
                raise KeyError(f"Missing key {'.'.join(path)} in HF config dict")
            return default
        cur = cur[key]
    return cur


def _has_nested(d: dict, path: tuple[str, ...]) -> bool:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return True


def _get_attr_path(config: Config, path: tuple[str, ...]) -> typing.Any:
    cur = config
    for name in path:
        cur = getattr(cur, name)
    return cur


# ============================================================
# Config conversion primitives (declarative)
# ============================================================


class ConfigConverter(abc.ABC):
    """A declarative description of how one or more Fast-LLM config fields map to one or more HF config keys.

    Each primitive owns a set of ``fast_llm_paths`` (tuples of attribute names rooted at the section's config) and
    ``hf_paths`` (tuples of dict keys rooted at the section's HF subdict). The walker calls ``export_to`` to produce
    HF entries from a Fast-LLM config object, and ``import_to`` to produce a Fast-LLM config dict from an HF dict.
    """

    fast_llm_paths: tuple[tuple[str, ...], ...] = ()
    hf_paths: tuple[tuple[str, ...], ...] = ()

    @property
    def consumed_fast_llm_fields(self) -> set[str]:
        """Top-level Fast-LLM field names this primitive consumes at the current section level.

        Used by the section walker for the architecture-hint coverage check.
        """
        return {path[0] for path in self.fast_llm_paths if path}

    @abc.abstractmethod
    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None: ...

    @abc.abstractmethod
    def import_to(self, hf_dict: dict, fast_llm_out: dict, parent_context: dict | None = None) -> None: ...


class RenameConfigConverter(ConfigConverter):
    """One-to-one rename between a Fast-LLM attribute path and an HF dict path."""

    def __init__(self, fast_llm_path: tuple[str, ...], hf_path: tuple[str, ...]):
        self.fast_llm_paths = (fast_llm_path,)
        self.hf_paths = (hf_path,)

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        value = _get_attr_path(fast_llm_config, self.fast_llm_paths[0])
        set_nested_dict_value(hf_out, self.hf_paths[0], value)

    def import_to(self, hf_dict: dict, fast_llm_out: dict, parent_context: dict | None = None) -> None:
        value = _get_nested(hf_dict, self.hf_paths[0])
        set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], value)


class ConstantExportConfigConverter(ConfigConverter):
    """Write a constant to the HF dict on export. On import, assert that the HF dict has this constant value.

    Used when a HF format requires a key whose value Fast-LLM doesn't store (or always pins to a constant).
    """

    def __init__(self, hf_path: tuple[str, ...], value: typing.Any):
        self.hf_paths = (hf_path,)
        self._value = value

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        set_nested_dict_value(hf_out, self.hf_paths[0], self._value)

    def import_to(self, hf_dict: dict, fast_llm_out: dict, parent_context: dict | None = None) -> None:
        if _has_nested(hf_dict, self.hf_paths[0]):
            actual = _get_nested(hf_dict, self.hf_paths[0])
            Assert.eq(actual, self._value)


class ConstantImportConfigConverter(ConfigConverter):
    """Inject a constant into the Fast-LLM dict on import. On export, assert the config matches the constant.

    Used when a Fast-LLM field is required but the HF format implies a fixed value (e.g., gated MLP for Llama).
    """

    def __init__(self, fast_llm_path: tuple[str, ...], value: typing.Any):
        self.fast_llm_paths = (fast_llm_path,)
        self._value = value

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        actual = _get_attr_path(fast_llm_config, self.fast_llm_paths[0])
        Assert.eq(actual, self._value)

    def import_to(self, hf_dict: dict, fast_llm_out: dict, parent_context: dict | None = None) -> None:
        set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], self._value)


class DefaultConfigConverter(ConfigConverter):
    """Rename with an HF-side fallback used when the HF key is missing on import.

    ``hf_default_fn`` is called with the full HF dict if the path is absent; otherwise it's a plain rename.
    On export, behaves like ``RenameConfigConverter``.
    """

    def __init__(
        self,
        fast_llm_path: tuple[str, ...],
        hf_path: tuple[str, ...],
        hf_default_fn: typing.Callable[[dict], typing.Any],
    ):
        self.fast_llm_paths = (fast_llm_path,)
        self.hf_paths = (hf_path,)
        self._hf_default_fn = hf_default_fn

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        value = _get_attr_path(fast_llm_config, self.fast_llm_paths[0])
        set_nested_dict_value(hf_out, self.hf_paths[0], value)

    def import_to(self, hf_dict: dict, fast_llm_out: dict, parent_context: dict | None = None) -> None:
        if _has_nested(hf_dict, self.hf_paths[0]):
            value = _get_nested(hf_dict, self.hf_paths[0])
        else:
            value = self._hf_default_fn(hf_dict)
        set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], value)


class OptionalConfigConverter(ConfigConverter):
    """Emit/import only when the value differs from a sentinel (default ``None``).

    Useful for fields that round-trip cleanly only when present (e.g. ``window_size``).
    """

    def __init__(self, fast_llm_path: tuple[str, ...], hf_path: tuple[str, ...], sentinel: typing.Any = None):
        self.fast_llm_paths = (fast_llm_path,)
        self.hf_paths = (hf_path,)
        self._sentinel = sentinel

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        value = _get_attr_path(fast_llm_config, self.fast_llm_paths[0])
        if value != self._sentinel:
            set_nested_dict_value(hf_out, self.hf_paths[0], value)

    def import_to(self, hf_dict: dict, fast_llm_out: dict, parent_context: dict | None = None) -> None:
        if _has_nested(hf_dict, self.hf_paths[0]):
            value = _get_nested(hf_dict, self.hf_paths[0])
            if value != self._sentinel:
                set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], value)


class IgnoredConfigConverter(ConfigConverter):
    """Declares Fast-LLM architecture fields as intentionally not converted by this format.

    Use when the HF format has no representation for the field and the Fast-LLM default round-trips correctly.
    Acts as a no-op on both directions while satisfying the architecture-coverage check.
    """

    def __init__(self, *fast_llm_paths: tuple[str, ...]):
        self.fast_llm_paths = fast_llm_paths
        self.hf_paths = ()

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        return

    def import_to(self, hf_dict: dict, fast_llm_out: dict, parent_context: dict | None = None) -> None:
        return


class CustomConfigConverter(ConfigConverter):
    """Escape hatch for cross-field transforms (e.g., rotary, where one HF blob ↔ several Fast-LLM fields).

    The export/import callables receive the section's full config and return/produce arbitrary mappings within
    the declared paths. Both ``fast_llm_paths`` and ``hf_paths`` are still declared so the coverage check works.
    """

    def __init__(
        self,
        fast_llm_paths: tuple[tuple[str, ...], ...],
        hf_paths: tuple[tuple[str, ...], ...],
        export_fn: typing.Callable[[Config], dict],
        import_fn: typing.Callable[[dict, dict | None], dict],
    ):
        self.fast_llm_paths = fast_llm_paths
        self.hf_paths = hf_paths
        self._export_fn = export_fn
        self._import_fn = import_fn

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        produced = self._export_fn(fast_llm_config)
        for path, value in produced.items():
            set_nested_dict_value(hf_out, path, value)

    def import_to(self, hf_dict: dict, fast_llm_out: dict, parent_context: dict | None = None) -> None:
        produced = self._import_fn(hf_dict, parent_context)
        for path, value in produced.items():
            set_nested_dict_value(fast_llm_out, path, value)


class NestedConfigConverter(ConfigConverter):
    """Recurse into a fixed-typed sub-config field via another section converter class.

    Exists for Fast-LLM-side modularity: lets a parent converter delegate handling of a sub-config to its own
    converter class. The HF side is assumed flat — the sub-converter's output is merged into the parent's HF dict.
    For non-flat HF formats, use ``CustomConfigConverter``.
    """

    def __init__(
        self,
        fast_llm_path: tuple[str, ...],
        converter_class: "type[ConfigSectionConverter]",
    ):
        self.fast_llm_paths = (fast_llm_path,)
        self._converter_class = converter_class

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        sub_config = _get_attr_path(fast_llm_config, self.fast_llm_paths[0])
        sub_hf = self._converter_class.export_config(sub_config)
        for key, value in sub_hf.items():
            if key in hf_out:
                Assert.eq(hf_out[key], value)
            else:
                hf_out[key] = value

    def import_to(self, hf_dict: dict, fast_llm_out: dict, parent_context: dict | None = None) -> None:
        sub_fast_llm = self._converter_class.import_config(hf_dict)
        set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], sub_fast_llm)


class DispatchConfigConverter(ConfigConverter):
    """Polymorphic sub-config dispatch.

    The Fast-LLM field's runtime type selects the section converter; the HF format selects via a ``type`` discriminator.
    Both registries (Fast-LLM type → converter class, HF discriminator → converter class) must agree at runtime.
    """

    def __init__(
        self,
        fast_llm_path: tuple[str, ...],
        hf_path: tuple[str, ...] | None,
        registry: "dict[type[Config], type[ConfigSectionConverter]]",
        hf_discriminator_key: str = "type",
    ):
        self.fast_llm_paths = (fast_llm_path,)
        self.hf_paths = (hf_path,) if hf_path is not None else ()
        self._registry = registry
        self._hf_discriminator_key = hf_discriminator_key
        self._hf_to_class = {cls.hf_type_name: cls for cls in registry.values() if cls.hf_type_name is not None}

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        sub_config = _get_attr_path(fast_llm_config, self.fast_llm_paths[0])
        converter_class = self._registry.get(type(sub_config))
        if converter_class is None:
            raise NotImplementedError(
                f"No converter registered for {type(sub_config).__name__} at {'.'.join(self.fast_llm_paths[0])}"
            )
        sub_hf = converter_class.export_config(sub_config)
        if converter_class.hf_type_name is not None:
            sub_hf = {self._hf_discriminator_key: converter_class.hf_type_name, **sub_hf}
        if self.hf_paths:
            set_nested_dict_value(hf_out, self.hf_paths[0], sub_hf)
        else:
            for key, value in sub_hf.items():
                hf_out[key] = value

    def import_to(self, hf_dict: dict, fast_llm_out: dict, parent_context: dict | None = None) -> None:
        sub_hf = _get_nested(hf_dict, self.hf_paths[0]) if self.hf_paths else hf_dict
        type_name = sub_hf.get(self._hf_discriminator_key)
        converter_class = self._hf_to_class.get(type_name)
        if converter_class is None:
            raise NotImplementedError(
                f"No converter registered for HF discriminator {type_name!r} at " f"{'.'.join(self.fast_llm_paths[0])}"
            )
        sub_fast_llm = converter_class.import_config(sub_hf)
        set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], sub_fast_llm)


# ============================================================
# Section converter — converts one Fast-LLM config class
# ============================================================


class ConfigSectionConverter(abc.ABC):
    """Base class for converting one Fast-LLM ``Config`` class ↔ one HF dict subtree.

    Subclasses declare the conversion via ``_create_config_converters`` (config side) and
    ``_create_weight_converters`` (weight side; receives the live config).

    Subclasses that participate in :class:`DispatchConfigConverter` set ``hf_type_name`` to the discriminator value
    used by the HF format (e.g. ``"attention"``, ``"mamba"``).
    """

    fast_llm_config_class: typing.ClassVar[type[Config]]
    hf_type_name: typing.ClassVar[str | None] = None

    @classmethod
    @abc.abstractmethod
    def _create_config_converters(cls) -> dict[str, ConfigConverter]:
        """Return declarations keyed by stable string name. Subclasses override entries by re-declaring the key."""

    @classmethod
    def _create_weight_converters(
        cls, config: Config, fast_llm_prefix: str, hf_prefix: str
    ) -> dict[str, "WeightConverter"]:
        """Return weight converters keyed by stable string name. Default is empty (no weights at this level)."""
        return {}

    @classmethod
    def export_config(cls, config: Config) -> dict:
        """Convert a Fast-LLM config object to an HF config dict via this section's declarations."""
        declarations = cls._create_config_converters()
        cls._check_architecture_coverage(config, declarations)
        out: dict = {}
        for converter in declarations.values():
            converter.export_to(config, out)
        return out

    @classmethod
    def import_config(cls, hf_dict: dict) -> dict:
        """Convert an HF config dict to a Fast-LLM config dict via this section's declarations."""
        out: dict = {}
        for converter in cls._create_config_converters().values():
            converter.import_to(hf_dict, out)
        return out

    @classmethod
    def _check_architecture_coverage(cls, config: Config, declarations: dict[str, ConfigConverter]) -> None:
        """Raise if any architecture-hint field on the section's declared config class is not consumed.

        Coverage is structural (based on field hints), not value-based: every architecture field must be
        explicitly accounted for, even if it currently holds its Fast-LLM default. Sub-config fields are
        consumed by ``NestedConfigConverter``/``DispatchConfigConverter``, which delegate the deeper coverage
        check to the nested section's own converter.

        The check only runs when ``type(config)`` exactly matches ``cls.fast_llm_config_class`` — when the
        config is a strict subclass (e.g. ``MoEMLPConfig`` fed via ``super().export_config()`` from a yet-to-be-
        migrated ``MixtralMLPConverter``), the subclass converter is responsible for declaring the additional
        fields and running its own check.
        """
        declared_class = getattr(cls, "fast_llm_config_class", None)
        if declared_class is None or type(config) is not declared_class:
            return
        consumed: set[str] = set()
        for converter in declarations.values():
            consumed |= converter.consumed_fast_llm_fields
        missing: list[str] = []
        for name, field in type(config).fields():
            if field._field_type != dataclasses._FIELD:
                continue
            if not field.init:
                continue
            if field.hint != FieldHint.architecture:
                continue
            if name in consumed:
                continue
            missing.append(name)
        if missing:
            raise ValueError(
                f"{cls.__name__}: architecture-hint fields on {type(config).__name__} "
                f"have no converter declaration: {missing}"
            )


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
