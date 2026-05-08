import abc
import dataclasses
import logging
import pathlib
import typing

import torch

from fast_llm import __version__
from fast_llm.config import Config, FieldHint, get_nested_dict_value, set_nested_dict_value
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.checkpoint.config import CheckpointLoadMetadataConfig
from fast_llm.engine.checkpoint.state_dict import StateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import CheckpointMetadata, FastLLMModelConfig
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


def _get_attr_path(config: Config, path: tuple[str, ...]) -> typing.Any:
    cur = config
    for name in path:
        cur = getattr(cur, name)
    return cur


def _collect_architecture_paths(config: Config) -> list[tuple[str, ...]]:
    """Walk ``config`` and return every architecture-hint field path reachable from it.

    Descends into any field whose runtime value is a :class:`Config`, a ``dict[str, Config]``
    (paths are extended with the entry's string key), or a ``list[Config]`` (paths are extended
    with the entry's index as a string), so the path list reflects the actual instance.
    """
    paths: list[tuple[str, ...]] = []

    def descend(value: typing.Any, prefix: tuple[str, ...]) -> None:
        if isinstance(value, Config):
            walk(value, prefix)
        elif isinstance(value, dict):
            for key, sub in value.items():
                descend(sub, prefix + (str(key),))
        elif isinstance(value, (list, tuple)):
            for index, sub in enumerate(value):
                descend(sub, prefix + (str(index),))

    def walk(node: Config, prefix: tuple[str, ...]) -> None:
        for name, field in type(node).fields():
            if field._field_type != dataclasses._FIELD or not field.init:
                continue
            full = prefix + (name,)
            if field.hint == FieldHint.architecture:
                paths.append(full)
            descend(getattr(node, name), full)

    walk(config, ())
    return paths


# ============================================================
# Config conversion primitives (declarative)
# ============================================================


class ConfigConverter(abc.ABC):
    """A declarative description of how one or more Fast-LLM config fields map to one or more HF config keys.

    Each primitive owns a set of ``fast_llm_paths`` (tuples of attribute names rooted at the section's config) and
    ``hf_paths`` (tuples of dict keys rooted at the section's HF subdict). The walker calls ``export_to`` to produce
    HF entries from a Fast-LLM config object, and ``import_to`` to produce a Fast-LLM config dict from an HF dict.

    ``recurses`` controls how :meth:`ConfigSectionConverter._check_architecture_coverage` interprets the paths:

    * ``recurses = False`` (default) — paths are exact-match leaves. Every architecture-hint field at every depth
      under the section's config class must be exactly listed by some declaration.
    * ``recurses = True`` — paths are recursive prefixes covering the entire subtree. Used by primitives that
      delegate to a sub-converter that runs its own coverage check (Nested/Dispatch/TypedDictContainer), by
      :class:`IgnoredConfigConverter` (the format intentionally does not represent the subtree), and by
      :class:`CustomConfigConverter` when its author opts in (escape hatch for cases like rotary that don't
      decompose into per-leaf renames).
    """

    fast_llm_paths: tuple[tuple[str, ...], ...] = ()
    hf_paths: tuple[tuple[str, ...], ...] = ()
    recurses: typing.ClassVar[bool] = False

    @abc.abstractmethod
    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None: ...

    @abc.abstractmethod
    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None: ...


class RenameConfigConverter(ConfigConverter):
    """One-to-one rename between a Fast-LLM attribute path and an HF dict path."""

    def __init__(self, fast_llm_path: tuple[str, ...], hf_path: tuple[str, ...]):
        self.fast_llm_paths = (fast_llm_path,)
        self.hf_paths = (hf_path,)

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        value = _get_attr_path(fast_llm_config, self.fast_llm_paths[0])
        set_nested_dict_value(hf_out, self.hf_paths[0], value)

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        value = get_nested_dict_value(hf_dict, self.hf_paths[0])
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

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        try:
            actual = get_nested_dict_value(hf_dict, self.hf_paths[0])
        except KeyError:
            return
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

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
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

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        try:
            value = get_nested_dict_value(hf_dict, self.hf_paths[0])
        except KeyError:
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

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        try:
            value = get_nested_dict_value(hf_dict, self.hf_paths[0])
        except KeyError:
            return
        if value != self._sentinel:
            set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], value)


class IgnoredConfigConverter(ConfigConverter):
    """Declares Fast-LLM architecture fields as intentionally not converted by this format.

    Use when the HF format has no representation for the field and the Fast-LLM default round-trips correctly.
    Acts as a no-op on both directions while satisfying the architecture-coverage check. The claim covers the
    entire subtree under each listed path: deeper architecture fields are also implicitly ignored, on the
    assumption that a format which does not represent the parent likewise does not represent its children.
    """

    recurses: typing.ClassVar[bool] = True

    def __init__(self, *fast_llm_paths: tuple[str, ...]):
        self.fast_llm_paths = fast_llm_paths
        self.hf_paths = ()

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        return

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        return


class CustomConfigConverter(ConfigConverter):
    """Escape hatch for cross-field transforms (e.g., rotary, where one HF blob ↔ several Fast-LLM fields).

    ``fast_llm_paths`` is declared so the coverage check sees the fields as consumed. The HF side is intentionally
    not declared — there is no symmetric HF-side coverage check yet, so an ``hf_paths`` argument would be cosmetic.
    Cross-field validators that produce nothing on the HF side belong on :py:meth:`ConfigSectionConverter._validate_export`
    instead; this primitive is for shape-changing transforms.

    Pass ``recurses=True`` when the converter genuinely owns a sub-config subtree (e.g. rotary, per-layer biases) —
    the listed paths then act as recursive prefixes and the architecture-coverage check stops at them. The author
    is trusted to handle every architecture field of the claimed subtree; prefer Nested/Dispatch primitives when
    the subtree decomposes cleanly.
    """

    def __init__(
        self,
        fast_llm_paths: tuple[tuple[str, ...], ...],
        export_fn: typing.Callable[[Config], dict],
        import_fn: typing.Callable[[dict], dict],
        recurses: bool = False,
    ):
        self.fast_llm_paths = fast_llm_paths
        self._export_fn = export_fn
        self._import_fn = import_fn
        self.recurses = recurses

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        produced = self._export_fn(fast_llm_config)
        for path, value in produced.items():
            set_nested_dict_value(hf_out, path, value)

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        produced = self._import_fn(hf_dict)
        for path, value in produced.items():
            set_nested_dict_value(fast_llm_out, path, value)


class ImportOnlyConfigConverter(ConfigConverter):
    """One-way mapping that runs only on import; emits nothing on export.

    Used when the HF format derives a Fast-LLM field from sibling fields (e.g. ``head_size`` from
    ``hidden_size // num_attention_heads`` in Qwen2) or implies a value the Fast-LLM side stores
    explicitly (e.g. Qwen2's hardcoded Q/K/V biases, Pixtral's mirrored ``patch_size`` ↔ ``patch_width``).
    On export the field is redundant and validated through ``_validate_export``; on import the
    ``import_fn`` produces the Fast-LLM dict entries. The fast_llm_paths still register as consumed
    for the architecture-coverage check.

    Pass ``recurses=True`` when the converter populates a sub-config subtree (e.g. Qwen2's per-layer
    biases that target ``query_layer``/``key_layer``/...). Same trade-off as
    :class:`CustomConfigConverter`: the listed paths become recursive prefixes and the framework no
    longer enforces leaf coverage under them.
    """

    def __init__(
        self,
        fast_llm_paths: tuple[tuple[str, ...], ...],
        import_fn: typing.Callable[[dict], dict],
        recurses: bool = False,
    ):
        self.fast_llm_paths = fast_llm_paths
        self._import_fn = import_fn
        self.recurses = recurses

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        return

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        produced = self._import_fn(hf_dict)
        for path, value in produced.items():
            set_nested_dict_value(fast_llm_out, path, value)


class NestedConfigConverter(ConfigConverter):
    """Recurse into a fixed-typed sub-config field via another section converter class.

    Default (``hf_path=None``): the HF side is flat-merged — the sub-converter's output becomes top-level keys
    of the parent's HF dict, asserting any pre-existing keys agree.

    With ``hf_path`` set: the sub-converter's output is placed under that nested key. Use this for HF formats
    that mirror Fast-LLM's modular layout (e.g. Apriel2's ``"decoder": {...}`` and ``"head": {...}`` blocks).
    """

    recurses: typing.ClassVar[bool] = True

    def __init__(
        self,
        fast_llm_path: tuple[str, ...],
        converter_class: "type[ConfigSectionConverter]",
        hf_path: tuple[str, ...] | None = None,
    ):
        self.fast_llm_paths = (fast_llm_path,)
        self._converter_class = converter_class
        self._hf_path = hf_path

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        sub_config = _get_attr_path(fast_llm_config, self.fast_llm_paths[0])
        sub_hf = self._converter_class.export_config(sub_config)
        if self._hf_path is None:
            for key, value in sub_hf.items():
                if key in hf_out:
                    Assert.eq(hf_out[key], value)
                else:
                    hf_out[key] = value
        else:
            set_nested_dict_value(hf_out, self._hf_path, sub_hf)

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        sub_hf = get_nested_dict_value(hf_dict, self._hf_path) if self._hf_path is not None else hf_dict
        sub_fast_llm = self._converter_class.import_config(sub_hf)
        set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], sub_fast_llm)


class DispatchConfigConverter(ConfigConverter):
    """Polymorphic sub-config dispatch.

    The Fast-LLM field's runtime type selects the section converter; the HF format selects via a ``type`` discriminator.
    Both registries (Fast-LLM type → converter class, HF discriminator → converter class) must agree at runtime.
    """

    recurses: typing.ClassVar[bool] = True

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

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        sub_hf = get_nested_dict_value(hf_dict, self.hf_paths[0]) if self.hf_paths else hf_dict
        type_name = sub_hf.get(self._hf_discriminator_key)
        converter_class = self._hf_to_class.get(type_name)
        if converter_class is None:
            raise NotImplementedError(
                f"No converter registered for HF discriminator {type_name!r} at " f"{'.'.join(self.fast_llm_paths[0])}"
            )
        sub_fast_llm = converter_class.import_config(sub_hf)
        # Inject the Fast-LLM dynamic-type discriminator so the parent's `from_dict` dispatches to the
        # correct subclass. Reads from the registered Config class rather than the HF discriminator so
        # mismatched Fast-LLM/HF type names work too.
        fast_llm_type = getattr(converter_class.fast_llm_config_class, "dynamic_type_name", None)
        if fast_llm_type is not None:
            sub_fast_llm = {"type": fast_llm_type, **sub_fast_llm}
        set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], sub_fast_llm)


class TypedDictContainerConfigConverter(ConfigConverter):
    """Maps a Fast-LLM ``dict[str, Config]`` field to an HF ``dict[str, dict]`` where each entry is round-tripped
    through a per-class section converter selected via the entry's runtime type (export) or HF discriminator (import).

    Each entry's HF subdict carries a discriminator key (``"type"`` by default) populated from the converter's
    ``hf_type_name``. For homogeneous dicts, register a single class with ``hf_type_name = None``; the discriminator
    is then omitted on export and ignored on import.
    """

    recurses: typing.ClassVar[bool] = True

    def __init__(
        self,
        fast_llm_path: tuple[str, ...],
        hf_path: tuple[str, ...],
        registry: "dict[type[Config], type[ConfigSectionConverter]]",
        hf_discriminator_key: str = "type",
    ):
        self.fast_llm_paths = (fast_llm_path,)
        self.hf_paths = (hf_path,)
        self._registry = registry
        self._hf_discriminator_key = hf_discriminator_key
        self._hf_to_class = {cls.hf_type_name: cls for cls in registry.values() if cls.hf_type_name is not None}
        self._homogeneous = len(registry) == 1 and next(iter(registry.values())).hf_type_name is None
        if self._homogeneous:
            self._homogeneous_class = next(iter(registry.values()))

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        sub_dict = _get_attr_path(fast_llm_config, self.fast_llm_paths[0])
        out: dict = {}
        for name, sub_config in sub_dict.items():
            if self._homogeneous:
                converter_class = self._homogeneous_class
            else:
                converter_class = self._registry.get(type(sub_config))
                if converter_class is None:
                    raise NotImplementedError(
                        f"No converter registered for {type(sub_config).__name__} at "
                        f"{'.'.join(self.fast_llm_paths[0])}[{name!r}]"
                    )
            sub_hf = converter_class.export_config(sub_config)
            if converter_class.hf_type_name is not None:
                sub_hf = {self._hf_discriminator_key: converter_class.hf_type_name, **sub_hf}
            out[name] = sub_hf
        set_nested_dict_value(hf_out, self.hf_paths[0], out)

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        sub_hf_dict = get_nested_dict_value(hf_dict, self.hf_paths[0])
        out: dict = {}
        for name, sub_hf in sub_hf_dict.items():
            if self._homogeneous:
                converter_class = self._homogeneous_class
            else:
                type_name = sub_hf.get(self._hf_discriminator_key)
                converter_class = self._hf_to_class.get(type_name)
                if converter_class is None:
                    raise NotImplementedError(
                        f"No converter registered for HF discriminator {type_name!r} at "
                        f"{'.'.join(self.hf_paths[0])}[{name!r}]"
                    )
            sub_fast_llm = converter_class.import_config(sub_hf)
            fast_llm_type = getattr(converter_class.fast_llm_config_class, "dynamic_type_name", None)
            if fast_llm_type is not None:
                sub_fast_llm = {"type": fast_llm_type, **sub_fast_llm}
            out[name] = sub_fast_llm
        set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], out)


# ============================================================
# Section converter — converts one Fast-LLM config class
# ============================================================


class ConfigSectionConverter(abc.ABC):
    """Base class for converting one Fast-LLM ``Config`` class ↔ one HF dict subtree.

    Subclasses declare the conversion via ``_create_config_converters``. Format-specific cross-field
    invariants go on the ``_validate_export`` hook. The weight side is still imperative (per-converter
    ``get_converters`` classmethods on the concrete subclasses); a declarative weight-side primitive will be
    added when the weight-converter migration lands.

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
    def _validate_export(cls, config: Config) -> None:
        """Hook for format-specific export-time validation. Default no-op.

        Runs after the architecture-coverage check and before any declaration emits. Use this for cross-field
        invariants the format imposes on the Fast-LLM config (e.g. per-layer biases must match a parent flag,
        certain sub-configs must be at their default). Subclasses override; super-calls are not required when
        the rule is fully replaced (e.g. Qwen2 vs Llama attention biases).
        """
        return

    @classmethod
    def export_config(cls, config: Config) -> dict:
        """Convert a Fast-LLM config object to an HF config dict via this section's declarations."""
        declarations = cls._create_config_converters()
        cls._check_architecture_coverage(config, declarations)
        cls._validate_export(config)
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
        """Raise if any architecture-hint field reachable from the section's config (recursively) is not consumed.

        Coverage is structural (based on field hints), not value-based: every architecture field at every depth
        must be accounted for, even when it currently holds its Fast-LLM default. The walker descends into any
        field whose runtime value is a :class:`Config`, collecting an architecture-leaf list, and matches each
        leaf against the section's declarations:

        * Recursive declarations (``recurses = True`` — Nested/Dispatch/TypedDictContainer/Ignored, plus Custom
          when its author opts in) cover the entire subtree under each listed prefix. Nested/Dispatch/TypedDict
          delegate to a sub-converter that runs its own coverage check; Ignored and recursive Custom assume the
          author has handled the subtree.
        * Non-recursive declarations (Rename, ConstantImport/Export, Default, Optional, ImportOnly, Custom by
          default) must list every architecture leaf they consume by exact path.

        The check only runs when ``type(config)`` exactly matches ``cls.fast_llm_config_class`` — when the
        config is a strict subclass (e.g. ``MoEMLPConfig`` fed through ``LlamaMLPConverter`` declarations
        before the dispatching ``MixtralMLPConverter`` overrides ``fast_llm_config_class``), the subclass
        converter is responsible for declaring the additional fields and running its own check.
        """
        if type(config) is not cls.fast_llm_config_class:
            return
        explicit_paths: set[tuple[str, ...]] = set()
        recursive_prefixes: list[tuple[str, ...]] = []
        for converter in declarations.values():
            if converter.recurses:
                recursive_prefixes.extend(converter.fast_llm_paths)
            else:
                explicit_paths.update(converter.fast_llm_paths)
        missing: list[tuple[str, ...]] = []
        for path in _collect_architecture_paths(config):
            if path in explicit_paths:
                continue
            if any(len(prefix) <= len(path) and path[: len(prefix)] == prefix for prefix in recursive_prefixes):
                continue
            missing.append(path)
        if missing:
            # If every missing path shares a top-level prefix that IS claimed (just non-recursively),
            # the contributor likely needs a recursive primitive there — surface that as a hint.
            shared_prefixes = {path[:1] for path in missing if path[:1] in explicit_paths}
            hint = ""
            if shared_prefixes:
                names = sorted(prefix[0] for prefix in shared_prefixes)
                hint = (
                    f" (declarations for {names} claim the parent path non-recursively; "
                    f"either list every architecture sub-field or switch to Nested/Dispatch — "
                    f"or pass ``recurses=True`` to a Custom/ImportOnly converter when claiming the whole subtree)"
                )
            raise ValueError(
                f"{cls.__name__}: architecture-hint fields on {type(config).__name__} "
                f"have no converter declaration: {[ '.'.join(p) for p in missing ]}{hint}"
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
