import abc
import dataclasses
import functools
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
    current = config
    for name in path:
        current = getattr(current, name)
    return current


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


def _safe_set_nested_dict_value(out: dict, path: tuple[str, ...], value: typing.Any) -> None:
    """Set ``out[path] = value``, but raise via :class:`Assert.eq` if the path already holds a different value.

    Converters share a single output dict during ``export_config``/``import_config``. Multiple converters may
    legitimately write the same path (e.g. Llama's decoder block and head both produce ``rms_norm_eps``
    via flat-merge), but only with the same value — a divergent value means a cross-section invariant has
    been mis-modelled and the silent override would drop one side of the mismatch.
    """
    try:
        existing = get_nested_dict_value(out, path)
    except KeyError:
        set_nested_dict_value(out, path, value)
        return
    Assert.eq(existing, value)


# ============================================================
# Config conversion primitives (declarative)
# ============================================================


class ConfigConverter(abc.ABC):
    """A declarative description of how one or more Fast-LLM config fields map to one or more HF config keys.

    Each primitive owns a set of ``fast_llm_paths`` (tuples of attribute names rooted at the section's config) and
    ``hf_paths`` (tuples of dict keys rooted at the section's HF subdict). The walker calls ``export_to`` to produce
    HF entries from a Fast-LLM config object, and ``import_to`` to produce a Fast-LLM config dict from an HF dict.

    ``fast_llm_recurses`` controls how :meth:`ConfigSectionConverter.check_architecture_coverage` interprets
    ``fast_llm_paths``:

    * ``fast_llm_recurses = False`` (default) — paths are exact-match leaves. Every architecture-hint field at
      every depth under the section's config class must be exactly listed by some declaration.
    * ``fast_llm_recurses = True`` — paths are recursive prefixes covering the entire subtree. Used by primitives
      that delegate to a sub-converter that runs its own coverage check (Nested/Dispatch/TypedDictContainer), by
      :class:`IgnoredConfigConverter` (the format intentionally does not represent the subtree), and by
      :class:`CustomConfigConverter` when its author opts in (escape hatch for cases like rotary that don't
      decompose into per-leaf renames).

    The HF-side counterpart :meth:`ConfigSectionConverter.check_hf_coverage` always treats every entry in
    ``hf_paths`` as a recursive prefix, regardless of ``fast_llm_recurses``. HF configs frequently carry leaf
    values (numbers, strings) that the walker has no further depth to descend into, and sub-dicts that round-trip
    through transformers' ``save_pretrained`` may sprout new keys that don't correspond to Fast-LLM fields —
    treating any claimed HF path as a recursive prefix is the simplest contract that survives both. Use
    :class:`IgnoredConfigConverter` with ``hf_paths=...`` to claim HF subtrees the format doesn't surface to
    Fast-LLM (transformers' generated metadata, defaulted-but-unused inference fields, ...).
    """

    fast_llm_paths: tuple[tuple[str, ...], ...] = ()
    hf_paths: tuple[tuple[str, ...], ...] = ()
    fast_llm_recurses: typing.ClassVar[bool] = False

    @abc.abstractmethod
    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None: ...

    @abc.abstractmethod
    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None: ...

    def _consumed_hf_paths(self) -> frozenset[tuple[str, ...]]:
        """HF paths this declaration claims for the section's coverage check.

        Default: every non-empty entry in ``hf_paths``. Override when the claim set depends on
        a sub-converter registry, a nested-path prefix, or a discriminator key (Nested, Dispatch,
        TypedDictContainer, ListDispatch).
        """
        return frozenset(path for path in self.hf_paths if path)


class RenameConfigConverter(ConfigConverter):
    """One-to-one rename between a Fast-LLM attribute path and an HF dict path."""

    def __init__(self, fast_llm_path: tuple[str, ...], hf_path: tuple[str, ...]):
        self.fast_llm_paths = (fast_llm_path,)
        self.hf_paths = (hf_path,)

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        value = _get_attr_path(fast_llm_config, self.fast_llm_paths[0])
        _safe_set_nested_dict_value(hf_out, self.hf_paths[0], value)

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        value = get_nested_dict_value(hf_dict, self.hf_paths[0])
        _safe_set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], value)


class ConstantExportConfigConverter(ConfigConverter):
    """Write a constant to the HF dict on export. On import, assert that the HF dict has this constant value.

    Used when a HF format requires a key whose value Fast-LLM doesn't store (or always pins to a constant).
    """

    def __init__(self, hf_path: tuple[str, ...], value: typing.Any):
        self.hf_paths = (hf_path,)
        self._value = value

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        _safe_set_nested_dict_value(hf_out, self.hf_paths[0], self._value)

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
        _safe_set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], self._value)


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
        _safe_set_nested_dict_value(hf_out, self.hf_paths[0], value)

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        try:
            value = get_nested_dict_value(hf_dict, self.hf_paths[0])
        except KeyError:
            value = self._hf_default_fn(hf_dict)
        _safe_set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], value)


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
            _safe_set_nested_dict_value(hf_out, self.hf_paths[0], value)

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        try:
            value = get_nested_dict_value(hf_dict, self.hf_paths[0])
        except KeyError:
            return
        if value != self._sentinel:
            _safe_set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], value)


class IgnoredConfigConverter(ConfigConverter):
    """Declares Fast-LLM architecture fields and/or HF dict keys as intentionally not converted by this format.

    Use ``fast_llm_paths`` (positional) when Fast-LLM has architecture fields with no HF representation; the
    Fast-LLM default round-trips. Use ``hf_paths`` (kw-only) when the HF format carries fields Fast-LLM does
    not consume (generation-only toggles like Mixtral's ``router_aux_loss_coef``, Qwen2's ``sliding_window``).
    Both kinds of claim are no-ops at conversion time and serve only the per-side coverage checks. The claim
    covers the entire subtree under each listed path on the side it applies to.
    """

    fast_llm_recurses: typing.ClassVar[bool] = True

    def __init__(self, *fast_llm_paths: tuple[str, ...], hf_paths: tuple[tuple[str, ...], ...] = ()):
        self.fast_llm_paths = fast_llm_paths
        self.hf_paths = hf_paths

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        pass

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        pass


class CustomConfigConverter(ConfigConverter):
    """Escape hatch for cross-field transforms (e.g., rotary, where one HF blob ↔ several Fast-LLM fields).

    ``fast_llm_paths`` and ``hf_paths`` are declared so the per-side coverage checks see the fields as consumed.
    Cross-field validators that produce nothing on the HF side belong on :py:meth:`ConfigSectionConverter._validate_export`
    instead; this primitive is for shape-changing transforms.

    Pass ``fast_llm_recurses=True`` when the converter genuinely owns a sub-config subtree (e.g. rotary, per-layer
    biases) — the listed ``fast_llm_paths`` then act as recursive prefixes and the architecture-coverage check stops
    at them. The author is trusted to handle every architecture field of the claimed subtree; prefer
    Nested/Dispatch primitives when the subtree decomposes cleanly. The HF-side coverage walker treats every entry
    of ``hf_paths`` as a recursive prefix regardless (see :class:`ConfigConverter`).
    """

    def __init__(
        self,
        fast_llm_paths: tuple[tuple[str, ...], ...],
        export_fn: typing.Callable[[Config], dict],
        import_fn: typing.Callable[[dict], dict],
        hf_paths: tuple[tuple[str, ...], ...] = (),
        fast_llm_recurses: bool = False,
    ):
        self.fast_llm_paths = fast_llm_paths
        self.hf_paths = hf_paths
        self._export_fn = export_fn
        self._import_fn = import_fn
        self.fast_llm_recurses = fast_llm_recurses

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        produced = self._export_fn(fast_llm_config)
        for path, value in produced.items():
            _safe_set_nested_dict_value(hf_out, path, value)

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        produced = self._import_fn(hf_dict)
        for path, value in produced.items():
            _safe_set_nested_dict_value(fast_llm_out, path, value)


class ImportOnlyConfigConverter(ConfigConverter):
    """One-way mapping that runs only on import; emits nothing on export.

    Used when the HF format derives a Fast-LLM field from sibling fields (e.g. ``head_size`` from
    ``hidden_size // num_attention_heads`` in Qwen2) or implies a value the Fast-LLM side stores
    explicitly (e.g. Qwen2's hardcoded Q/K/V biases, Pixtral's mirrored ``patch_size`` ↔ ``patch_width``).
    On export the field is redundant and validated through ``_validate_export``; on import the
    ``import_fn`` produces the Fast-LLM dict entries. The fast_llm_paths register as consumed for the
    architecture-coverage check; ``hf_paths`` register as consumed for the HF-side check.

    Pass ``fast_llm_recurses=True`` when the converter populates a sub-config subtree (e.g. Qwen2's per-layer
    biases that target ``query_layer``/``key_layer``/...). Same trade-off as :class:`CustomConfigConverter`: the
    listed ``fast_llm_paths`` become recursive prefixes and the framework no longer enforces leaf coverage under
    them. The HF-side coverage walker treats every entry of ``hf_paths`` as a recursive prefix regardless.
    """

    def __init__(
        self,
        fast_llm_paths: tuple[tuple[str, ...], ...],
        import_fn: typing.Callable[[dict], dict],
        hf_paths: tuple[tuple[str, ...], ...] = (),
        fast_llm_recurses: bool = False,
    ):
        self.fast_llm_paths = fast_llm_paths
        self.hf_paths = hf_paths
        self._import_fn = import_fn
        self.fast_llm_recurses = fast_llm_recurses

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        pass

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        produced = self._import_fn(hf_dict)
        for path, value in produced.items():
            _safe_set_nested_dict_value(fast_llm_out, path, value)


class NestedConfigConverter(ConfigConverter):
    """Recurse into a fixed-typed sub-config field via another section converter class.

    Default (``hf_path=None``): the HF side is flat-merged — the sub-converter's output becomes top-level keys
    of the parent's HF dict, asserting any pre-existing keys agree.

    With ``hf_path`` set: the sub-converter's output is placed under that nested key. Use this for HF formats
    that mirror Fast-LLM's modular layout (e.g. Apriel2's ``"decoder": {...}`` and ``"head": {...}`` blocks).

    When the target ``converter_class`` declares ``hf_type_name``, an HF discriminator (``"type"`` by default)
    is auto-injected on export and validated/stripped on import — matching DispatchConfigConverter's behavior
    for homogeneous (single-target) cases.
    """

    fast_llm_recurses: typing.ClassVar[bool] = True

    def __init__(
        self,
        fast_llm_path: tuple[str, ...],
        converter_class: "type[ConfigSectionConverter]",
        hf_path: tuple[str, ...] | None = None,
        hf_discriminator_key: str = "type",
    ):
        self.fast_llm_paths = (fast_llm_path,)
        self._converter_class = converter_class
        self._hf_path = hf_path
        self._hf_discriminator_key = hf_discriminator_key

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        sub_config = _get_attr_path(fast_llm_config, self.fast_llm_paths[0])
        sub_hf = self._converter_class.export_config(sub_config)
        if self._converter_class.hf_type_name is not None:
            sub_hf = {self._hf_discriminator_key: self._converter_class.hf_type_name, **sub_hf}
        if self._hf_path is None:
            for key, value in sub_hf.items():
                _safe_set_nested_dict_value(hf_out, (key,), value)
        else:
            _safe_set_nested_dict_value(hf_out, self._hf_path, sub_hf)

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        sub_hf = get_nested_dict_value(hf_dict, self._hf_path) if self._hf_path is not None else hf_dict
        if self._converter_class.hf_type_name is not None and self._hf_discriminator_key in sub_hf:
            Assert.eq(sub_hf[self._hf_discriminator_key], self._converter_class.hf_type_name)
            sub_hf = {key: value for key, value in sub_hf.items() if key != self._hf_discriminator_key}
        sub_fast_llm = self._converter_class.import_config(sub_hf)
        _safe_set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], sub_fast_llm)

    def _consumed_hf_paths(self) -> frozenset[tuple[str, ...]]:
        sub_class = self._converter_class
        if self._hf_path is None:
            # Flat-merge: sub-converter shares the parent's HF namespace.
            paths = set(sub_class._consumed_hf_paths())
            if sub_class.hf_type_name is not None:
                paths.add((self._hf_discriminator_key,))
        else:
            # Nested: prepend hf_path so the walker descends into the subdict and flags unknown keys.
            prefix = self._hf_path
            paths = {prefix + sub_path for sub_path in sub_class._consumed_hf_paths()}
            if sub_class.hf_type_name is not None:
                paths.add(prefix + (self._hf_discriminator_key,))
        return frozenset(paths)


class DispatchConfigConverter(ConfigConverter):
    """Polymorphic sub-config dispatch.

    The Fast-LLM field's runtime type selects the section converter; the HF format selects via a ``type`` discriminator.
    Both registries (Fast-LLM type → converter class, HF discriminator → converter class) must agree at runtime.
    """

    fast_llm_recurses: typing.ClassVar[bool] = True

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
            _safe_set_nested_dict_value(hf_out, self.hf_paths[0], sub_hf)
        else:
            for key, value in sub_hf.items():
                _safe_set_nested_dict_value(hf_out, (key,), value)

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        sub_hf = get_nested_dict_value(hf_dict, self.hf_paths[0]) if self.hf_paths else hf_dict
        type_name = sub_hf.get(self._hf_discriminator_key)
        converter_class = self._hf_to_class.get(type_name)
        if converter_class is None:
            raise NotImplementedError(
                f"No converter registered for HF discriminator {type_name!r} at {'.'.join(self.fast_llm_paths[0])}"
            )
        sub_fast_llm = converter_class.import_config(sub_hf)
        _safe_set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], sub_fast_llm)

    def _consumed_hf_paths(self) -> frozenset[tuple[str, ...]]:
        # Union of all registered sub-classes' claims (under the shared hf_path prefix when present).
        # At runtime only one sub-class fires; the static union is a safe over-claim — we only need to
        # not flag known keys, never to flag missing ones.
        paths: set[tuple[str, ...]] = set()
        if self.hf_paths:
            prefix = self.hf_paths[0]
            paths.add(prefix + (self._hf_discriminator_key,))
            for sub_class in self._registry.values():
                for sub_path in sub_class._consumed_hf_paths():
                    paths.add(prefix + sub_path)
        else:
            paths.add((self._hf_discriminator_key,))
            for sub_class in self._registry.values():
                paths |= sub_class._consumed_hf_paths()
        return frozenset(paths)


class TypedDictContainerConfigConverter(ConfigConverter):
    """Maps a Fast-LLM ``dict[str, Config]`` field to an HF ``dict[str, dict]`` where each entry is round-tripped
    through a per-class section converter selected via the entry's runtime type (export) or HF discriminator (import).

    Each entry's HF subdict carries a discriminator key (``"type"`` by default) populated from the converter's
    ``hf_type_name``. For homogeneous dicts, register a single class with ``hf_type_name = None``; the discriminator
    is then omitted on export and ignored on import.
    """

    fast_llm_recurses: typing.ClassVar[bool] = True

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
        _safe_set_nested_dict_value(hf_out, self.hf_paths[0], out)

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
            out[name] = sub_fast_llm
        _safe_set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], out)

    def _consumed_hf_paths(self) -> frozenset[tuple[str, ...]]:
        # Blanket prefix: per-entry sub-dicts are user-named pattern keys; we can't statically
        # enumerate which entries appear or what keys those entries claim.
        return frozenset({self.hf_paths[0]})


# ============================================================
# Section converter — converts one Fast-LLM config class
# ============================================================


class ConfigSectionConverter(abc.ABC):
    """Base class for converting one Fast-LLM ``Config`` class ↔ one HF dict subtree.

    Subclasses declare the conversion via ``_create_config_converters`` (config side) and
    ``_create_weight_converters`` (weight side). Format-specific cross-field invariants go on the
    ``_validate_export`` hook.

    Subclasses that participate in :class:`DispatchConfigConverter` set ``hf_type_name`` to the discriminator value
    used by the HF format (e.g. ``"attention"``, ``"mamba"``).

    .. warning::
       Both ``_create_config_converters`` and ``_create_weight_converters`` are ``@functools.cache``\\ d on
       the base class. Subclasses that override them must return a *fresh* dict (idiomatically
       ``{**super()._create_..._converters(), ...}``); mutating the parent's returned dict in place would
       corrupt the cache entry for every subsequent caller.
    """

    fast_llm_config_class: typing.ClassVar[type[Config]]
    hf_type_name: typing.ClassVar[str | None] = None

    @classmethod
    @functools.cache
    def _create_config_converters(cls) -> dict[str, ConfigConverter]:
        """Return declarations keyed by stable string name. Subclasses override entries by re-declaring the key.

        Cached per class — declarations are immutable and depend only on ``cls``. Subclasses must build
        and return a *fresh* dict (idiomatically ``{**super()._create_config_converters(), ...}``); mutating
        the returned dict in place would corrupt the parent's cache entry for every subsequent caller.
        """
        raise NotImplementedError

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, "WeightConverter"]:
        """Return weight-conversion declarations keyed by stable string name.

        Same shape and caching rules as :meth:`_create_config_converters`. Section-relative names; the
        walker (:meth:`emit_weight_converters`) prepends the section's full ``(fast_llm_prefix, hf_prefix)``
        pair as it descends. Defaults to no declarations — sections that don't own any weights leave this
        unoverridden.
        """
        return {}

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
        cls._validate_export(config)
        out: dict = {}
        for converter in cls._create_config_converters().values():
            converter.export_to(config, out)
        return out

    @classmethod
    def import_config(cls, hf_dict: dict) -> dict:
        """Convert an HF config dict to a Fast-LLM config dict via this section's declarations.

        When ``fast_llm_config_class`` carries a ``dynamic_type_name`` (i.e. the target is a registered
        dynamic-type subclass), inject ``"type": <name>`` so the caller's ``from_dict`` dispatches to the
        correct subclass without each section converter having to prepend it manually.
        """
        out: dict = {}
        for converter in cls._create_config_converters().values():
            converter.import_to(hf_dict, out)
        fast_llm_type = getattr(cls.fast_llm_config_class, "dynamic_type_name", None)
        if fast_llm_type is not None:
            out = {"type": fast_llm_type, **out}
        return out

    @classmethod
    def get_converters(
        cls,
        config: Config,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list["WeightConverter"]:
        """Imperative-shape entry point — delegates to the declarative walker.

        Section converters that haven't migrated override this with a custom body; migrated sections leave
        it inherited. The ``drop_on_export`` parameter is accepted for signature compatibility with the
        pre-migration shape but is unused — the walker handles tied embeddings via
        :class:`OutputProjectionWeightConverter`. Once every consumer is migrated this shim and the
        parameter are removed.
        """
        return cls.emit_weight_converters(config, fast_llm_prefix, hf_prefix)

    @classmethod
    def emit_weight_converters(
        cls,
        config: Config,
        fast_llm_prefix: str,
        hf_prefix: str,
        *,
        root_config: Config | None = None,
    ) -> list["WeightConverter"]:
        """Walk this section's weight declarations against ``config`` into a flat list of fully-qualified
        :class:`WeightConverter` instances.

        Each declaration in :meth:`_create_weight_converters` returns one or more ``WeightConverter`` leaves via
        its :meth:`WeightConverter._emit` hook. Structural primitives (Nested, BlockSequence) recurse into
        sub-section converters; leaves return a single prefixed copy of themselves. ``root_config`` carries the
        top-level config through the recursion for primitives whose behaviour depends on it (e.g.
        :class:`OutputProjectionWeightConverter` consults ``root_config.tied_embedding_weight``); the walker
        seeds it from ``config`` on the outermost call.
        """
        if root_config is None:
            root_config = config
        out: list["WeightConverter"] = []
        for declaration in cls._create_weight_converters().values():
            out.extend(declaration._emit(config, fast_llm_prefix, hf_prefix, root_config=root_config))
        return out

    @classmethod
    @functools.cache
    def _consumed_hf_paths(cls) -> frozenset[tuple[str, ...]]:
        """Set of HF dict paths consumed by this section's declaration tree.

        Each entry is a tuple-of-keys from the section's HF subdict root. The :meth:`check_hf_coverage`
        walker treats every entry as a *recursive prefix* — once an input path matches any prefix,
        descent into deeper sub-dicts stops.

        Each declaration advertises its claims via :meth:`ConfigConverter._consumed_hf_paths`
        (default: ``hf_paths``; overridden by primitives whose claims depend on a sub-converter
        registry, a nested-path prefix, or a discriminator key). The aggregate is cached per
        section class.
        """
        paths: set[tuple[str, ...]] = set()
        for declaration in cls._create_config_converters().values():
            paths |= declaration._consumed_hf_paths()
        return frozenset(paths)

    @classmethod
    def check_hf_coverage(cls, hf_dict: dict, *, allowlist: frozenset[str] = frozenset()) -> None:
        """Raise :class:`ValueError` if the input HF dict carries keys not consumed by any declaration.

        Walks ``hf_dict`` recursively. A path is considered covered if it (or any of its prefixes) is in
        :meth:`_consumed_hf_paths`, or if any segment of the path appears in ``allowlist`` (so transformers'
        generic ``PretrainedConfig`` metadata keys — ``architectures``, ``torch_dtype``, ``transformers_version``,
        … — are accepted at any depth, including under nested sub-configs like Llava's ``vision_config``).
        Uncovered leaves raise; uncovered sub-dicts trigger descent into their entries to surface the offending
        leaf path.

        Catches transformers-version drift, manual edits, and corrupted configs at the import boundary —
        the symmetric counterpart to the architecture-coverage check (which is statically verified by
        ``tests/models/test_converters.py``).
        """
        prefixes = cls._consumed_hf_paths()

        def walk(value: typing.Any, path: tuple[str, ...]) -> None:
            for length in range(1, len(path) + 1):
                if path[:length] in prefixes:
                    return
            if any(segment in allowlist for segment in path):
                return
            if isinstance(value, dict):
                for key, sub in value.items():
                    walk(sub, path + (key,))
                return
            raise ValueError(
                f"{cls.__name__}: HF config has unknown key '{'.'.join(path)}' (value: {value!r}). "
                "Possible transformers-version mismatch, manual edit, or corrupted config."
            )

        for key, value in hf_dict.items():
            walk(value, (key,))

    @classmethod
    def check_architecture_coverage(cls, config: Config) -> None:
        """Raise if any architecture-hint field reachable from the section's config (recursively) is not consumed.

        Coverage is structural (based on field hints), not value-based: every architecture field at every depth
        must be accounted for, even when it currently holds its Fast-LLM default. The walker descends into any
        field whose runtime value is a :class:`Config`, collecting an architecture-leaf list, and matches each
        leaf against the section's declarations:

        * Recursive declarations (``fast_llm_recurses = True`` — Nested/Dispatch/TypedDictContainer/Ignored, plus
          Custom when its author opts in) cover the entire subtree under each listed prefix.
          Nested/Dispatch/TypedDict delegate to a sub-converter that runs its own coverage check; Ignored and
          recursive Custom assume the author has handled the subtree.
        * Non-recursive declarations (Rename, ConstantImport/Export, Default, Optional, ImportOnly, Custom by
          default) must list every architecture leaf they consume by exact path.

        Invoked from a test fixture (``tests/models/test_converters.py``) — not from the production
        export/import paths. Architecture coverage is a structural invariant of the converter declarations,
        so it only needs to hold once per (converter, config-class) pair, not on every save.
        """
        Assert.is_(type(config), cls.fast_llm_config_class)
        declarations = cls._create_config_converters()
        explicit_paths: set[tuple[str, ...]] = set()
        recursive_prefixes: list[tuple[str, ...]] = []
        for converter in declarations.values():
            if converter.fast_llm_recurses:
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
                    f"or pass ``fast_llm_recurses=True`` to a Custom/ImportOnly converter when claiming the whole "
                    f"subtree)"
                )
            raise ValueError(
                f"{cls.__name__}: architecture-hint fields on {type(config).__name__} "
                f"have no converter declaration: {[ '.'.join(p) for p in missing ]}{hint}"
            )


def _prepend_prefix(prefix: str, names: tuple[str, ...]) -> tuple[str, ...]:
    """Prepend ``prefix`` to each name. Empty ``prefix`` is a no-op; empty ``names`` (drop side) stays empty."""
    if not prefix:
        return names
    return tuple(f"{prefix}.{name}" for name in names)


class WeightConverter:
    """Leaf weight-conversion declaration / emitted instance.

    As a declaration in :meth:`ConfigSectionConverter._create_weight_converters`, the ``fast_llm_name`` and
    ``export_name`` are *section-relative*; the walker constructs a fully-qualified emitted copy by prepending
    the section prefixes via :meth:`_emit`. Subclasses that need extra construction context (e.g. capturing
    a sub-config for use inside ``export_weight``/``import_weight``) override :meth:`_emit` accordingly.
    """

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

    def _emit(
        self,
        config: Config,
        fast_llm_prefix: str,
        hf_prefix: str,
        *,
        root_config: Config,
    ) -> list["WeightConverter"]:
        """Return a fully-qualified emitted copy of this leaf.

        Subclasses that capture extra construction state (e.g. :class:`KeyValueWeightConverter` stashing
        an :class:`AttentionConfig`) override this hook to pass that state into the emitted copy.
        """
        return [
            type(self)(
                _prepend_prefix(fast_llm_prefix, self.fast_llm_name),
                _prepend_prefix(hf_prefix, self.export_name),
                config,
            )
        ]


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


class TransposeSplitWeightConverter(WeightConverter):
    """Split a merged weight across the last dim with an additional transpose.

    Equivalent to :class:`SplitWeightConverter` for non-gated MLPs (trivial split) and for 1-D biases
    (trivial transpose); the real behaviour kicks in for the down-projection of a gated MLP where HF
    stores the weight in transposed orientation.
    """

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (merged_weight,) = weight
        return tuple(t.contiguous() for t in merged_weight[:].t().chunk(len(self.export_name), dim=-1))

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        merged_weight = torch.cat([weight_[:] for weight_ in weight], dim=-1)
        return (merged_weight.t().contiguous(),)


class KeyValueWeightConverter(WeightConverter):
    """Pack/unpack a fused key-value tensor across the two HF names.

    Fast-LLM packs key/value as a single concatenated tensor; HF stores them as two siblings
    (``k_proj`` / ``v_proj``). Identity for bias because biases are concatenated the same way.
    """

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (key_value,) = weight
        key, value = key_value[:].chunk(2)
        return key, value

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        key, value = weight
        key_value = torch.cat([key[:], value[:]])
        return (key_value,)


class PatchEmbeddingWeightConverter(WeightConverter):
    """Reshape a vision patch-embedding weight from Fast-LLM's flat ``(out, channels*h*w)`` shape to HF's
    ``(out, channels, h, w)`` (and back). Requires a config exposing ``input_channels``/``patch_height``/
    ``patch_width`` via the constructor's ``config`` argument."""

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        return tuple(
            weight_[:].view(
                *weight_[:].shape[:-1],
                self._config.input_channels,
                self._config.patch_height,
                self._config.patch_width,
            )
            for weight_ in weight
        )

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        return tuple(
            weight_[:].view(
                *weight_[:].shape[:-3],
                self._config.input_channels * self._config.patch_height * self._config.patch_width,
            )
            for weight_ in weight
        )


class OutputProjectionWeightConverter(WeightConverter):
    """Marker for the LM-head output projection (typically ``head.output_weights`` ↔ ``lm_head.weight``).

    When the root config has ``tied_embedding_weight=True``, the walker drops this declaration entirely —
    HF stores tied embeddings as just ``embed_tokens.weight`` with no separate ``lm_head.weight``. Replaces
    the per-call ``drop_on_export=exported_config["tie_word_embeddings"]`` plumbing.
    """

    def _emit(
        self,
        config: Config,
        fast_llm_prefix: str,
        hf_prefix: str,
        *,
        root_config: Config,
    ) -> list[WeightConverter]:
        if getattr(root_config, "tied_embedding_weight", False):
            return []
        return super()._emit(config, fast_llm_prefix, hf_prefix, root_config=root_config)


class NestedWeightConverter(WeightConverter):
    """Recurse into a sub-section's weight declarations.

    The sub-section's config is read from ``getattr(config, config_attr)`` (defaults to ``fast_llm_prefix``
    when the state-dict prefix and the parent's attribute name agree). The walker descends into
    ``sub_converter_class._create_weight_converters()`` with extended prefixes. Mirrors
    :class:`NestedConfigConverter` on the config side.

    The separate ``config_attr`` covers cases like a block's single ``normalization`` config feeding two
    state-dict prefixes (``norm_1`` / ``norm_2``).
    """

    def __init__(
        self,
        fast_llm_prefix: str,
        hf_prefix: str,
        sub_converter_class: type["ConfigSectionConverter"],
        *,
        config_attr: str | None = None,
    ):
        super().__init__((), ())
        self._fast_llm_prefix = fast_llm_prefix
        self._hf_prefix = hf_prefix
        self._sub_converter_class = sub_converter_class
        self._config_attr = config_attr if config_attr is not None else fast_llm_prefix

    def _emit(
        self,
        config: Config,
        fast_llm_prefix: str,
        hf_prefix: str,
        *,
        root_config: Config,
    ) -> list[WeightConverter]:
        sub_config = getattr(config, self._config_attr)
        return self._sub_converter_class.emit_weight_converters(
            sub_config,
            f"{fast_llm_prefix}.{self._fast_llm_prefix}" if fast_llm_prefix else self._fast_llm_prefix,
            f"{hf_prefix}.{self._hf_prefix}" if hf_prefix and self._hf_prefix else (hf_prefix or self._hf_prefix),
            root_config=root_config,
        )


class BlockSequenceWeightConverter(WeightConverter):
    """Fan out a per-block sub-section across every position in a block sequence.

    The sub-section's converter class is resolved per-position from ``block_converter_class``: by default,
    the same class for every position; when ``dispatch_registry`` is provided, the per-position config is
    matched against the registry keys (Apriel's hybrid-block dispatch — different mixer types per layer).

    Handles both ``FixedBlockSequenceConfig`` (single repeated block) and ``PatternBlockSequenceConfig``
    (per-position blocks indexed via ``decoder.expanded_pattern``).
    """

    def __init__(
        self,
        fast_llm_prefix: str,
        hf_prefix: str,
        block_converter_class: type["ConfigSectionConverter"],
        *,
        config_attr: str | None = None,
        dispatch_registry: dict[type[Config], type["ConfigSectionConverter"]] | None = None,
    ):
        super().__init__((), ())
        self._fast_llm_prefix = fast_llm_prefix
        self._hf_prefix = hf_prefix
        self._block_converter_class = block_converter_class
        self._config_attr = config_attr if config_attr is not None else fast_llm_prefix
        self._dispatch_registry = dispatch_registry

    def _emit(
        self,
        config: Config,
        fast_llm_prefix: str,
        hf_prefix: str,
        *,
        root_config: Config,
    ) -> list[WeightConverter]:
        # Lazy import to keep external.py free of layers/ dependencies.
        from fast_llm.layers.block.config import FixedBlockSequenceConfig, PatternBlockSequenceConfig

        block_sequence = getattr(config, self._config_attr)
        if isinstance(block_sequence, FixedBlockSequenceConfig):
            per_position_blocks = [block_sequence.block] * block_sequence.num_blocks
        elif isinstance(block_sequence, PatternBlockSequenceConfig):
            per_position_blocks = [block_sequence.blocks[name] for name in block_sequence.expanded_pattern]
        else:
            raise NotImplementedError(type(block_sequence).__name__)

        fast_llm_root = f"{fast_llm_prefix}.{self._fast_llm_prefix}" if fast_llm_prefix else self._fast_llm_prefix
        hf_root = f"{hf_prefix}.{self._hf_prefix}" if hf_prefix and self._hf_prefix else (hf_prefix or self._hf_prefix)
        out: list[WeightConverter] = []
        for index, block in enumerate(per_position_blocks):
            block_class = (
                self._dispatch_registry[type(block.mixer)]
                if self._dispatch_registry is not None
                else self._block_converter_class
            )
            out += block_class.emit_weight_converters(
                block,
                f"{fast_llm_root}.{index}",
                f"{hf_root}.{index}",
                root_config=root_config,
            )
        return out


class LinearWeightConverter(WeightConverter):
    """Bundle a linear layer's ``.weight`` and (conditionally) ``.bias`` declarations into one entry.

    Bias presence is resolved at emission time from the live section config: ``bias_fn(config)`` returns
    a bool. The default reads ``config.add_linear_biases`` — the shared flag every Llama-style attention/MLP
    section carries. Sections with per-layer overrides (e.g. Apriel Mamba's ``dt_layer`` / ``convolution_layer``)
    pass a lambda that resolves the override.

    ``transform`` selects the leaf class for both weight and bias: :class:`WeightConverter` for plain rename
    (the default), :class:`SplitWeightConverter` for fused → split, :class:`KeyValueWeightConverter` for
    fused KV → separate K/V, :class:`TransposeSplitWeightConverter` for MLP down-projection.

    Replaces the imperative ``get_weight_and_bias_converters`` / ``effective_bias`` helpers.
    """

    def __init__(
        self,
        fast_llm_prefix: str,
        hf_prefix: str | tuple[str, ...] | typing.Callable[[Config], str | tuple[str, ...]],
        *,
        transform: type[WeightConverter] = WeightConverter,
        bias_fn: typing.Callable[[Config], bool] = lambda c: getattr(c, "add_linear_biases", False),
    ):
        super().__init__((), ())
        self._fast_llm_prefix = fast_llm_prefix
        # ``hf_prefix`` may be a callable (e.g. Mixtral's ``experts.{i}.w1``-style fan-out where the
        # expert count comes from the live config).
        self._hf_prefix = hf_prefix
        self._transform = transform
        self._bias_fn = bias_fn

    def _emit(
        self,
        config: Config,
        fast_llm_prefix: str,
        hf_prefix: str,
        *,
        root_config: Config,
    ) -> list[WeightConverter]:
        resolved = self._hf_prefix(config) if callable(self._hf_prefix) else self._hf_prefix
        hf_prefixes: tuple[str, ...] = (resolved,) if isinstance(resolved, str) else tuple(resolved)
        weight_fast_llm = _prepend_prefix(fast_llm_prefix, (f"{self._fast_llm_prefix}.weight",))
        weight_hf = _prepend_prefix(hf_prefix, tuple(f"{p}.weight" for p in hf_prefixes))
        emitted: list[WeightConverter] = [self._transform(weight_fast_llm, weight_hf, config)]
        if self._bias_fn(config):
            bias_fast_llm = _prepend_prefix(fast_llm_prefix, (f"{self._fast_llm_prefix}.bias",))
            bias_hf = _prepend_prefix(hf_prefix, tuple(f"{p}.bias" for p in hf_prefixes))
            emitted.append(self._transform(bias_fast_llm, bias_hf, config))
        return emitted


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
