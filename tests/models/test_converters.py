"""Static checks on every checkpoint format's converter tree.

For each registered ``HuggingfaceStateDictCheckpointHandler``, walk its modular converter structure —
``base_model_converter_class`` and the ``ConfigSectionConverter`` classes reached transitively through
``Nested``/``Dispatch``/``TypedDictContainer`` declarations — and verify, at every node:

* Architecture-hint fields on ``cls.fast_llm_config_class`` are all consumed by some declaration.
* OptionalConfigConverter sentinels match the resolved field default. Otherwise an exported value equal
  to the sentinel becomes absent on disk and re-imports as a different default, silently breaking round-trip.

Replaces the per-export ``check_architecture_coverage`` invocation that used to happen on every save.
"""

import typing

import pytest

# Force registration of every format handler.
import fast_llm.models.gpt.conversion.auto  # noqa: F401
import fast_llm.models.multimodal.conversion.auto  # noqa: F401
from fast_llm.engine.checkpoint.external import (
    ConfigSectionConverter,
    DispatchConfigConverter,
    NestedConfigConverter,
    OptionalConfigConverter,
    TypedDictContainerConfigConverter,
    _get_attr_path,
)
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.block.config import PatternBlockSequenceConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig, StochasticMixerConfig

# Configs that don't default-construct cleanly need a minimal-valid factory.
_DEFAULT_FACTORIES: dict[type, typing.Callable[[], typing.Any]] = {
    PatternBlockSequenceConfig: lambda: PatternBlockSequenceConfig(
        blocks={"x": DecoderBlockConfig()},
        pattern=("x",),
    ),
    StochasticMixerConfig: lambda: StochasticMixerConfig(
        mixers={"x": AttentionConfig()},
        main_mixer_name="x",
    ),
}


def _default_instance(config_class: type) -> typing.Any:
    factory = _DEFAULT_FACTORIES.get(config_class)
    return factory() if factory is not None else config_class()


def _all_format_handlers() -> list[type[HuggingfaceStateDictCheckpointHandler]]:
    seen: set[type[HuggingfaceStateDictCheckpointHandler]] = set()
    out: list[type[HuggingfaceStateDictCheckpointHandler]] = []

    def visit(cls: type) -> None:
        for sub in cls.__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            # Concrete handlers declare a ``base_model_converter_class``; abstract intermediaries don't.
            if getattr(sub, "base_model_converter_class", None) is not None:
                out.append(sub)
            visit(sub)

    visit(HuggingfaceStateDictCheckpointHandler)
    return out


def _children(node: type) -> list[type]:
    """Return every sub-converter class reachable from ``node``.

    Picks up three complementary structures:
    * ``ConfigSectionConverter`` declarations — the ``_converter_class`` on each Nested/Dispatch/TypedDict.
    * ``*_converter_class`` ClassVars — the polymorphism extension points used by aggregator nodes
      (e.g. ``LlavaBaseModelConverter`` is not itself a section converter but exposes
      ``vision_model_converter_class`` and ``language_model_converter_class``).
    * ``*_converter_classes`` ClassVar dicts — imperative dispatchers that fan out to per-key sub-converters
      (e.g. ``AprielBlockConverter._converter_classes`` keyed on mixer config class).
    """
    out: list[type] = []
    if isinstance(node, type) and issubclass(node, ConfigSectionConverter):
        for declaration in node._create_config_converters().values():
            if isinstance(declaration, NestedConfigConverter):
                out.append(declaration._converter_class)
            elif isinstance(declaration, (DispatchConfigConverter, TypedDictContainerConfigConverter)):
                out.extend(declaration._registry.values())
    for name in dir(node):
        if name == "base_model_converter_class":
            continue
        if name.endswith("_converter_class"):
            attr = getattr(node, name, None)
            if isinstance(attr, type):
                out.append(attr)
        elif name.endswith("_converter_classes"):
            attr = getattr(node, name, None)
            if isinstance(attr, dict):
                out.extend(value for value in attr.values() if isinstance(value, type))
    return out


def _walk(root: type) -> typing.Iterator[type]:
    """Yield ``root`` and every converter class reachable from it (each at most once)."""
    seen: set[type] = set()
    stack: list[type] = [root]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        yield node
        stack.extend(_children(node))


_HANDLERS = _all_format_handlers()


@pytest.mark.parametrize("handler_class", _HANDLERS, ids=lambda h: h.__name__)
def test_format_converter_tree(handler_class: type[HuggingfaceStateDictCheckpointHandler]) -> None:
    """Walk the format's converter tree from ``base_model_converter_class``; check every section node."""
    for converter_class in _walk(handler_class.base_model_converter_class):
        if not (isinstance(converter_class, type) and issubclass(converter_class, ConfigSectionConverter)):
            continue
        if getattr(converter_class, "fast_llm_config_class", None) is None:
            continue
        config = _default_instance(converter_class.fast_llm_config_class)
        converter_class.check_architecture_coverage(config)
        for name, declaration in converter_class._create_config_converters().items():
            if not isinstance(declaration, OptionalConfigConverter):
                continue
            path = declaration.fast_llm_paths[0]
            default = _get_attr_path(config, path)
            assert declaration._sentinel == default, (
                f"{converter_class.__name__}.{name}: sentinel {declaration._sentinel!r} "
                f"does not match field default {default!r} at path {'.'.join(path)}"
            )
