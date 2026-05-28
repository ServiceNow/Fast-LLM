"""Static checks on every checkpoint format's converter tree.

For each registered ``HuggingfaceStateDictCheckpointHandler``, walk its modular converter structure —
``base_model_converter_class`` and the ``ConfigSectionConverter`` classes reached transitively through
``Nested``/``Dispatch``/``TypedDictContainer`` declarations — and verify, at every node:

* Architecture-hint fields on ``cls.fast_llm_config_class`` are all consumed by some declaration.
* OptionalConfigConverter sentinels match the resolved field default. Otherwise an exported value equal
  to the sentinel becomes absent on disk and re-imports as a different default, silently breaking round-trip.

Plus an end-to-end weight-coverage walker (:func:`test_format_weight_coverage`) — for each test
fixture with a checkpoint format, materialise the Fast-LLM model and assert every parameter is consumed
by some leaf :class:`WeightConverter`. Catches the "silent drop" failure mode where a model param has
no converter and ``_convert_state_dict`` skips it on export.
"""

import typing

import pytest

# Force registration of every format handler.
import fast_llm.models.gpt.conversion.auto  # noqa: F401
import fast_llm.models.multimodal.conversion.auto  # noqa: F401
from fast_llm.engine.checkpoint.external import (
    ConfigSectionConverter,
    NestedConfigConverter,
    OptionalConfigConverter,
    _get_attr_path,
    _safe_set_nested_dict_value,
)
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.block.config import PatternBlockSequenceConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig, StochasticMixerConfig
from tests.utils.model_configs import MODEL_CONFIGS

# Configs that don't default-construct cleanly need a minimal-valid factory.
_DEFAULT_FACTORIES: dict[type, typing.Callable[[], typing.Any]] = {
    PatternBlockSequenceConfig: lambda: PatternBlockSequenceConfig(
        blocks={"x": DecoderBlockConfig()},
        pattern=["x"],
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
            elif isinstance(getattr(declaration, "_registry", None), dict):
                # Dispatch / ListDispatch / TypedDictContainer (and any future dispatch primitive)
                # share the ``_registry: dict[..., type[ConfigSectionConverter]]`` convention.
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


def test_safe_set_nested_dict_value_collision() -> None:
    """A divergent write to an already-populated path must raise instead of silently overwriting.

    Regression for the cross-section invariant: e.g. Llama's decoder block and head both export
    ``rms_norm_eps`` via flat-merge — a divergent value used to silently overwrite under a plain
    ``set_nested_dict_value`` semantics.
    """
    out: dict = {}
    _safe_set_nested_dict_value(out, ("rms_norm_eps",), 1e-5)
    # Same value at the same path: no-op.
    _safe_set_nested_dict_value(out, ("rms_norm_eps",), 1e-5)
    assert out == {"rms_norm_eps": 1e-5}
    # Divergent value: raise.
    with pytest.raises(AssertionError):
        _safe_set_nested_dict_value(out, ("rms_norm_eps",), 1e-6)
    # Works recursively for nested paths.
    _safe_set_nested_dict_value(out, ("nested", "key"), "value")
    with pytest.raises(AssertionError):
        _safe_set_nested_dict_value(out, ("nested", "key"), "other")


@pytest.mark.parametrize(
    "fixture_name", [name for name, cfg in MODEL_CONFIGS.items() if cfg.checkpoint_format is not None]
)
def test_format_weight_coverage(fixture_name: str) -> None:
    """Every Fast-LLM parameter must be consumed by some :class:`WeightConverter`.

    Materialises the fixture's base model (CPU, meta tensors via ``ParameterMeta`` — no distributed
    setup) and compares ``named_parameters()`` against the set of ``fast_llm_name`` entries emitted by
    ``base_model_converter_class.get_converters(config)``. Runtime-tied parameters
    (``BaseModel.get_tied_parameters``) count as covered if any member of their group has a converter,
    matching the export-time behaviour where a single shared weight is serialised once.
    """
    model_testing_config = MODEL_CONFIGS[fixture_name]
    handler = model_testing_config.checkpoint_format.get_handler_class()
    base_model_config = model_testing_config.base_model_config_class.from_dict(
        model_testing_config.config_dict["model"]["base_model"]
    )
    base_model = base_model_config.base_model_class(base_model_config, DistributedConfig())

    param_id_to_name = {id(parameter): name for name, parameter in base_model.named_parameters()}
    model_names = set(param_id_to_name.values())
    tied_groups = [
        frozenset(param_id_to_name[id(parameter)] for parameter in parameters)
        for parameters in base_model.get_tied_parameters().values()
    ]

    consumed: set[str] = set()
    for leaf in handler.base_model_converter_class.get_converters(base_model_config):
        consumed.update(leaf.fast_llm_name)

    # Tied closure: any group with at least one explicit consumer is covered in full.
    covered = set(consumed)
    for group in tied_groups:
        if group & consumed:
            covered |= group

    missing = sorted(model_names - covered)
    phantom = sorted(consumed - model_names)
    assert not missing and not phantom, (
        f"{handler.__name__}: weight coverage mismatch — "
        f"Fast-LLM params with no converter: {missing}; "
        f"converters with no matching param: {phantom}"
    )


def test_llama_export_rejects_mismatched_block_and_head_norm_epsilon() -> None:
    """End-to-end regression: a Llama config with mismatched block/head normalization epsilon must fail to
    export. Both the decoder Custom and the head Nested write ``rms_norm_eps`` into the same HF dict; a
    silent override would drop the head value (or the block value, depending on declaration order)."""
    import copy

    from fast_llm.models.gpt.config import GPTBaseModelConfig
    from fast_llm.models.gpt.conversion.llama import LlamaBaseModelConverter

    cfg = copy.deepcopy(MODEL_CONFIGS["llama"].config_dict["model"]["base_model"])
    # Default head normalization inherits the block default (1e-5); pin head to a different value.
    cfg["head"] = {"normalization": {"type": "rms_norm", "epsilon": 1.234e-9}}
    config = GPTBaseModelConfig.from_dict(cfg)
    assert config.decoder.block.normalization.epsilon != config.head.normalization.epsilon
    with pytest.raises(AssertionError):
        LlamaBaseModelConverter.export_config(config)
