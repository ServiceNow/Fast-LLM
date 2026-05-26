import dataclasses
import functools
import logging
import typing

import transformers

from fast_llm.config import Config
from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    BlockSequenceWeightConverter,
    ConfigSectionConverter,
    ConstantImportConfigConverter,
    CustomConfigConverter,
    DefaultConfigConverter,
    IgnoredConfigConverter,
    KeyValueWeightConverter,
    LinearWeightConverter,
    NestedConfigConverter,
    NestedWeightConverter,
    OutputProjectionWeightConverter,
    RenameConfigConverter,
    SelfBlockSequenceWeightConverter,
    SplitWeightConverter,
    TransposeSplitWeightConverter,
    WeightConverter,
)
from fast_llm.engine.checkpoint.huggingface import HuggingFaceBaseModelConverter, HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.attention.rotary.config import DefaultRotaryConfig, Llama3RotaryConfig, YarnRotaryConfig
from fast_llm.layers.block.config import FixedBlockSequenceConfig, PatternBlockSequenceConfig
from fast_llm.layers.common.linear.config import AffineLinearConfig
from fast_llm.layers.common.normalization.config import RMSNormalizationConfig
from fast_llm.layers.common.peft.config import NoPeftConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.layers.decoder.mlp.config import MLPConfig
from fast_llm.layers.language_model.config import (
    LanguageModelEmbeddingsConfig,
    LanguageModelHeadConfig,
)
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.models.gpt.conversion.config import LlamaCheckpointFormat
from fast_llm.models.gpt.model import GPTModel
from fast_llm.utils import Assert, div

_TRANSFORMERS_V4 = not dataclasses.is_dataclass(transformers.PretrainedConfig)

logger = logging.getLogger(__name__)


def assert_no_peft(config: GPTBaseModelConfig) -> None:
    """Reject any non-trivial PEFT config: HuggingFace formats serialize the base weights only,
    so a configured LoRA (or other adapter) would be silently dropped on export."""
    Assert.is_(type(config.peft), NoPeftConfig)


def effective_bias(layer_config: AffineLinearConfig, default: bool) -> bool:
    """Resolve a layer's effective bias flag: explicit ``bias.enabled`` if set, else the parent default."""
    return default if layer_config.bias.enabled is None else layer_config.bias.enabled


# ============================================================
# Config converters (declarative)
# ============================================================


def _llama_rotary_export(config: AttentionConfig) -> dict:
    """Build the HF rotary block(s) from a Fast-LLM rotary config.

    Returns a dict keyed by the (Llama-flat) HF paths the converter declares; values vary with rotary subtype and
    the active transformers major version (v4 puts ``rope_theta`` flat with optional ``rope_scaling``;
    v5 consolidates everything into ``rope_parameters``).
    """
    rotary = config.rotary
    rope_parameters = {"rope_theta": rotary.theta}
    if type(rotary) is DefaultRotaryConfig:
        rope_parameters["rope_type"] = "default"
    elif type(rotary) is Llama3RotaryConfig:
        rope_parameters.update(
            {
                "rope_type": "llama3",
                "factor": rotary.scale_factor,
                "low_freq_factor": rotary.low_frequency_factor,
                "high_freq_factor": rotary.high_frequency_factor,
                "original_max_position_embeddings": rotary.original_context_length,
            }
        )
    elif type(rotary) is YarnRotaryConfig:
        rope_parameters.update(
            {
                "rope_type": "yarn",
                "attention_factor": rotary.attention_factor,
                "beta_fast": rotary.beta_fast,
                "beta_slow": rotary.beta_slow,
                "original_max_position_embeddings": rotary.original_context_length,
            }
        )
    else:
        raise NotImplementedError(f"Unsupported rotary type: {type(rotary).__name__}")

    if _TRANSFORMERS_V4:
        out: dict = {("rope_theta",): rope_parameters["rope_theta"]}
        if type(rotary) is not DefaultRotaryConfig:
            out[("rope_scaling",)] = {k: v for k, v in rope_parameters.items() if k != "rope_theta"}
        return out
    return {("rope_parameters",): rope_parameters}


def _llama_rotary_import(hf_dict: dict) -> dict:
    """Reverse of :func:`_llama_rotary_export`. Detects v4/v5 layout from the HF dict."""
    if "rope_parameters" in hf_dict:  # transformers v5
        rope_params = hf_dict["rope_parameters"]
        rope_theta = rope_params["rope_theta"]
    else:  # transformers v4
        rope_params = hf_dict.get("rope_scaling") or {}
        rope_theta = hf_dict["rope_theta"]
    rope_type = rope_params.get("rope_type", "default")
    rotary_config: dict = {"type": rope_type, "theta": rope_theta}
    if rope_type == "default":
        pass
    elif rope_type == "llama3":
        rotary_config.update(
            {
                "scale_factor": rope_params["factor"],
                "low_frequency_factor": rope_params["low_freq_factor"],
                "high_frequency_factor": rope_params["high_freq_factor"],
                "original_context_length": rope_params["original_max_position_embeddings"],
            }
        )
    elif rope_type == "yarn":
        rotary_config.update(
            {
                "attention_factor": rope_params["attention_factor"],
                "beta_fast": rope_params["beta_fast"],
                "beta_slow": rope_params["beta_slow"],
                "original_context_length": rope_params["original_max_position_embeddings"],
            }
        )
    else:
        raise NotImplementedError(f"Unsupported rotary type: {rope_type}")
    return {("rotary",): rotary_config}


class LlamaNormalizationConverter(ConfigSectionConverter):
    """Converts ``RMSNormalizationConfig`` ↔ Llama's flat ``rms_norm_eps`` field."""

    fast_llm_config_class = RMSNormalizationConfig

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "type": ConstantImportConfigConverter(("type",), "rms_norm"),
            "epsilon": RenameConfigConverter(("epsilon",), ("rms_norm_eps",)),
            "weight": IgnoredConfigConverter(("weight",)),
            "zero_centered": ConstantImportConfigConverter(("zero_centered",), False),
        }

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {"weight": WeightConverter("weight", "weight")}


class LlamaMLPConverter(ConfigSectionConverter):
    """Converts ``MLPConfig`` ↔ Llama's flat ``intermediate_size``/``mlp_bias``/``hidden_act`` fields.

    Llama is always gated (``ConstantImportConfigConverter(("gated",), True)``).
    """

    fast_llm_config_class = MLPConfig

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "intermediate_size": RenameConfigConverter(("intermediate_size",), ("intermediate_size",)),
            "add_linear_biases": RenameConfigConverter(("add_linear_biases",), ("mlp_bias",)),
            "activation": CustomConfigConverter(
                fast_llm_paths=(("activation",),),
                hf_paths=(("hidden_act",),),
                export_fn=lambda c: {("hidden_act",): c.activation.hf_name},
                import_fn=lambda hf: {("activation",): ActivationType.from_hf_name(hf["hidden_act"])},
            ),
            "gated": ConstantImportConfigConverter(("gated",), True),
            # Llama doesn't expose per-layer bias overrides; the bias-match check lives on _validate_export.
            "layers": IgnoredConfigConverter(("layer_1",), ("layer_2",)),
            "pre_norm": ConstantImportConfigConverter(("pre_norm",), None),
            "post_norm": ConstantImportConfigConverter(("post_norm",), None),
        }

    @classmethod
    def _validate_export(cls, config: MLPConfig) -> None:
        Assert.incl(config.layer_1.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.layer_2.bias.enabled, (None, config.add_linear_biases))

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "layer_1": LinearWeightConverter("layer_1", ("gate_proj", "up_proj"), transform=SplitWeightConverter),
            "layer_2": LinearWeightConverter("layer_2", "down_proj", transform=TransposeSplitWeightConverter),
        }


class LlamaAttentionConverter(ConfigSectionConverter):
    """Converts ``AttentionConfig`` ↔ Llama's flat attention fields.

    Notable wrinkles:
      - ``head_dim`` is computed from ``hidden_size // num_attention_heads`` when missing on import.
      - Rotary handling is delegated to a :class:`CustomConfigConverter` because it spans v4/v5 transformers
        layouts and three rotary subtypes.
      - Per-layer linear biases (query/key/value/dense) are validated to match ``add_linear_biases`` on
        ``_validate_export``; Llama does not expose layer-level overrides, so the sub-config fields are
        blanket-consumed via :class:`IgnoredConfigConverter`.
    """

    fast_llm_config_class = AttentionConfig

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "heads": RenameConfigConverter(("heads",), ("num_attention_heads",)),
            "head_groups": RenameConfigConverter(("head_groups",), ("num_key_value_heads",)),
            "head_size": DefaultConfigConverter(
                ("head_size",),
                ("head_dim",),
                hf_default_fn=lambda hf: div(hf["hidden_size"], hf["num_attention_heads"]),
            ),
            "add_linear_biases": RenameConfigConverter(("add_linear_biases",), ("attention_bias",)),
            "dropout": RenameConfigConverter(("dropout",), ("attention_dropout",)),
            "causal": ConstantImportConfigConverter(("causal",), True),
            "softmax_scale_power": ConstantImportConfigConverter(("softmax_scale_power",), 0.5),
            "linear_layers": IgnoredConfigConverter(
                ("query_layer",), ("key_layer",), ("value_layer",), ("dense_layer",)
            ),
            "rotary": CustomConfigConverter(
                fast_llm_paths=(("rotary",),),
                hf_paths=(("rope_theta",), ("rope_scaling",), ("rope_parameters",)),
                export_fn=_llama_rotary_export,
                import_fn=_llama_rotary_import,
                fast_llm_recurses=True,
            ),
            "query_norm": ConstantImportConfigConverter(("query_norm",), None),
            "key_norm": ConstantImportConfigConverter(("key_norm",), None),
            "value_norm": ConstantImportConfigConverter(("value_norm",), None),
            "shared_key_value": ConstantImportConfigConverter(("shared_key_value",), False),
        }

    @classmethod
    def _validate_export(cls, config: AttentionConfig) -> None:
        """Default: Llama requires per-layer biases to be unset (``None``) or to match ``add_linear_biases``.

        Subclasses (e.g. Qwen2 with always-on Q/K/V biases and no dense bias) override.
        """
        Assert.is_(type(config), AttentionConfig)
        Assert.incl(config.query_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.key_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.value_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.dense_layer.bias.enabled, (None, config.add_linear_biases))

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "query": LinearWeightConverter("query", "q_proj"),
            "key_value": LinearWeightConverter("key_value", ("k_proj", "v_proj"), transform=KeyValueWeightConverter),
            "dense": LinearWeightConverter("dense", "o_proj"),
        }


class LlamaBlockConverter(ConfigSectionConverter):
    """Converts ``DecoderBlockConfig`` ↔ Llama block fields (flat-merged into the parent's HF dict)."""

    fast_llm_config_class = DecoderBlockConfig

    mixer_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = LlamaAttentionConverter
    mlp_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = LlamaMLPConverter
    normalization_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = LlamaNormalizationConverter

    hf_mixer_name: typing.ClassVar[str] = "self_attn"
    hf_mlp_name: typing.ClassVar[str] = "mlp"
    hf_norm_1_name: typing.ClassVar[str] = "input_layernorm"
    hf_norm_2_name: typing.ClassVar[str] = "post_attention_layernorm"

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "mixer": NestedConfigConverter(("mixer",), cls.mixer_converter_class),
            "mlp": NestedConfigConverter(("mlp",), cls.mlp_converter_class),
            "normalization": NestedConfigConverter(("normalization",), cls.normalization_converter_class),
            "pre_mixer_normalization": ConstantImportConfigConverter(("pre_mixer_normalization",), None),
            "pre_mlp_normalization": ConstantImportConfigConverter(("pre_mlp_normalization",), None),
            "post_mixer_normalization": ConstantImportConfigConverter(("post_mixer_normalization",), None),
            "post_mlp_normalization": ConstantImportConfigConverter(("post_mlp_normalization",), None),
            "output_scale": IgnoredConfigConverter(("output_scale",)),
        }

    @classmethod
    def _validate_export(cls, config: DecoderBlockConfig) -> None:
        Assert.custom(lambda v: not v, config.output_scale.enabled)

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "mixer": NestedWeightConverter("mixer", cls.hf_mixer_name, cls.mixer_converter_class),
            "mlp": NestedWeightConverter("mlp", cls.hf_mlp_name, cls.mlp_converter_class),
            "norm_1": NestedWeightConverter(
                "norm_1", cls.hf_norm_1_name, cls.normalization_converter_class, config_attr="normalization"
            ),
            "norm_2": NestedWeightConverter(
                "norm_2", cls.hf_norm_2_name, cls.normalization_converter_class, config_attr="normalization"
            ),
        }


def _llama_decoder_export(
    decoder_config: FixedBlockSequenceConfig | PatternBlockSequenceConfig,
    block_converter_class: type[ConfigSectionConverter],
) -> dict:
    """Convert a Fast-LLM polymorphic Fixed/Pattern block sequence to Llama's flat HF representation.

    Pattern: assert all blocks export identical HF (Llama's format has no per-block discriminator), then use
    the common export. Fixed: just delegate to the single block.
    """
    if isinstance(decoder_config, PatternBlockSequenceConfig):
        exports = [block_converter_class.export_config(block) for block in decoder_config.blocks.values()]
        for other in exports[1:]:
            Assert.eq(exports[0], other)
        block_hf = exports[0]
    elif isinstance(decoder_config, FixedBlockSequenceConfig):
        block_hf = block_converter_class.export_config(decoder_config.block)
    else:
        raise NotImplementedError(f"Unsupported decoder type: {type(decoder_config).__name__}")
    return {**block_hf, "num_hidden_layers": decoder_config.num_blocks}


class LlamaDecoderConverter(ConfigSectionConverter):
    """Converts ``FixedBlockSequenceConfig`` ↔ Llama's flat decoder shape (per-block fields + ``num_hidden_layers``).

    Used by formats that don't compose at the :class:`LlamaBaseModelConverter` level — currently only
    Pixtral's vision encoder (:class:`PixtralEncoderConverter`). The standard text formats
    (Mistral/Qwen2/Mixtral) use the inline dispatch inside
    :class:`LlamaBaseModelConverter._create_config_converters` instead, parameterised by
    ``block_converter_class``.
    """

    fast_llm_config_class = FixedBlockSequenceConfig
    block_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = LlamaBlockConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "block": NestedConfigConverter(("block",), cls.block_converter_class),
            "num_blocks": RenameConfigConverter(("num_blocks",), ("num_hidden_layers",)),
        }

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        # The section config IS a ``FixedBlockSequenceConfig`` (no parent attribute holding it).
        return {
            "blocks": SelfBlockSequenceWeightConverter(cls.block_converter_class),
        }


class LlamaEmbeddingsConverter(ConfigSectionConverter):
    """Converts ``LanguageModelEmbeddingsConfig`` ↔ Llama (flat ``vocab_size``).

    Llama has no learnable position embeddings; ``num_position_embeddings`` is irrelevant when
    ``position_embeddings.enabled`` is ``False``/``None`` and is therefore blanket-consumed.
    """

    fast_llm_config_class = LanguageModelEmbeddingsConfig

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "vocab_size": RenameConfigConverter(("vocab_size",), ("vocab_size",)),
            "word_embeddings": IgnoredConfigConverter(("word_embeddings",)),
            "position_embeddings": IgnoredConfigConverter(("position_embeddings",), ("num_position_embeddings",)),
            "embedding_scale": ConstantImportConfigConverter(("embedding_scale",), 1.0),
        }

    @classmethod
    def _validate_export(cls, config: LanguageModelEmbeddingsConfig) -> None:
        Assert.incl(config.position_embeddings.enabled, (None, False))

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {"word_embeddings": WeightConverter("word_embeddings_weight", "embed_tokens.weight")}


class LlamaHeadConverter(ConfigSectionConverter):
    """Converts ``LanguageModelHeadConfig`` ↔ Llama final-norm fields (flat-merged)."""

    fast_llm_config_class = LanguageModelHeadConfig

    normalization_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = LlamaNormalizationConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "normalization": NestedConfigConverter(("normalization",), cls.normalization_converter_class),
            "output_weight": IgnoredConfigConverter(("output_weight",)),
            # Llama HF format does not represent ``prediction_heads``; pin to 1 so any non-default value
            # fails on export instead of silently round-tripping. MTP-Llama overrides this entry with a
            # ``RenameConfigConverter`` (the override replaces the parent's declaration in the returned
            # dict, so this ConstantImport never fires for MTP-Llama configs).
            "prediction_heads": ConstantImportConfigConverter(("prediction_heads",), 1),
            "final_logit_softcap": ConstantImportConfigConverter(("final_logit_softcap",), None),
        }

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        # ``final_norm`` reads the head's own ``normalization`` config; ``output_weights`` is the marker the
        # walker drops automatically when the root config has ``tied_embedding_weight=True``.
        return {
            "final_norm": NestedWeightConverter(
                "final_norm", "model.norm", cls.normalization_converter_class, config_attr="normalization"
            ),
            "output_weights": OutputProjectionWeightConverter("output_weights", "lm_head.weight"),
        }

    @classmethod
    def get_converters(
        cls,
        config: GPTBaseModelConfig,
    ) -> list[WeightConverter]:
        """Aggregator entry-point: the base-model converter passes the full :class:`GPTBaseModelConfig`
        so subclasses extending the head can read sibling sections (e.g. the decoder) when needed.
        Tied-embedding handling lives on :class:`OutputProjectionWeightConverter` and reads
        ``root_config.tied_embedding_weight``.
        """
        return cls.emit_weight_converters(config.head, "head", "", root_config=config)


class LlamaBaseModelConverter(HuggingFaceBaseModelConverter):
    """Top-level converter for ``GPTBaseModelConfig`` ↔ Llama HF dict.

    Subclasses (Mistral, Qwen2, Mixtral, MTP-Llama, …) override ``block_converter_class`` to plug their
    per-block declarations into the polymorphic Fixed/Pattern decoder dispatch held here.
    """

    fast_llm_config_class = GPTBaseModelConfig

    embeddings_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = LlamaEmbeddingsConverter
    block_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = LlamaBlockConverter
    head_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = LlamaHeadConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        block_converter_class = cls.block_converter_class

        def _decoder_export(parent: Config) -> dict:
            return {(k,): v for k, v in _llama_decoder_export(parent.decoder, block_converter_class).items()}

        def _decoder_import(hf_dict: dict) -> dict:
            return {
                ("decoder",): {
                    "block": block_converter_class.import_config(hf_dict),
                    "num_blocks": hf_dict["num_hidden_layers"],
                }
            }

        return {
            "embeddings": NestedConfigConverter(("embeddings",), cls.embeddings_converter_class),
            "head": NestedConfigConverter(("head",), cls.head_converter_class),
            "decoder": CustomConfigConverter(
                fast_llm_paths=(("decoder",),),
                # The block converter's flat-merge declarations claim all per-block top-level keys; pull
                # them up here so the HF coverage check sees them as covered. ``num_hidden_layers`` is
                # consumed by the Fixed/Pattern dispatch above.
                hf_paths=(
                    ("num_hidden_layers",),
                    *block_converter_class._consumed_hf_paths(),
                ),
                export_fn=_decoder_export,
                import_fn=_decoder_import,
                fast_llm_recurses=True,
            ),
            "hidden_size": RenameConfigConverter(("hidden_size",), ("hidden_size",)),
            "tied_embedding_weight": RenameConfigConverter(("tied_embedding_weight",), ("tie_word_embeddings",)),
            # Llama format cannot represent PEFT; the NoPeftConfig assertion lives on _validate_export so a
            # user-configured LoRA fails clearly rather than being silently dropped on export.
            "peft": IgnoredConfigConverter(("peft",)),
        }

    @classmethod
    def _validate_export(cls, config: GPTBaseModelConfig) -> None:
        assert_no_peft(config)

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        # ``head`` is added at the aggregator level (in :meth:`get_converters`) because the head
        # converter takes the full base-model config so subclasses extending the head can read
        # sibling sections.
        return {
            "embeddings": NestedWeightConverter("embeddings", "model", cls.embeddings_converter_class),
            "decoder": BlockSequenceWeightConverter("decoder", "model.layers", cls.block_converter_class),
        }

    @classmethod
    def get_converters(cls, config: GPTBaseModelConfig) -> list[WeightConverter]:
        return [
            *cls.emit_weight_converters(config, "", ""),
            *cls.head_converter_class.get_converters(config),
        ]


class LlamaHuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    _model: GPTModel
    _model_class: typing.ClassVar[FastLLMModelConfig] = GPTModelConfig
    format: typing.ClassVar[type[CheckpointFormat]] = LlamaCheckpointFormat
    architecture: typing.ClassVar[str] = "LlamaForCausalLM"
    base_model_converter_class: typing.ClassVar[type[LlamaBaseModelConverter]] = LlamaBaseModelConverter

    @classmethod
    def get_transformers_configuration_class(cls):
        import transformers

        return transformers.LlamaConfig
