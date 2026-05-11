import dataclasses
import logging
import typing

import torch
import transformers

from fast_llm.config import Config
from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    ConfigSectionConverter,
    ConstantImportConfigConverter,
    CustomConfigConverter,
    DefaultConfigConverter,
    IgnoredConfigConverter,
    IgnoreExportWeightConverter,
    IgnoreImportWeightConverter,
    NestedConfigConverter,
    RenameConfigConverter,
    SplitWeightConverter,
    WeightConverter,
)
from fast_llm.engine.checkpoint.huggingface import HuggingFaceBaseModelConverter, HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.attention.rotary.config import DefaultRotaryConfig, Llama3RotaryConfig, YarnRotaryConfig
from fast_llm.layers.block.config import FixedBlockSequenceConfig, PatternBlockSequenceConfig
from fast_llm.layers.common.normalization.config import RMSNormalizationConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.layers.decoder.mlp.config import MLPConfig
from fast_llm.layers.language_model.config import (
    LanguageModelConfig,
    LanguageModelEmbeddingsConfig,
    LanguageModelHeadConfig,
)
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.models.gpt.conversion.config import LlamaCheckpointFormat
from fast_llm.models.gpt.model import GPTModel
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert, div

_TRANSFORMERS_V4 = not dataclasses.is_dataclass(transformers.PretrainedConfig)

logger = logging.getLogger(__name__)


def assert_no_peft(config: GPTBaseModelConfig) -> None:
    """Reject any non-trivial PEFT config: HuggingFace formats serialize the base weights only,
    so a configured LoRA (or other adapter) would be silently dropped on export."""
    from fast_llm.layers.common.peft.config import NoPeftConfig

    Assert.custom(isinstance, config.peft, NoPeftConfig)


def effective_bias(layer_config, default: bool) -> bool:
    """Resolve a layer's effective bias flag: explicit ``bias.enabled`` if set, else the parent default."""
    return default if layer_config.bias.enabled is None else layer_config.bias.enabled


# ============================================================
# Weight converters (imperative — kept as-is during config migration)
# ============================================================


def get_parameter_converter(
    fast_llm_name: str | tuple[str, ...],
    hf_name: str | tuple[str, ...],
    cls=WeightConverter,
    config=None,
    drop_on_export: bool = False,
    drop_on_import: bool = False,
) -> WeightConverter:
    if isinstance(fast_llm_name, str):
        fast_llm_name = (fast_llm_name,)
    if isinstance(hf_name, str):
        hf_name = (hf_name,)
    if drop_on_export:
        cls = IgnoreExportWeightConverter
    if drop_on_import:
        cls = IgnoreImportWeightConverter
    return cls(
        () if drop_on_import else fast_llm_name,
        () if drop_on_export else hf_name,
        config,
    )


def get_weight_and_bias_converters(
    fast_llm_prefix: str | tuple[str, ...],
    hf_prefix: str | tuple[str, ...],
    use_bias: bool,
    cls=WeightConverter,
    config=None,
    drop_on_export: bool = False,
    drop_on_import: bool = False,
) -> list[WeightConverter]:
    if isinstance(fast_llm_prefix, str):
        fast_llm_prefix = (fast_llm_prefix,)
    if isinstance(hf_prefix, str):
        hf_prefix = (hf_prefix,)
    converters = [
        get_parameter_converter(
            () if drop_on_import else tuple(f"{prefix}.weight" for prefix in fast_llm_prefix),
            () if drop_on_export else tuple(f"{prefix}.weight" for prefix in hf_prefix),
            cls,
            config,
            drop_on_export,
            drop_on_import,
        )
    ]
    if use_bias:
        converters.append(
            get_parameter_converter(
                () if drop_on_import else tuple(f"{prefix}.bias" for prefix in fast_llm_prefix),
                () if drop_on_export else tuple(f"{prefix}.bias" for prefix in hf_prefix),
                cls,
                config,
                drop_on_export,
                drop_on_import,
            )
        )
    return converters


class MLPLayer2Converter(WeightConverter):
    # Similar to SplitWeightConverter, but handles the optional MLP transpose.
    # Still ok for non-gated (trivial split) and biases (trivial 1d transpose)

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
    # Hf uses the real format for rotary embeddings, and keeps the key and value separate.
    _config: AttentionConfig

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

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(
        cls,
        config: RMSNormalizationConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return get_weight_and_bias_converters(
            fast_llm_prefix,
            () if drop_on_export else hf_prefix,
            False,
            IgnoreExportWeightConverter if drop_on_export else WeightConverter,
        )


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
        }

    @classmethod
    def _validate_export(cls, config: MLPConfig) -> None:
        Assert.incl(config.layer_1.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.layer_2.bias.enabled, (None, config.add_linear_biases))

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(
        cls,
        config: MLPConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_1",
                (f"{hf_prefix}.gate_proj", f"{hf_prefix}.up_proj"),
                config.add_linear_biases,
                SplitWeightConverter,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_2",
                f"{hf_prefix}.down_proj",
                config.add_linear_biases,
                MLPLayer2Converter,
                drop_on_export=drop_on_export,
            ),
        ]


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
                recurses=True,
            ),
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

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(
        cls,
        config: AttentionConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.query",
                f"{hf_prefix}.q_proj",
                config.add_linear_biases,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.key_value",
                (f"{hf_prefix}.k_proj", f"{hf_prefix}.v_proj"),
                config.add_linear_biases,
                KeyValueWeightConverter,
                config,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.dense",
                f"{hf_prefix}.o_proj",
                config.add_linear_biases,
                drop_on_export=drop_on_export,
            ),
        ]


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
        }

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(
        cls, config: DecoderBlockConfig, fast_llm_prefix: str, hf_prefix: str, drop_on_export: bool = False
    ) -> list[WeightConverter]:
        return [
            *cls.mixer_converter_class.get_converters(
                config.mixer,
                f"{fast_llm_prefix}.mixer",
                f"{hf_prefix}.{cls.hf_mixer_name}",
                drop_on_export,
            ),
            *cls.mlp_converter_class.get_converters(
                config.mlp,
                f"{fast_llm_prefix}.mlp",
                f"{hf_prefix}.{cls.hf_mlp_name}",
                drop_on_export,
            ),
            *cls.normalization_converter_class.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.norm_1",
                f"{hf_prefix}.{cls.hf_norm_1_name}",
                drop_on_export,
            ),
            *cls.normalization_converter_class.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.norm_2",
                f"{hf_prefix}.{cls.hf_norm_2_name}",
                drop_on_export,
            ),
        ]


class LlamaDecoderConverter:
    """Converts ``BlockSequenceConfig`` (polymorphic Fixed/Pattern) ↔ Llama's flat block + ``num_hidden_layers``.

    Kept as a regular class (not a :class:`ConfigSectionConverter`) so it can stay imperative — the polymorphism
    between Fixed/Pattern block sequences doesn't lend itself to the declarative shape, and subclasses (Mistral,
    Qwen2, MTP-Llama, ...) plug in different block converters via ``block_converter_class``.
    """

    block_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = LlamaBlockConverter

    @classmethod
    def import_config(cls, hf_dict: dict) -> dict:
        return {
            "block": cls.block_converter_class.import_config(hf_dict),
            "num_blocks": hf_dict["num_hidden_layers"],
        }

    @classmethod
    def export_config(cls, decoder_config: FixedBlockSequenceConfig | PatternBlockSequenceConfig) -> dict:
        if isinstance(decoder_config, PatternBlockSequenceConfig):
            exports = [cls.block_converter_class.export_config(block) for block in decoder_config.blocks.values()]
            for other in exports[1:]:
                Assert.eq(exports[0], other)
            block_hf = exports[0]
        elif isinstance(decoder_config, FixedBlockSequenceConfig):
            block_hf = cls.block_converter_class.export_config(decoder_config.block)
        else:
            raise NotImplementedError(f"Unsupported decoder type: {type(decoder_config).__name__}")
        return {**block_hf, "num_hidden_layers": decoder_config.num_blocks}

    @classmethod
    def get_converters(
        cls,
        config: FixedBlockSequenceConfig | PatternBlockSequenceConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        block_config = (
            config.block if isinstance(config, FixedBlockSequenceConfig) else next(iter(config.blocks.values()))
        )
        converters: list[WeightConverter] = []
        for block_index in range(config.num_blocks):
            converters += cls.block_converter_class.get_converters(
                block_config,
                f"{fast_llm_prefix}.{block_index}",
                f"{hf_prefix}.{block_index}",
                drop_on_export,
            )
        return converters


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
        }

    @classmethod
    def _validate_export(cls, config: LanguageModelEmbeddingsConfig) -> None:
        Assert.incl(config.position_embeddings.enabled, (None, False))

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(
        cls, config: LanguageModelEmbeddingsConfig, fast_llm_prefix: str, hf_prefix: str
    ) -> list[WeightConverter]:
        return [WeightConverter(f"{fast_llm_prefix}.word_embeddings_weight", f"{hf_prefix}.embed_tokens.weight")]


class LlamaHeadConverter(ConfigSectionConverter):
    """Converts ``LanguageModelHeadConfig`` ↔ Llama final-norm fields (flat-merged)."""

    fast_llm_config_class = LanguageModelHeadConfig

    normalization_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = LlamaNormalizationConverter
    # Used by MTP-Llama subclass to emit per-prediction-head block weight converters; Llama itself doesn't read it.
    block_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = LlamaBlockConverter

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
        }

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(
        cls,
        config: LanguageModelConfig,
        exported_config: dict,
    ) -> list[WeightConverter]:
        return [
            *cls.normalization_converter_class.get_converters(
                config.head.normalization,
                f"head.final_norm",
                f"model.norm",
            ),
            get_parameter_converter(
                f"head.output_weights",
                "lm_head.weight",
                drop_on_import=exported_config["tie_word_embeddings"],
                drop_on_export=exported_config["tie_word_embeddings"],
            ),
        ]


class LlamaBaseModelConverter(ConfigSectionConverter, HuggingFaceBaseModelConverter):
    """Top-level converter for ``GPTBaseModelConfig`` ↔ Llama HF dict."""

    fast_llm_config_class = GPTBaseModelConfig

    # TODO: Peft?
    decoder_converter_class: typing.ClassVar[type[LlamaDecoderConverter]] = LlamaDecoderConverter
    embeddings_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = LlamaEmbeddingsConverter
    head_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = LlamaHeadConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        decoder_converter_class = cls.decoder_converter_class

        def _decoder_export(parent: Config) -> dict:
            return {(k,): v for k, v in decoder_converter_class.export_config(parent.decoder).items()}

        def _decoder_import(hf_dict: dict) -> dict:
            return {("decoder",): decoder_converter_class.import_config(hf_dict)}

        return {
            "embeddings": NestedConfigConverter(("embeddings",), cls.embeddings_converter_class),
            "head": NestedConfigConverter(("head",), cls.head_converter_class),
            "decoder": CustomConfigConverter(
                fast_llm_paths=(("decoder",),),
                # The Custom wraps the imperative LlamaDecoderConverter, which delegates to
                # cls.decoder_converter_class.block_converter_class (a ConfigSectionConverter). The
                # block converter's flat-merge declarations claim all per-block top-level keys; pull
                # them up here so the HF coverage check sees them as covered. ``num_hidden_layers``
                # is consumed by LlamaDecoderConverter itself.
                hf_paths=(
                    ("num_hidden_layers",),
                    *cls.decoder_converter_class.block_converter_class._consumed_hf_paths(),
                ),
                export_fn=_decoder_export,
                import_fn=_decoder_import,
                recurses=True,
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

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(cls, config: GPTBaseModelConfig, exported_config: dict) -> list[WeightConverter]:
        return [
            *cls.embeddings_converter_class.get_converters(config.embeddings, "embeddings", "model"),
            *cls.decoder_converter_class.get_converters(config.decoder, "decoder", "model.layers"),
            *cls.head_converter_class.get_converters(config, exported_config),
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
