import logging
import typing

import torch

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    IgnoreExportWeightConverter,
    IgnoreImportWeightConverter,
    SplitWeightConverter,
    WeightConverter,
)
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.attention.rotary.config import DefaultRotaryConfig, Llama3RotaryConfig, YarnRotaryConfig
from fast_llm.layers.attention.rotary.rotary import convert_rotary_complex_to_real, convert_rotary_real_to_complex
from fast_llm.layers.block.config import FixedBlockSequenceConfig
from fast_llm.layers.common.normalization.config import RMSNormalizationConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.layers.decoder.mlp.config import MLPConfig
from fast_llm.layers.language_model.config import LanguageModelEmbeddingsConfig, LanguageModelHeadConfig
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.models.gpt.conversion.config import LlamaCheckpointFormat
from fast_llm.models.gpt.model import GPTModel
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert, div, safe_merge_dicts

logger = logging.getLogger(__name__)


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


class LlamaNormalizationConverter:
    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {"type": "rms_norm", "epsilon": config["rms_norm_eps"]}

    @classmethod
    def export_config(cls, config: RMSNormalizationConfig) -> dict:
        Assert.custom(isinstance, config, RMSNormalizationConfig)
        assert not config.zero_centered
        return {"rms_norm_eps": config.epsilon}

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


class LlamaMLPConverter:
    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            "intermediate_size": config["intermediate_size"],
            "add_linear_biases": config["mlp_bias"],
            "activation": ActivationType.from_hf_name(config["hidden_act"]),
            "gated": True,
        }

    @classmethod
    def export_config(cls, config: MLPConfig) -> dict:
        Assert.custom(isinstance, config, MLPConfig)
        Assert.incl(config.layer_1.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.layer_2.bias.enabled, (None, config.add_linear_biases))
        assert config.gated
        return {
            "intermediate_size": config.intermediate_size,
            "mlp_bias": config.add_linear_biases,
            "hidden_act": config.activation.hf_name,
        }

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


class LlamaAttentionConverter:
    @classmethod
    def import_config(cls, config: dict) -> dict:
        try:
            rope_type = config["rope_scaling"]["rope_type"]
        except (KeyError, TypeError):
            rope_type = "default"
        rotary_config = {
            "type": rope_type,
            "theta": config["rope_theta"],
        }
        if rope_type == "default":
            pass
        elif rope_type == "llama3":
            rotary_config.update(
                {
                    "scale_factor": config["factor"],
                    "low_frequency_factor": config["low_freq_factor"],
                    "high_frequency_factor": config["high_freq_factor"],
                    "original_context_length": config["original_max_position_embeddings"],
                }
            )
        elif rope_type == "yarn":
            rotary_config.update(
                {
                    "attention_factor": config["attention_factor"],
                    "beta_fast": config["beta_fast"],
                    "beta_slow": config["beta_slow"],
                    "original_context_length": config["original_max_position_embeddings"],
                }
            )
        else:
            raise NotImplementedError(f"Unsupported rotary type: {type(config.rotary).__name__}")
        out = {
            "rotary": rotary_config,
            "heads": config["num_attention_heads"],
            "head_groups": config["num_key_value_heads"],
            "head_size": config.get("head_dim"),
            "add_linear_biases": config["attention_bias"],
            "dropout": config["attention_dropout"],
        }
        if out["head_size"] is None:
            out["head_size"] = div(config["hidden_size"], out["heads"])

        return out

    @classmethod
    def export_config(cls, config: AttentionConfig) -> dict:
        cls._check_config(config)
        Assert.eq(config.softmax_scale_power, 0.5)
        out = {
            "num_attention_heads": config.heads,
            "num_key_value_heads": config.head_groups,
            "head_dim": config.head_size,
            "attention_bias": config.add_linear_biases,
            "attention_dropout": config.dropout,
            "rope_theta": config.rotary.theta,
        }
        if type(config.rotary) is DefaultRotaryConfig:
            pass
        elif type(config.rotary) is Llama3RotaryConfig:
            out["rope_scaling"] = {
                "rope_type": "llama3",
                "factor": config.rotary.scale_factor,
                "low_freq_factor": config.rotary.low_frequency_factor,
                "high_freq_factor": config.rotary.high_frequency_factor,
                "original_max_position_embeddings": config.rotary.original_context_length,
            }
        elif type(config.rotary) is YarnRotaryConfig:
            out["rope_scaling"] = {
                "rope_type": "yarn",
                "attention_factor": config.rotary.attention_factor,
                "beta_fast": config.rotary.beta_fast,
                "beta_slow": config.rotary.beta_slow,
                "original_max_position_embeddings": config.rotary.original_context_length,
            }
        else:
            raise NotImplementedError(f"Unsupported rotary type: {type(config.rotary).__name__}")

        return out

    @classmethod
    def _check_config(cls, config: AttentionConfig) -> None:
        # Opportunity to make derived classes less constrained.
        Assert.is_(type(config), AttentionConfig)
        Assert.incl(config.query_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.key_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.value_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.dense_layer.bias.enabled, (None, config.add_linear_biases))

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
                QueryWeightConverter,
                config,
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


class QueryWeightConverter(WeightConverter):
    # Hf uses the real format for rotary embeddings.
    _config: AttentionConfig

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (query,) = weight
        if self._config.rotary.complex_format:
            query = convert_rotary_complex_to_real(query[:], self._config.head_size, 0)
        return (query,)

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (query,) = weight
        if self._config.rotary.complex_format:
            query = convert_rotary_real_to_complex(query[:], self._config.head_size, 0)
        return (query,)


class KeyValueWeightConverter(WeightConverter):
    # Hf uses the real format for rotary embeddings, and keeps the key and value separate.
    _config: AttentionConfig

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (key_value,) = weight
        key, value = key_value[:].chunk(2)
        if self._config.rotary.complex_format:
            key = convert_rotary_complex_to_real(key, self._config.head_size, 0)
        return key, value

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        key, value = weight
        if self._config.rotary.complex_format:
            key = convert_rotary_real_to_complex(key[:], self._config.head_size, 0)
        key_value = torch.cat([key[:], value[:]])
        return (key_value,)


class LlamaBlockConverter:
    mixer_converter_class: typing.ClassVar[type[LlamaAttentionConverter]] = LlamaAttentionConverter
    mlp_converter_class: typing.ClassVar[type[LlamaMLPConverter]] = LlamaMLPConverter
    normalization_converter_class: typing.ClassVar[type[LlamaNormalizationConverter]] = LlamaNormalizationConverter
    hf_mixer_name: typing.ClassVar[str] = "self_attn"
    hf_mlp_name: typing.ClassVar[str] = "mlp"
    hf_norm_1_name: typing.ClassVar[str] = "input_layernorm"
    hf_norm_2_name: typing.ClassVar[str] = "post_attention_layernorm"

    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            "mixer": cls.mixer_converter_class.import_config(config),
            "mlp": cls.mlp_converter_class.import_config(config),
            "normalization": cls.normalization_converter_class.import_config(config),
        }

    @classmethod
    def export_config(cls, config: DecoderBlockConfig) -> dict:
        Assert.custom(isinstance, config, DecoderBlockConfig)
        return safe_merge_dicts(
            cls.mixer_converter_class.export_config(config.mixer),
            cls.mlp_converter_class.export_config(config.mlp),
            cls.normalization_converter_class.export_config(config.normalization),
        )

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
    block_converter_class: typing.ClassVar[type[LlamaBlockConverter]] = LlamaBlockConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            "block": cls.block_converter_class.import_config(config),
            "num_blocks": config["num_hidden_layers"],
        }

    @classmethod
    def export_config(cls, config: FixedBlockSequenceConfig) -> dict:
        # TODO: Support PatternBlockSequenceConfig with compatible configs.
        Assert.custom(isinstance, config, FixedBlockSequenceConfig)
        return safe_merge_dicts(
            cls.block_converter_class.export_config(config.block),
            {"num_hidden_layers": config.num_blocks},
        )

    @classmethod
    def get_converters(
        cls,
        config: FixedBlockSequenceConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        converters = []
        for block_index in range(config.num_blocks):
            converters += cls.block_converter_class.get_converters(
                config.block,
                f"{fast_llm_prefix}.{block_index}",
                f"{hf_prefix}.{block_index}",
                drop_on_export,
            )
        return converters


class LlamaEmbeddingsConverter:
    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            "vocab_size": config["vocab_size"],
            "hidden_size": config["hidden_size"],
        }

    @classmethod
    def export_config(cls, config: LanguageModelEmbeddingsConfig) -> dict:
        Assert.custom(isinstance, config, LanguageModelEmbeddingsConfig)
        assert not config.position_embeddings.enabled
        return {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
        }

    @classmethod
    def get_converters(
        cls, config: LanguageModelEmbeddingsConfig, fast_llm_prefix: str, hf_prefix: str
    ) -> list[WeightConverter]:
        return [WeightConverter(f"{fast_llm_prefix}.word_embeddings_weight", f"{hf_prefix}.embed_tokens.weight")]


class LlamaHeadConverter:
    normalization_converter_class: typing.ClassVar[type[LlamaNormalizationConverter]] = LlamaNormalizationConverter
    block_converter_class: typing.ClassVar[type[LlamaBlockConverter]] = LlamaBlockConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {"normalization": cls.normalization_converter_class.import_config(config)}

    @classmethod
    def export_config(cls, config: LanguageModelHeadConfig) -> dict:
        Assert.custom(isinstance, config, LanguageModelHeadConfig)
        return cls.normalization_converter_class.export_config(config.normalization)

    @classmethod
    def get_converters(
        cls,
        config: LanguageModelHeadConfig,
        exported_config: dict,
        fast_llm_prefix: str,
    ) -> list[WeightConverter]:
        return [
            *cls.normalization_converter_class.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.final_norm",
                f"model.norm",
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.output_weights",
                "lm_head.weight",
                drop_on_import=exported_config["tie_word_embeddings"],
            ),
        ]


class LlamaBaseModelConverter:
    # TODO: Peft?
    decoder_converter_class: typing.ClassVar[type[LlamaDecoderConverter]] = LlamaDecoderConverter
    embeddings_converter_class: typing.ClassVar[type[LlamaEmbeddingsConverter]] = LlamaEmbeddingsConverter
    head_converter_class: typing.ClassVar[type[LlamaHeadConverter]] = LlamaHeadConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            "embeddings": cls.embeddings_converter_class.import_config(config),
            "decoder": cls.decoder_converter_class.import_config(config),
            "head": cls.head_converter_class.import_config(config),
            "tied_embedding_weight": config["tie_word_embeddings"],
        }

    @classmethod
    def export_config(cls, config: GPTBaseModelConfig) -> dict:
        Assert.custom(isinstance, config, GPTBaseModelConfig)
        return safe_merge_dicts(
            cls.embeddings_converter_class.export_config(config.embeddings),
            cls.decoder_converter_class.export_config(config.decoder),
            cls.head_converter_class.export_config(config.head),
            {"tie_word_embeddings": config.tied_embedding_weight},
        )

    @classmethod
    def get_converters(cls, config: GPTBaseModelConfig, exported_config: dict) -> list[WeightConverter]:
        return [
            *cls.embeddings_converter_class.get_converters(config.embeddings, "embeddings", "model"),
            *cls.decoder_converter_class.get_converters(config.decoder, "decoder", "model.layers"),
            *cls.head_converter_class.get_converters(config.head, exported_config, "head"),
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
