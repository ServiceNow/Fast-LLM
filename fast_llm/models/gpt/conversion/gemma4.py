import typing

import torch

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import SplitWeightConverter, WeightConverter
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.attention.rotary.config import DefaultRotaryConfig, ProportionalRotaryConfig
from fast_llm.layers.block.config import PatternBlockSequenceConfig
from fast_llm.layers.common.normalization.config import RMSNormalizationConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.layers.decoder.mlp.config import Gemma4MoEMLPConfig
from fast_llm.layers.language_model.config import LanguageModelEmbeddingsConfig, LanguageModelHeadConfig
from fast_llm.models.gpt.config import GPTBaseModelConfig
from fast_llm.models.gpt.conversion.config import Gemma4CheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    KeyValueWeightConverter,
    LlamaBaseModelConverter,
    LlamaEmbeddingsConverter,
    LlamaHeadConverter,
    LlamaHuggingfaceCheckpointHandler,
    LlamaNormalizationConverter,
    MLPLayer2Converter,
    QueryWeightConverter,
    get_parameter_converter,
    get_weight_and_bias_converters,
)
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert, safe_merge_dicts


class Gemma4AttentionConverter:
    @classmethod
    def import_config(cls, config: dict, *, full_attention: bool) -> dict:
        common = {
            "heads": config["num_attention_heads"],
            "add_linear_biases": False,
            "dropout": config.get("attention_dropout", 0.0),
            "implementation": "backup",
            "softmax_scale_power": 0.0,
            "query_norm": Gemma4NormalizationConverter.import_config(config),
            "key_norm": Gemma4NormalizationConverter.import_config(config),
            "value_norm": True,
            "value_norm_eps": config["rms_norm_eps"],
        }
        if full_attention:
            rope_parameters = cls._get_rope_parameters(config, "full_attention")
            return safe_merge_dicts(
                common,
                {
                    "head_groups": config["num_global_key_value_heads"],
                    "head_size": config["global_head_dim"],
                    "rotary": {
                        "type": "proportional",
                        "theta": rope_parameters.get("rope_theta", 1_000_000),
                        "partial_rotary_factor": rope_parameters.get("partial_rotary_factor", 0.25),
                    },
                    "attention_k_eq_v": config.get("attention_k_eq_v", False),
                },
            )
        rope_parameters = cls._get_rope_parameters(config, "sliding_attention")
        return safe_merge_dicts(
            common,
            {
                "head_groups": config["num_key_value_heads"],
                "head_size": config["head_dim"],
                "rotary": {"type": "default", "theta": rope_parameters.get("rope_theta", 10_000)},
                "window_size": config["sliding_window"],
                "attention_k_eq_v": False,
            },
        )

    @classmethod
    def export_config(cls, config: AttentionConfig, *, full_attention: bool) -> dict:
        Assert.custom(isinstance, config, AttentionConfig)
        assert not config.add_linear_biases
        Assert.custom(isinstance, config.query_norm, RMSNormalizationConfig)
        Assert.custom(isinstance, config.key_norm, RMSNormalizationConfig)
        assert config.value_norm
        common = {
            "num_attention_heads": config.heads,
            "attention_dropout": config.dropout,
            "rms_norm_eps": config.query_norm.epsilon,
        }
        if full_attention:
            Assert.custom(isinstance, config.rotary, ProportionalRotaryConfig)
            Assert.eq(config.window_size, None)
            return safe_merge_dicts(
                common,
                {
                    "num_global_key_value_heads": config.head_groups,
                    "global_head_dim": config.head_size,
                    "_full_rope_parameters": {
                        "rope_type": "proportional",
                        "rope_theta": config.rotary.theta,
                        "partial_rotary_factor": config.rotary.partial_rotary_factor,
                    },
                    "attention_k_eq_v": config.attention_k_eq_v,
                },
            )
        Assert.custom(isinstance, config.rotary, DefaultRotaryConfig)
        assert not config.attention_k_eq_v
        return safe_merge_dicts(
            common,
            {
                "num_key_value_heads": config.head_groups,
                "head_dim": config.head_size,
                "_sliding_rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": config.rotary.theta,
                },
                "sliding_window": config.window_size,
            },
        )

    @classmethod
    def _get_rope_parameters(cls, config: dict, attention_type: str) -> dict:
        if rope_parameters := config.get("rope_parameters"):
            return rope_parameters[attention_type]
        if attention_type == "full_attention":
            return {
                "rope_type": "proportional",
                "rope_theta": config.get("global_rope_theta", 1_000_000),
                "partial_rotary_factor": config.get("partial_rotary_factor", 0.25),
            }
        return {"rope_type": "default", "rope_theta": config.get("rope_theta", 10_000)}

    @classmethod
    def get_converters(
        cls,
        config: AttentionConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        key_value_converter = (
            get_weight_and_bias_converters(
                f"{fast_llm_prefix}.key_value",
                f"{hf_prefix}.k_proj",
                False,
                drop_on_export=drop_on_export,
            )
            if config.attention_k_eq_v
            else get_weight_and_bias_converters(
                f"{fast_llm_prefix}.key_value",
                (f"{hf_prefix}.k_proj", f"{hf_prefix}.v_proj"),
                False,
                KeyValueWeightConverter,
                config,
                drop_on_export=drop_on_export,
            )
        )
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.query",
                f"{hf_prefix}.q_proj",
                False,
                QueryWeightConverter,
                config,
                drop_on_export=drop_on_export,
            ),
            *key_value_converter,
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.dense",
                f"{hf_prefix}.o_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *Gemma4NormalizationConverter.get_converters(
                config.query_norm,
                f"{fast_llm_prefix}.q_norm",
                f"{hf_prefix}.q_norm",
                drop_on_export,
            ),
            *Gemma4NormalizationConverter.get_converters(
                config.key_norm,
                f"{fast_llm_prefix}.k_norm",
                f"{hf_prefix}.k_norm",
                drop_on_export,
            ),
        ]


class Gemma4NormalizationConverter(LlamaNormalizationConverter):
    pass


class Gemma4ExpertLayer1Converter(WeightConverter):
    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (expert_layer_1,) = weight
        return (expert_layer_1[:].unflatten(0, (self._config.experts, -1)).contiguous(),)

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (gate_up_proj,) = weight
        return (gate_up_proj[:].flatten(0, 1).contiguous(),)


class Gemma4ExpertLayer2Converter(WeightConverter):
    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (expert_layer_2,) = weight
        return (
            expert_layer_2[:]
            .unflatten(0, (self._config.experts, self._config.moe_intermediate_size))
            .transpose(-1, -2)
            .contiguous(),
        )

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (down_proj,) = weight
        return (down_proj[:].transpose(-1, -2).flatten(0, 1).contiguous(),)


class Gemma4MLPConverter:
    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            "type": "gemma4_moe",
            "intermediate_size": config["intermediate_size"],
            "moe_intermediate_size": config["moe_intermediate_size"],
            "experts": config["num_experts"],
            "experts_per_token": config["top_k_experts"],
            "add_linear_biases": False,
            "activation": ActivationType.from_hf_name(config.get("hidden_activation", config.get("hidden_act"))),
            "gated": True,
            "router_norm_eps": config["rms_norm_eps"],
            "recompute_level": "full",
            "post_feedforward_norm_1": Gemma4NormalizationConverter.import_config(config),
            "pre_feedforward_norm_2": Gemma4NormalizationConverter.import_config(config),
            "post_feedforward_norm_2": Gemma4NormalizationConverter.import_config(config),
        }

    @classmethod
    def export_config(cls, config: Gemma4MoEMLPConfig) -> dict:
        Assert.custom(isinstance, config, Gemma4MoEMLPConfig)
        assert not config.add_linear_biases
        assert config.gated
        return {
            "intermediate_size": config.intermediate_size,
            "moe_intermediate_size": config.moe_intermediate_size,
            "num_experts": config.experts,
            "top_k_experts": config.experts_per_token,
            "hidden_activation": config.activation.hf_name,
            "enable_moe_block": True,
        }

    @classmethod
    def get_converters(
        cls,
        config: Gemma4MoEMLPConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_1",
                (f"{hf_prefix}.mlp.gate_proj", f"{hf_prefix}.mlp.up_proj"),
                False,
                SplitWeightConverter,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_2",
                f"{hf_prefix}.mlp.down_proj",
                False,
                MLPLayer2Converter,
                drop_on_export=drop_on_export,
            ),
            *Gemma4NormalizationConverter.get_converters(
                config.post_feedforward_norm_1,
                f"{fast_llm_prefix}.post_feedforward_norm_1",
                f"{hf_prefix}.post_feedforward_layernorm_1",
                drop_on_export,
            ),
            *Gemma4NormalizationConverter.get_converters(
                config.pre_feedforward_norm_2,
                f"{fast_llm_prefix}.pre_feedforward_norm_2",
                f"{hf_prefix}.pre_feedforward_layernorm_2",
                drop_on_export,
            ),
            *Gemma4NormalizationConverter.get_converters(
                config.post_feedforward_norm_2,
                f"{fast_llm_prefix}.post_feedforward_norm_2",
                f"{hf_prefix}.post_feedforward_layernorm_2",
                drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.router",
                f"{hf_prefix}.router.proj",
                False,
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.router_scale",
                f"{hf_prefix}.router.scale",
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.per_expert_scale",
                f"{hf_prefix}.router.per_expert_scale",
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.expert_layer_1.weight",
                f"{hf_prefix}.experts.gate_up_proj",
                Gemma4ExpertLayer1Converter,
                config,
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.expert_layer_2.weight",
                f"{hf_prefix}.experts.down_proj",
                Gemma4ExpertLayer2Converter,
                config,
                drop_on_export=drop_on_export,
            ),
        ]


class Gemma4BlockConverter:
    @classmethod
    def import_config(cls, config: dict, *, full_attention: bool) -> dict:
        return {
            "mixer": Gemma4AttentionConverter.import_config(config, full_attention=full_attention),
            "mlp": Gemma4MLPConverter.import_config(config),
            "normalization": Gemma4NormalizationConverter.import_config(config),
            "post_mixer_normalization": Gemma4NormalizationConverter.import_config(config),
            "post_mlp_normalization": Gemma4NormalizationConverter.import_config(config),
            "layer_scalar": {"enabled": True},
        }

    @classmethod
    def export_config(cls, config: DecoderBlockConfig, *, full_attention: bool) -> dict:
        Assert.custom(isinstance, config, DecoderBlockConfig)
        return safe_merge_dicts(
            Gemma4AttentionConverter.export_config(config.mixer, full_attention=full_attention),
            Gemma4MLPConverter.export_config(config.mlp),
            Gemma4NormalizationConverter.export_config(config.normalization),
        )

    @classmethod
    def get_converters(
        cls,
        config: DecoderBlockConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return [
            *Gemma4AttentionConverter.get_converters(
                config.mixer,
                f"{fast_llm_prefix}.mixer",
                f"{hf_prefix}.self_attn",
                drop_on_export,
            ),
            *Gemma4MLPConverter.get_converters(
                config.mlp,
                f"{fast_llm_prefix}.mlp",
                hf_prefix,
                drop_on_export,
            ),
            *Gemma4NormalizationConverter.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.norm_1",
                f"{hf_prefix}.input_layernorm",
                drop_on_export,
            ),
            *Gemma4NormalizationConverter.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.norm_2",
                f"{hf_prefix}.pre_feedforward_layernorm",
                drop_on_export,
            ),
            *Gemma4NormalizationConverter.get_converters(
                config.post_mixer_normalization,
                f"{fast_llm_prefix}.post_mixer_normalization",
                f"{hf_prefix}.post_attention_layernorm",
                drop_on_export,
            ),
            *Gemma4NormalizationConverter.get_converters(
                config.post_mlp_normalization,
                f"{fast_llm_prefix}.post_mlp_normalization",
                f"{hf_prefix}.post_feedforward_layernorm",
                drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.layer_scalar",
                f"{hf_prefix}.layer_scalar",
                drop_on_export=drop_on_export,
            ),
        ]


class Gemma4DecoderConverter:
    block_converter_class: typing.ClassVar[type[Gemma4BlockConverter]] = Gemma4BlockConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        pattern = cls._get_pattern(config)
        return {
            "type": "pattern",
            "blocks": {
                "sliding": cls.block_converter_class.import_config(config, full_attention=False),
                "full": cls.block_converter_class.import_config(config, full_attention=True),
            },
            "pattern": pattern,
            "num_blocks": config["num_hidden_layers"],
        }

    @classmethod
    def export_config(cls, config: PatternBlockSequenceConfig) -> dict:
        Assert.custom(isinstance, config, PatternBlockSequenceConfig)
        out = safe_merge_dicts(
            cls.block_converter_class.export_config(config.blocks["sliding"], full_attention=False),
            cls.block_converter_class.export_config(config.blocks["full"], full_attention=True),
            {
                "num_hidden_layers": config.num_blocks,
                "layer_types": [
                    "full_attention" if block_name == "full" else "sliding_attention"
                    for block_name in config.expanded_pattern
                ],
            },
        )
        out["rope_parameters"] = {
            "sliding_attention": out.pop("_sliding_rope_parameters"),
            "full_attention": out.pop("_full_rope_parameters"),
        }
        return out

    @classmethod
    def get_converters(
        cls,
        config: PatternBlockSequenceConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        Assert.custom(isinstance, config, PatternBlockSequenceConfig)
        converters = []
        for block_index, block_name in enumerate(config.expanded_pattern):
            converters += cls.block_converter_class.get_converters(
                config.blocks[block_name],
                f"{fast_llm_prefix}.{block_index}",
                f"{hf_prefix}.{block_index}",
                drop_on_export,
            )
        return converters

    @classmethod
    def _get_pattern(cls, config: dict) -> list[str]:
        if layer_types := config.get("layer_types"):
            return [cls._normalize_layer_type(layer_type) for layer_type in layer_types]
        return ["sliding", "sliding", "sliding", "sliding", "sliding", "full"]

    @classmethod
    def _normalize_layer_type(cls, layer_type: str) -> str:
        if "sliding" in layer_type:
            return "sliding"
        if "full" in layer_type or "global" in layer_type:
            return "full"
        raise NotImplementedError(f"Unsupported Gemma4 layer type: {layer_type}")


class Gemma4EmbeddingsConverter(LlamaEmbeddingsConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        return safe_merge_dicts(super().import_config(config), {"scale_by_sqrt_hidden_size": True})

    @classmethod
    def export_config(cls, config: LanguageModelEmbeddingsConfig) -> dict:
        Assert.custom(isinstance, config, LanguageModelEmbeddingsConfig)
        assert config.scale_by_sqrt_hidden_size
        return super().export_config(config)


class Gemma4HeadConverter(LlamaHeadConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        return safe_merge_dicts(
            super().import_config(config),
            {"final_logit_softcap": config.get("final_logit_softcapping")},
        )

    @classmethod
    def export_config(cls, config: LanguageModelHeadConfig) -> dict:
        return safe_merge_dicts(
            super().export_config(config),
            {"final_logit_softcapping": config.final_logit_softcap},
        )


class Gemma4BaseModelConverter(LlamaBaseModelConverter):
    decoder_converter_class: typing.ClassVar[type[Gemma4DecoderConverter]] = Gemma4DecoderConverter
    embeddings_converter_class: typing.ClassVar[type[Gemma4EmbeddingsConverter]] = Gemma4EmbeddingsConverter
    head_converter_class: typing.ClassVar[type[Gemma4HeadConverter]] = Gemma4HeadConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        cls._check_supported_config(config)
        return {
            "embeddings": cls.embeddings_converter_class.import_config(config),
            "decoder": cls.decoder_converter_class.import_config(config),
            "head": cls.head_converter_class.import_config(config),
            "hidden_size": config["hidden_size"],
            "tied_embedding_weight": config["tie_word_embeddings"],
        }

    @classmethod
    def export_config(cls, config: GPTBaseModelConfig) -> dict:
        Assert.custom(isinstance, config, GPTBaseModelConfig)
        return safe_merge_dicts(
            cls.embeddings_converter_class.export_config(config.embeddings),
            cls.decoder_converter_class.export_config(config.decoder),
            cls.head_converter_class.export_config(config.head),
            {
                "tie_word_embeddings": config.tied_embedding_weight,
                "hidden_size": config.hidden_size,
                "enable_moe_block": True,
                "hidden_size_per_layer_input": 0,
                "num_kv_shared_layers": 0,
                "use_double_wide_mlp": False,
            },
        )

    @classmethod
    def get_converters(cls, config: GPTBaseModelConfig, exported_config: dict) -> list[WeightConverter]:
        return [
            *cls.embeddings_converter_class.get_converters(config.embeddings, "embeddings", "model"),
            *cls.decoder_converter_class.get_converters(config.decoder, "decoder", "model.layers"),
            *cls.head_converter_class.get_converters(config, exported_config),
        ]

    @classmethod
    def _check_supported_config(cls, config: dict) -> None:
        if config.get("hidden_size_per_layer_input", 0):
            raise NotImplementedError("Gemma4 per-layer-input branch is not supported.")
        if config.get("num_kv_shared_layers", 0):
            raise NotImplementedError("Gemma4 KV sharing across layers is not supported.")
        if config.get("use_double_wide_mlp", False):
            raise NotImplementedError("Gemma4 use_double_wide_mlp is not supported.")
        if config.get("use_bidirectional_attention") == "all":
            raise NotImplementedError("Gemma4 non-causal text attention is not supported.")
        if not config.get("enable_moe_block", True):
            raise NotImplementedError("Gemma4 dense-only feedforward blocks are not supported.")
        if config.get("attention_bias", False):
            raise NotImplementedError("Gemma4 attention biases are not supported.")
        if not config.get("attention_k_eq_v", False):
            raise NotImplementedError("Gemma4 full attention without shared K=V is not supported.")


class Gemma4HuggingfaceCheckpointHandler(LlamaHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = Gemma4CheckpointFormat
    architecture: typing.ClassVar[str] = "Gemma4ForCausalLM"
    base_model_converter_class: typing.ClassVar[type[Gemma4BaseModelConverter]] = Gemma4BaseModelConverter

    @classmethod
    def get_huggingface_model_type(cls) -> str:
        return "gemma4_text"

    @classmethod
    def get_transformers_configuration_class(cls):
        import transformers

        return transformers.Gemma4TextConfig
