"""Gemma4 checkpoint format converter."""

import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    SplitWeightConverter,
    WeightConverter,
)
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.attention.rotary.config import DefaultRotaryConfig, ProportionalRotaryConfig
from fast_llm.layers.block.config import PatternBlockSequenceConfig
from fast_llm.layers.common.normalization.config import FixedRMSNormConfig, RMSNormalizationConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.layers.decoder.mlp.config import HybridMoEMLPConfig, MLPConfig, MoEMLPConfig
from fast_llm.layers.language_model.config import (
    LanguageModelEmbeddingsConfig,
    LanguageModelHeadConfig,
)
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.models.gpt.conversion.config import Gemma4CheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    KeyValueWeightConverter,
    LlamaEmbeddingsConverter,
    LlamaHeadConverter,
    LlamaNormalizationConverter,
    MLPLayer2Converter,
    QueryWeightConverter,
    get_parameter_converter,
    get_weight_and_bias_converters,
)
from fast_llm.models.gpt.model import GPTModel
from fast_llm.utils import Assert, safe_merge_dicts

_SLIDING_ATTENTION = "sliding_attention"
_FULL_ATTENTION = "full_attention"


class Gemma4MoELayer1Converter(WeightConverter):
    """Converts batched gate_up_proj [experts, 2*intermediate, hidden] ↔ Fast-LLM layer_1 [experts*2*intermediate, hidden]."""

    _config: MoEMLPConfig

    def export_weight(self, weight):
        (layer_1,) = weight
        return (layer_1.reshape(self._config.experts, -1, layer_1.shape[-1]),)

    def import_weight(self, weight):
        (gate_up_proj,) = weight
        # `[:]` materializes the HF safetensors slice into a real tensor; `reshape` rejects the slice.
        w = gate_up_proj[:]
        return (w.reshape(-1, w.shape[-1]),)


class Gemma4MoELayer2Converter(WeightConverter):
    """Converts batched down_proj [experts, hidden, intermediate] ↔ Fast-LLM layer_2 [experts*intermediate, hidden]."""

    _config: MoEMLPConfig

    def export_weight(self, weight):
        (layer_2,) = weight
        return (layer_2.reshape(self._config.experts, -1, layer_2.shape[-1]).permute(0, 2, 1).contiguous(),)

    def import_weight(self, weight):
        (down_proj,) = weight
        # `[:]` materializes the HF safetensors slice into a real tensor; `permute`/`reshape` reject the slice.
        w = down_proj[:]
        return (w.permute(0, 2, 1).reshape(-1, w.shape[1]).contiguous(),)


class Gemma4AttentionConverter:
    @classmethod
    def import_config(cls, config: dict, is_sliding: bool) -> dict:
        eps = config["rms_norm_eps"]
        if is_sliding:
            rope_params = config["rope_parameters"][_SLIDING_ATTENTION]
            rotary = {"type": "default", "theta": rope_params["rope_theta"]}
            head_size = config["head_dim"]
            head_groups = config["num_key_value_heads"]
            window_size = config["sliding_window"]
        else:
            rope_params = config["rope_parameters"][_FULL_ATTENTION]
            rotary = {
                "type": "proportional",
                "theta": rope_params["rope_theta"],
                "partial_rotary_factor": rope_params["partial_rotary_factor"],
            }
            head_size = config["global_head_dim"]
            num_global_kv_heads = config.get("num_global_key_value_heads")
            head_groups = config["num_key_value_heads"] if num_global_kv_heads is None else num_global_kv_heads
            window_size = None
        out = {
            "heads": config["num_attention_heads"],
            "head_groups": head_groups,
            "head_size": head_size,
            "add_linear_biases": False,
            "dropout": config["attention_dropout"],
            "softmax_scale_power": 0,
            "rotary": rotary,
            "query_norm": {"type": "rms_norm", "epsilon": eps},
            "key_norm": {"type": "rms_norm", "epsilon": eps},
            "value_norm": {"type": "fixed_rms_norm", "epsilon": eps},
        }
        if not is_sliding and config.get("attention_k_eq_v", False):
            out["shared_key_value"] = True
        if window_size is not None:
            out["window_size"] = window_size
        return out

    @classmethod
    def export_config(cls, sliding_config: AttentionConfig, full_config: AttentionConfig) -> dict:
        Assert.custom(isinstance, sliding_config, AttentionConfig)
        Assert.custom(isinstance, full_config, AttentionConfig)
        if sliding_config.add_linear_biases:
            raise NotImplementedError(f"`add_linear_biases=True` is not supported by `{cls.__name__}`.")
        Assert.custom(isinstance, sliding_config.rotary, DefaultRotaryConfig)
        Assert.custom(isinstance, full_config.rotary, ProportionalRotaryConfig)
        Assert.custom(isinstance, sliding_config.query_norm, RMSNormalizationConfig)
        Assert.custom(isinstance, sliding_config.key_norm, RMSNormalizationConfig)
        Assert.custom(isinstance, sliding_config.value_norm, FixedRMSNormConfig)
        eps = sliding_config.query_norm.epsilon
        num_global_kv_heads = (
            None if full_config.head_groups == sliding_config.head_groups else full_config.head_groups
        )
        return {
            "num_attention_heads": sliding_config.heads,
            "num_key_value_heads": sliding_config.head_groups,
            "head_dim": sliding_config.head_size,
            "global_head_dim": full_config.head_size,
            "num_global_key_value_heads": num_global_kv_heads,
            "attention_bias": False,
            "attention_dropout": sliding_config.dropout,
            "sliding_window": sliding_config.window_size,
            "rms_norm_eps": eps,
            "attention_k_eq_v": full_config.shared_key_value,
            "rope_parameters": {
                _SLIDING_ATTENTION: {
                    "rope_type": "default",
                    "rope_theta": sliding_config.rotary.theta,
                },
                _FULL_ATTENTION: {
                    "rope_type": "proportional",
                    "rope_theta": full_config.rotary.theta,
                    "partial_rotary_factor": full_config.rotary.partial_rotary_factor,
                },
            },
        }

    @classmethod
    def get_converters(
        cls,
        config: AttentionConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        if config.shared_key_value:
            # K=V: single k_proj reused as value; no v_proj in HF
            kv_converters = get_weight_and_bias_converters(
                f"{fast_llm_prefix}.key_value",
                f"{hf_prefix}.k_proj",
                False,
                drop_on_export=drop_on_export,
            )
        else:
            kv_converters = get_weight_and_bias_converters(
                f"{fast_llm_prefix}.key_value",
                (f"{hf_prefix}.k_proj", f"{hf_prefix}.v_proj"),
                False,
                KeyValueWeightConverter,
                config,
                drop_on_export=drop_on_export,
            )
        converters = [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.query",
                f"{hf_prefix}.q_proj",
                False,
                QueryWeightConverter,
                config,
                drop_on_export=drop_on_export,
            ),
            *kv_converters,
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.dense",
                f"{hf_prefix}.o_proj",
                False,
                drop_on_export=drop_on_export,
            ),
        ]
        if config.query_norm is not None:
            converters += LlamaNormalizationConverter.get_converters(
                config.query_norm,
                f"{fast_llm_prefix}.query_norm",
                f"{hf_prefix}.q_norm",
                drop_on_export=drop_on_export,
            )
        if config.key_norm is not None:
            converters += LlamaNormalizationConverter.get_converters(
                config.key_norm,
                f"{fast_llm_prefix}.key_norm",
                f"{hf_prefix}.k_norm",
                drop_on_export=drop_on_export,
            )
        # value_norm is FixedRMSNorm — no learnable weight to convert
        return converters


class Gemma4MLPConverter:
    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            "intermediate_size": config["intermediate_size"],
            "add_linear_biases": False,
            "activation": ActivationType.from_hf_name(config["hidden_activation"]),
            "gated": True,
        }

    @classmethod
    def export_config(cls, config: MLPConfig) -> dict:
        Assert.custom(isinstance, config, MLPConfig)
        if not config.gated:
            raise NotImplementedError(f"`gated=False` is not supported by `{cls.__name__}`.")
        if config.add_linear_biases:
            raise NotImplementedError(f"`add_linear_biases=True` is not supported by `{cls.__name__}`.")
        return {
            "intermediate_size": config.intermediate_size,
            "hidden_activation": config.activation.hf_name,
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
                False,
                SplitWeightConverter,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_2",
                f"{hf_prefix}.down_proj",
                False,
                MLPLayer2Converter,
                drop_on_export=drop_on_export,
            ),
        ]


class Gemma4MoEMLPConverter:
    @classmethod
    def import_config(cls, config: dict) -> dict:
        eps = config["rms_norm_eps"]
        return {
            "type": "moe",
            "intermediate_size": config["moe_intermediate_size"],
            "add_linear_biases": False,
            "activation": ActivationType.from_hf_name(config["hidden_activation"]),
            "gated": True,
            "experts": config["num_experts"],
            "experts_per_token": config["top_k_experts"],
            "router_normalization": {"type": "fixed_rms_norm", "epsilon": eps},
            "router_scale": {"enabled": True},
            "router_input_scale": config["hidden_size"] ** -0.5,
            "router_per_expert_scale": {"enabled": True},
        }

    @classmethod
    def export_config(cls, config: MoEMLPConfig, hidden_size: int) -> dict:
        Assert.custom(isinstance, config, MoEMLPConfig)
        if not config.gated:
            raise NotImplementedError(f"`gated=False` is not supported by `{cls.__name__}`.")
        if config.add_linear_biases:
            raise NotImplementedError(f"`add_linear_biases=True` is not supported by `{cls.__name__}`.")
        if not isinstance(config.router_normalization, FixedRMSNormConfig):
            raise NotImplementedError(
                f"`router_normalization` must be `FixedRMSNormConfig` for `{cls.__name__}`,"
                f" got `{type(config.router_normalization).__name__}`."
            )
        if not config.router_scale.enabled:
            raise NotImplementedError(f"`router_scale` must be enabled for `{cls.__name__}`.")
        expected_input_scale = hidden_size**-0.5
        if config.router_input_scale != expected_input_scale:
            raise NotImplementedError(
                f"`router_input_scale` must be `hidden_size ** -0.5` (= {expected_input_scale}) for"
                f" `{cls.__name__}`, got {config.router_input_scale}."
            )
        if not config.router_per_expert_scale.enabled:
            raise NotImplementedError(f"`router_per_expert_scale` must be enabled for `{cls.__name__}`.")
        return {
            "num_experts": config.experts,
            "top_k_experts": config.experts_per_token,
            "moe_intermediate_size": config.intermediate_size,
        }

    @classmethod
    def get_converters(
        cls,
        config: MoEMLPConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        converters = [
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
                f"{fast_llm_prefix}.router_per_expert_scale",
                f"{hf_prefix}.router.per_expert_scale",
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.layer_1.weight",
                f"{hf_prefix}.experts.gate_up_proj",
                Gemma4MoELayer1Converter,
                config,
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.layer_2.weight",
                f"{hf_prefix}.experts.down_proj",
                Gemma4MoELayer2Converter,
                config,
                drop_on_export=drop_on_export,
            ),
        ]
        # router.norm is FixedRMSNorm — no learnable weight to convert.
        return converters


class Gemma4HybridMoEMLPConverter:
    @classmethod
    def import_config(cls, config: dict) -> dict:
        eps = config["rms_norm_eps"]

        def make_norm() -> dict:
            return {"type": "rms_norm", "epsilon": eps}

        dense = Gemma4MLPConverter.import_config(config)
        dense["pre_norm"] = make_norm()
        dense["post_norm"] = make_norm()
        routed = Gemma4MoEMLPConverter.import_config(config)
        routed["pre_norm"] = make_norm()
        routed["post_norm"] = make_norm()
        return {"type": "hybrid_moe", "dense": dense, "routed": routed}

    @classmethod
    def export_config(cls, config: HybridMoEMLPConfig, hidden_size: int) -> dict:
        Assert.custom(isinstance, config, HybridMoEMLPConfig)
        return safe_merge_dicts(
            Gemma4MLPConverter.export_config(config.dense),
            Gemma4MoEMLPConverter.export_config(config.routed, hidden_size),
            {"enable_moe_block": True},
        )

    @classmethod
    def get_converters(
        cls,
        config: HybridMoEMLPConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return [
            *Gemma4MLPConverter.get_converters(
                config.dense,
                f"{fast_llm_prefix}.dense",
                f"{hf_prefix}.mlp",
                drop_on_export=drop_on_export,
            ),
            *Gemma4MoEMLPConverter.get_converters(
                config.routed,
                f"{fast_llm_prefix}.routed",
                hf_prefix,
                drop_on_export=drop_on_export,
            ),
            *LlamaNormalizationConverter.get_converters(
                config.dense.pre_norm,
                f"{fast_llm_prefix}.dense.pre_norm",
                f"{hf_prefix}.pre_feedforward_layernorm",
                drop_on_export=drop_on_export,
            ),
            *LlamaNormalizationConverter.get_converters(
                config.dense.post_norm,
                f"{fast_llm_prefix}.dense.post_norm",
                f"{hf_prefix}.post_feedforward_layernorm_1",
                drop_on_export=drop_on_export,
            ),
            *LlamaNormalizationConverter.get_converters(
                config.routed.pre_norm,
                f"{fast_llm_prefix}.routed.pre_norm",
                f"{hf_prefix}.pre_feedforward_layernorm_2",
                drop_on_export=drop_on_export,
            ),
            *LlamaNormalizationConverter.get_converters(
                config.routed.post_norm,
                f"{fast_llm_prefix}.routed.post_norm",
                f"{hf_prefix}.post_feedforward_layernorm_2",
                drop_on_export=drop_on_export,
            ),
        ]


class Gemma4BlockConverter:
    @classmethod
    def import_config(cls, config: dict, is_sliding: bool) -> dict:
        def make_norm():
            return {"type": "rms_norm", "epsilon": config["rms_norm_eps"]}

        out = {
            "mixer": Gemma4AttentionConverter.import_config(config, is_sliding),
            "normalization": make_norm(),
            "post_mixer_normalization": make_norm(),
            "post_mlp_normalization": make_norm(),
            # HF stores `layer_scalar` as a non-trained buffer; freeze on our side to match.
            "output_scale": {"enabled": True, "lr_scale": 0},
        }
        if config.get("enable_moe_block"):
            out["mlp"] = Gemma4HybridMoEMLPConverter.import_config(config)
            out["pre_mlp_normalization"] = {"type": "none"}
        else:
            out["mlp"] = Gemma4MLPConverter.import_config(config)
            out["pre_mlp_normalization"] = make_norm()
        return out

    @classmethod
    def export_config(
        cls, sliding_config: DecoderBlockConfig, full_config: DecoderBlockConfig, hidden_size: int
    ) -> dict:
        Assert.custom(isinstance, sliding_config, DecoderBlockConfig)
        norm_config = sliding_config.normalization
        Assert.custom(isinstance, norm_config, RMSNormalizationConfig)
        is_moe = isinstance(sliding_config.mlp, HybridMoEMLPConfig)
        out = safe_merge_dicts(
            Gemma4AttentionConverter.export_config(sliding_config.mixer, full_config.mixer),
            LlamaNormalizationConverter.export_config(norm_config),
        )
        if is_moe:
            out = safe_merge_dicts(out, Gemma4HybridMoEMLPConverter.export_config(sliding_config.mlp, hidden_size))
        else:
            out = safe_merge_dicts(out, Gemma4MLPConverter.export_config(sliding_config.mlp))
            out["enable_moe_block"] = False
        return out

    @classmethod
    def get_converters(
        cls,
        config: DecoderBlockConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        is_moe = isinstance(config.mlp, HybridMoEMLPConfig)
        converters = [
            *Gemma4AttentionConverter.get_converters(
                config.mixer,
                f"{fast_llm_prefix}.mixer",
                f"{hf_prefix}.self_attn",
                drop_on_export=drop_on_export,
            ),
        ]
        if is_moe:
            converters += Gemma4HybridMoEMLPConverter.get_converters(
                config.mlp,
                f"{fast_llm_prefix}.mlp",
                hf_prefix,
                drop_on_export=drop_on_export,
            )
        else:
            converters += Gemma4MLPConverter.get_converters(
                config.mlp,
                f"{fast_llm_prefix}.mlp",
                f"{hf_prefix}.mlp",
                drop_on_export=drop_on_export,
            )
            converters += LlamaNormalizationConverter.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.norm_2",
                f"{hf_prefix}.pre_feedforward_layernorm",
                drop_on_export=drop_on_export,
            )
        converters += [
            *LlamaNormalizationConverter.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.norm_1",
                f"{hf_prefix}.input_layernorm",
                drop_on_export=drop_on_export,
            ),
            *LlamaNormalizationConverter.get_converters(
                config.post_mixer_normalization,
                f"{fast_llm_prefix}.post_mixer_norm",
                f"{hf_prefix}.post_attention_layernorm",
                drop_on_export=drop_on_export,
            ),
            *LlamaNormalizationConverter.get_converters(
                config.post_mlp_normalization,
                f"{fast_llm_prefix}.post_mlp_norm",
                f"{hf_prefix}.post_feedforward_layernorm",
                drop_on_export=drop_on_export,
            ),
        ]
        converters.append(
            get_parameter_converter(
                f"{fast_llm_prefix}.output_scale",
                f"{hf_prefix}.layer_scalar",
                drop_on_export=drop_on_export,
            )
        )
        return converters


class Gemma4DecoderConverter:
    block_converter_class: typing.ClassVar[type[Gemma4BlockConverter]] = Gemma4BlockConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        layer_types = config["layer_types"]
        unique_types = list(dict.fromkeys(layer_types))
        blocks = {
            layer_type: cls.block_converter_class.import_config(config, layer_type == _SLIDING_ATTENTION)
            for layer_type in unique_types
        }
        return {
            "type": "pattern",
            "blocks": blocks,
            "pattern": layer_types,
            "num_blocks": config["num_hidden_layers"],
        }

    @classmethod
    def export_config(cls, config: PatternBlockSequenceConfig, hidden_size: int) -> dict:
        Assert.custom(isinstance, config, PatternBlockSequenceConfig)
        Assert.incl(_SLIDING_ATTENTION, config.blocks)
        Assert.incl(_FULL_ATTENTION, config.blocks)
        return safe_merge_dicts(
            cls.block_converter_class.export_config(
                config.blocks[_SLIDING_ATTENTION],
                config.blocks[_FULL_ATTENTION],
                hidden_size,
            ),
            {
                "num_hidden_layers": config.num_blocks,
                "layer_types": list(config.expanded_pattern),
            },
        )

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
        for block_index in range(config.num_blocks):
            block_config = config.blocks[config.expanded_pattern[block_index]]
            converters += cls.block_converter_class.get_converters(
                block_config,
                f"{fast_llm_prefix}.{block_index}",
                f"{hf_prefix}.{block_index}",
                drop_on_export=drop_on_export,
            )
        return converters


class Gemma4EmbeddingsConverter(LlamaEmbeddingsConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            "vocab_size": config["vocab_size"],
            "embedding_scale": config["hidden_size"] ** 0.5,
        }

    @classmethod
    def export_config(cls, config: LanguageModelEmbeddingsConfig, hidden_size: int) -> dict:
        Assert.custom(isinstance, config, LanguageModelEmbeddingsConfig)
        if config.position_embeddings.enabled:
            raise NotImplementedError(f"`position_embeddings` is not supported by `{cls.__name__}`.")
        # Gemma 4 hard-codes embed_scale = hidden_size ** 0.5; reject divergent values rather than
        # silently dropping them.
        Assert.eq(config.embedding_scale, hidden_size**0.5)
        return {"vocab_size": config.vocab_size}


class Gemma4HeadConverter(LlamaHeadConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        out = {"normalization": LlamaNormalizationConverter.import_config(config)}
        if (softcap := config.get("final_logit_softcapping")) is not None:
            out["final_logit_softcap"] = softcap
        return out

    @classmethod
    def export_config(cls, config: LanguageModelHeadConfig) -> dict:
        out = LlamaNormalizationConverter.export_config(config.normalization)
        if config.final_logit_softcap is not None:
            out["final_logit_softcapping"] = config.final_logit_softcap
        return out


class Gemma4BaseModelConverter:
    decoder_converter_class: typing.ClassVar[type[Gemma4DecoderConverter]] = Gemma4DecoderConverter
    embeddings_converter_class: typing.ClassVar[type[Gemma4EmbeddingsConverter]] = Gemma4EmbeddingsConverter
    head_converter_class: typing.ClassVar[type[Gemma4HeadConverter]] = Gemma4HeadConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        if config.get("hidden_size_per_layer_input") not in (None, 0):
            raise NotImplementedError(
                "Gemma 4 Per-Layer Embeddings (`hidden_size_per_layer_input != 0`) are not supported."
            )
        if config.get("num_kv_shared_layers", 0):
            raise NotImplementedError("Gemma 4 cross-layer KV sharing (`num_kv_shared_layers != 0`) is not supported.")
        if config.get("use_double_wide_mlp", False):
            raise NotImplementedError("Gemma 4 `use_double_wide_mlp=True` is not supported.")
        # `use_bidirectional_attention="vision"` only affects vision tokens; the text path stays causal.
        # Only `"all"` toggles `is_causal=False` for the text decoder, which we don't implement.
        if config.get("use_bidirectional_attention") == "all":
            raise NotImplementedError(
                'Gemma 4 `use_bidirectional_attention="all"` is not supported (text path stays causal).'
            )
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
            cls.embeddings_converter_class.export_config(config.embeddings, config.hidden_size),
            cls.decoder_converter_class.export_config(config.decoder, config.hidden_size),
            cls.head_converter_class.export_config(config.head),
            {
                "tie_word_embeddings": config.tied_embedding_weight,
                "hidden_size": config.hidden_size,
                # TODO: Implement Per-Layer Embeddings (PLE). Gemma4TextConfig defaults to 256;
                # explicitly zero to disable the feature in the exported model until Fast-LLM
                # supports it natively.
                "hidden_size_per_layer_input": 0,
                # Fast-LLM is text-only; bidirectional attention (used for vision tokens in the
                # multimodal model) is not implemented.
                "use_bidirectional_attention": None,
            },
        )

    @classmethod
    def get_converters(cls, config: GPTBaseModelConfig, exported_config: dict) -> list[WeightConverter]:
        return [
            *cls.embeddings_converter_class.get_converters(config.embeddings, "embeddings", "model"),
            *cls.decoder_converter_class.get_converters(config.decoder, "decoder", "model.layers"),
            *cls.head_converter_class.get_converters(config, exported_config),
        ]


class Gemma4HuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    _model: GPTModel
    _model_class: typing.ClassVar = GPTModelConfig
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
