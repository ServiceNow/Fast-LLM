"""Gemma4 checkpoint format converter."""

import functools
import typing

import torch

from fast_llm.config import Config
from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    BlockSequenceWeightConverter,
    ConfigSectionConverter,
    ConstantExportConfigConverter,
    CustomConfigConverter,
    IgnoredConfigConverter,
    LinearWeightConverter,
    NestedWeightConverter,
    RenameConfigConverter,
    SplitWeightConverter,
    TransposeSplitWeightConverter,
    WeightConverter,
    _join_prefix,
)
from fast_llm.engine.checkpoint.huggingface import HuggingFaceBaseModelConverter, HuggingfaceStateDictCheckpointHandler
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
    LlamaEmbeddingsConverter,
    LlamaHeadConverter,
    LlamaNormalizationConverter,
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


class _Gemma4BlockMLPWeightConverter(WeightConverter):
    """Dispatch ``block.mlp`` to dense :class:`Gemma4MLPConverter` (under ``mlp.<...>``) or hybrid
    :class:`Gemma4HybridMoEMLPConverter` (flat-merged into the block's HF root) based on the runtime
    type of ``config.mlp``. The two targets diverge on HF prefix, which the generic
    :class:`DispatchWeightConverter` doesn't accommodate.
    """

    def __init__(self) -> None:
        super().__init__((), ())

    def _emit(self, config, fast_llm_prefix, hf_prefix, *, root_config):
        fast_llm_mlp = _join_prefix(fast_llm_prefix, "mlp")
        if isinstance(config.mlp, HybridMoEMLPConfig):
            return Gemma4HybridMoEMLPConverter.emit_weight_converters(
                config.mlp, fast_llm_mlp, hf_prefix, root_config=root_config
            )
        return Gemma4MLPConverter.emit_weight_converters(
            config.mlp, fast_llm_mlp, _join_prefix(hf_prefix, "mlp"), root_config=root_config
        )


class _Gemma4BlockNorm2WeightConverter(WeightConverter):
    """Dense Gemma4 blocks store the pre-MLP norm at ``norm_2`` (drawn from the block's main
    ``normalization`` config). MoE blocks suppress this — the routed/dense branches inside the
    hybrid MoE own their own pre/post norms.
    """

    def __init__(self) -> None:
        super().__init__((), ())

    def _emit(self, config, fast_llm_prefix, hf_prefix, *, root_config):
        if isinstance(config.mlp, HybridMoEMLPConfig):
            return []
        return LlamaNormalizationConverter.emit_weight_converters(
            config.normalization,
            _join_prefix(fast_llm_prefix, "norm_2"),
            _join_prefix(hf_prefix, "pre_feedforward_layernorm"),
            root_config=root_config,
        )


class _Gemma4HybridMoENormWeightConverter(WeightConverter):
    """Emit a normalization config nested two attributes deep inside a hybrid MoE block
    (e.g. ``config.dense.pre_norm``, ``config.routed.post_norm``). The single-level
    :class:`NestedWeightConverter` can't express the chained descent — :class:`MLPConfig` and
    :class:`MoEMLPConfig` each carry their own pre/post norms, but those branches live one level
    below the hybrid MoE section root. Gemma4's hybrid MoE always sets these norms.
    """

    def __init__(self, branch: str, norm_attr: str, hf_name: str) -> None:
        super().__init__((), ())
        self._branch = branch
        self._norm_attr = norm_attr
        self._hf_name = hf_name

    def _emit(self, config, fast_llm_prefix, hf_prefix, *, root_config):
        norm_config = getattr(getattr(config, self._branch), self._norm_attr)
        return LlamaNormalizationConverter.emit_weight_converters(
            norm_config,
            _join_prefix(fast_llm_prefix, f"{self._branch}.{self._norm_attr}"),
            _join_prefix(hf_prefix, self._hf_name),
            root_config=root_config,
        )


class _Gemma4SharedKeyValueWeightConverter(WeightConverter):
    """``shared_key_value=True`` Gemma4 attention: Fast-LLM's ``key_value`` is a single K-shaped
    tensor (V is reused at runtime) and maps to a single HF ``k_proj`` — plain rename. Falls back to
    :class:`KeyValueWeightConverter` (chunk/cat across K and V) when not shared.
    """

    _config: AttentionConfig

    def export_weight(self, weight):
        if self._config.shared_key_value:
            return weight
        (key_value,) = weight
        return key_value[:].chunk(2)

    def import_weight(self, weight):
        if self._config.shared_key_value:
            return weight
        key, value = weight
        return (torch.cat([key[:], value[:]]),)


class Gemma4AttentionConverter(ConfigSectionConverter):
    """Gemma4's attention helper: ``import_config`` / ``export_config`` take non-standard arguments
    (sliding/full discrimination, twin block exports) and are invoked imperatively from
    :class:`Gemma4BlockConverter`. Only the weight side fits the standard declarative shape — biases
    are always disabled, query/key norms are emitted only when present, and the K/V layout collapses
    to a single ``k_proj`` when ``shared_key_value`` is set.

    The config side is owned by :class:`Gemma4BaseModelConverter`'s ``decoder`` :class:`CustomConfigConverter`
    (with ``fast_llm_recurses=True``); the blanket-claim below silences the static architecture-coverage
    walker — Gemma4's sliding/full divergence prevents a uniform declarative shape per single block.
    """

    fast_llm_config_class = AttentionConfig

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {"_blanket": IgnoredConfigConverter(())}

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
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "query": LinearWeightConverter("query", "q_proj", bias_fn=lambda c: False),
            "key_value": LinearWeightConverter(
                "key_value",
                lambda c: "k_proj" if c.shared_key_value else ("k_proj", "v_proj"),
                transform=_Gemma4SharedKeyValueWeightConverter,
                bias_fn=lambda c: False,
            ),
            "dense": LinearWeightConverter("dense", "o_proj", bias_fn=lambda c: False),
            # ``value_norm`` is :class:`FixedRMSNormConfig` (no learnable weight) — not declared.
            "query_norm": NestedWeightConverter("query_norm", "q_norm", LlamaNormalizationConverter, optional=True),
            "key_norm": NestedWeightConverter("key_norm", "k_norm", LlamaNormalizationConverter, optional=True),
        }


class Gemma4MLPConverter(ConfigSectionConverter):
    fast_llm_config_class = MLPConfig

    @classmethod
    def _create_config_converters(cls) -> dict:
        # Config side owned by the aggregator's ``decoder`` CustomConfigConverter; see Gemma4AttentionConverter.
        return {"_blanket": IgnoredConfigConverter(())}

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
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "layer_1": LinearWeightConverter(
                "layer_1", ("gate_proj", "up_proj"), transform=SplitWeightConverter, bias_fn=lambda c: False
            ),
            "layer_2": LinearWeightConverter(
                "layer_2", "down_proj", transform=TransposeSplitWeightConverter, bias_fn=lambda c: False
            ),
        }


class Gemma4MoEMLPConverter(ConfigSectionConverter):
    fast_llm_config_class = MoEMLPConfig

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {"_blanket": IgnoredConfigConverter(())}

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
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        # ``router.norm`` is :class:`FixedRMSNormConfig` (no learnable weight) — not declared.
        return {
            "router": LinearWeightConverter("router", "router.proj", bias_fn=lambda c: False),
            "router_scale": WeightConverter("router_scale", "router.scale"),
            "router_per_expert_scale": WeightConverter("router_per_expert_scale", "router.per_expert_scale"),
            "layer_1": Gemma4MoELayer1Converter("layer_1.weight", "experts.gate_up_proj"),
            "layer_2": Gemma4MoELayer2Converter("layer_2.weight", "experts.down_proj"),
        }


class Gemma4HybridMoEMLPConverter(ConfigSectionConverter):
    fast_llm_config_class = HybridMoEMLPConfig

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {"_blanket": IgnoredConfigConverter(())}

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
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "dense": NestedWeightConverter("dense", "mlp", Gemma4MLPConverter),
            # Routed branch lives at the block's HF root (sibling of ``mlp.<...>``).
            "routed": NestedWeightConverter("routed", "", Gemma4MoEMLPConverter),
            "dense_pre_norm": _Gemma4HybridMoENormWeightConverter("dense", "pre_norm", "pre_feedforward_layernorm"),
            "dense_post_norm": _Gemma4HybridMoENormWeightConverter(
                "dense", "post_norm", "post_feedforward_layernorm_1"
            ),
            "routed_pre_norm": _Gemma4HybridMoENormWeightConverter(
                "routed", "pre_norm", "pre_feedforward_layernorm_2"
            ),
            "routed_post_norm": _Gemma4HybridMoENormWeightConverter(
                "routed", "post_norm", "post_feedforward_layernorm_2"
            ),
        }


class Gemma4BlockConverter(ConfigSectionConverter):
    fast_llm_config_class = DecoderBlockConfig

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {"_blanket": IgnoredConfigConverter(())}

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
        Assert.eq(isinstance(full_config.mlp, HybridMoEMLPConfig), is_moe)
        # The MoE block intentionally sets `pre_mlp_normalization=NoNorm` (routed branch owns its norms).
        for block_config in (sliding_config, full_config):
            if block_config.pre_mixer_normalization is not None:
                Assert.eq(block_config.pre_mixer_normalization, block_config.normalization)
            if not is_moe and block_config.pre_mlp_normalization is not None:
                Assert.eq(block_config.pre_mlp_normalization, block_config.normalization)
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
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "mixer": NestedWeightConverter("mixer", "self_attn", Gemma4AttentionConverter),
            "mlp": _Gemma4BlockMLPWeightConverter(),
            "norm_1": NestedWeightConverter(
                "norm_1", "input_layernorm", LlamaNormalizationConverter, config_attr="normalization"
            ),
            "norm_2": _Gemma4BlockNorm2WeightConverter(),
            "post_mixer_norm": NestedWeightConverter(
                "post_mixer_norm",
                "post_attention_layernorm",
                LlamaNormalizationConverter,
                config_attr="post_mixer_normalization",
            ),
            "post_mlp_norm": NestedWeightConverter(
                "post_mlp_norm",
                "post_feedforward_layernorm",
                LlamaNormalizationConverter,
                config_attr="post_mlp_normalization",
            ),
            # HF stores ``layer_scalar`` as a non-trained buffer; Fast-LLM mirrors it with a frozen
            # ``output_scale`` (``lr_scale=0``).
            "output_scale": WeightConverter("output_scale", "layer_scalar"),
        }


class Gemma4DecoderConverter(ConfigSectionConverter):
    fast_llm_config_class = PatternBlockSequenceConfig

    block_converter_class: typing.ClassVar[type[Gemma4BlockConverter]] = Gemma4BlockConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {"_blanket": IgnoredConfigConverter(())}

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
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "blocks": BlockSequenceWeightConverter("", "", cls.block_converter_class, read_self=True),
        }


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


def _gemma4_bidirectional_export(_: Config) -> dict:
    # Fast-LLM is text-only; bidirectional attention (used for vision tokens in the multimodal
    # model) is not implemented. Always emit ``None``.
    return {("use_bidirectional_attention",): None}


def _gemma4_bidirectional_import(hf_dict: dict) -> dict:
    # ``use_bidirectional_attention="vision"`` only affects vision tokens; the text path stays
    # causal. Only ``"all"`` toggles ``is_causal=False`` for the text decoder, which we don't
    # implement.
    if hf_dict.get("use_bidirectional_attention") == "all":
        raise NotImplementedError(
            'Gemma 4 `use_bidirectional_attention="all"` is not supported (text path stays causal).'
        )
    return {}


class Gemma4BaseModelConverter(ConfigSectionConverter, HuggingFaceBaseModelConverter):
    """Top-level converter for ``GPTBaseModelConfig`` ↔ Gemma 4 HF dict.

    Gemma 4 has several wrinkles that prevent the standard per-section decomposition used by Llama:

    * The decoder is a :class:`PatternBlockSequenceConfig` whose two named blocks
      (``sliding_attention`` / ``full_attention``) share most HF keys but diverge on ``head_dim`` and
      rope parameters. The HF format emits both block variants from a single root-level config, so
      the block-level transform inherently sees both Fast-LLM blocks at once.
    * ``embedding_scale = hidden_size ** 0.5`` and ``router_input_scale = hidden_size ** -0.5`` make
      the embeddings and routed MLP cross-reference the root-level ``hidden_size``.

    Each section ((embeddings, decoder, head)) is therefore expressed as a :class:`CustomConfigConverter`
    that delegates to an imperative helper class (kept private to this module). Coverage at the
    section level is satisfied via ``fast_llm_recurses=True``.
    """

    fast_llm_config_class = GPTBaseModelConfig

    decoder_converter_class: typing.ClassVar[type[Gemma4DecoderConverter]] = Gemma4DecoderConverter
    embeddings_converter_class: typing.ClassVar[type[Gemma4EmbeddingsConverter]] = Gemma4EmbeddingsConverter
    head_converter_class: typing.ClassVar[type[Gemma4HeadConverter]] = Gemma4HeadConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        decoder_cls = cls.decoder_converter_class
        embeddings_cls = cls.embeddings_converter_class
        head_cls = cls.head_converter_class

        def _embeddings_export(parent: Config) -> dict:
            return {(k,): v for k, v in embeddings_cls.export_config(parent.embeddings, parent.hidden_size).items()}

        def _embeddings_import(hf_dict: dict) -> dict:
            return {("embeddings",): embeddings_cls.import_config(hf_dict)}

        def _decoder_export(parent: Config) -> dict:
            return {(k,): v for k, v in decoder_cls.export_config(parent.decoder, parent.hidden_size).items()}

        def _decoder_import(hf_dict: dict) -> dict:
            return {("decoder",): decoder_cls.import_config(hf_dict)}

        def _head_export(parent: Config) -> dict:
            return {(k,): v for k, v in head_cls.export_config(parent.head).items()}

        def _head_import(hf_dict: dict) -> dict:
            return {("head",): head_cls.import_config(hf_dict)}

        return {
            "embeddings": CustomConfigConverter(
                fast_llm_paths=(("embeddings",),),
                hf_paths=(("vocab_size",),),
                export_fn=_embeddings_export,
                import_fn=_embeddings_import,
                fast_llm_recurses=True,
            ),
            "decoder": CustomConfigConverter(
                fast_llm_paths=(("decoder",),),
                hf_paths=(
                    ("num_hidden_layers",),
                    ("layer_types",),
                    ("num_attention_heads",),
                    ("num_key_value_heads",),
                    ("head_dim",),
                    ("global_head_dim",),
                    ("num_global_key_value_heads",),
                    ("attention_bias",),
                    ("attention_dropout",),
                    ("sliding_window",),
                    ("rms_norm_eps",),
                    ("attention_k_eq_v",),
                    ("rope_parameters",),
                    ("intermediate_size",),
                    ("hidden_activation",),
                    ("enable_moe_block",),
                    ("num_experts",),
                    ("top_k_experts",),
                    ("moe_intermediate_size",),
                ),
                export_fn=_decoder_export,
                import_fn=_decoder_import,
                fast_llm_recurses=True,
            ),
            "head": CustomConfigConverter(
                fast_llm_paths=(("head",),),
                hf_paths=(("final_logit_softcapping",),),
                export_fn=_head_export,
                import_fn=_head_import,
                fast_llm_recurses=True,
            ),
            "hidden_size": RenameConfigConverter(("hidden_size",), ("hidden_size",)),
            "tied_embedding_weight": RenameConfigConverter(("tied_embedding_weight",), ("tie_word_embeddings",)),
            "peft": IgnoredConfigConverter(("peft",)),
            # TODO: Implement Per-Layer Embeddings (PLE). Gemma4TextConfig defaults to 256; explicitly
            # zero to disable the feature in the exported model until Fast-LLM supports it natively.
            "hidden_size_per_layer_input": ConstantExportConfigConverter(("hidden_size_per_layer_input",), 0),
            "num_kv_shared_layers": ConstantExportConfigConverter(("num_kv_shared_layers",), 0),
            "use_double_wide_mlp": ConstantExportConfigConverter(("use_double_wide_mlp",), False),
            "use_bidirectional_attention": CustomConfigConverter(
                fast_llm_paths=(),
                hf_paths=(("use_bidirectional_attention",),),
                export_fn=_gemma4_bidirectional_export,
                import_fn=_gemma4_bidirectional_import,
            ),
            # Vocab-size-per-layer is part of Per-Layer Embeddings (PLE), gated by
            # ``hidden_size_per_layer_input``. PLE is rejected above, so we ignore the size field too.
            "vocab_size_per_layer_input": IgnoredConfigConverter(hf_paths=(("vocab_size_per_layer_input",),)),
        }

    @classmethod
    def get_converters(cls, config: GPTBaseModelConfig) -> list[WeightConverter]:
        return [
            *cls.embeddings_converter_class.emit_weight_converters(
                config.embeddings, "embeddings", "model", root_config=config
            ),
            *cls.decoder_converter_class.emit_weight_converters(
                config.decoder, "decoder", "model.layers", root_config=config
            ),
            *cls.head_converter_class.get_converters(config),
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
