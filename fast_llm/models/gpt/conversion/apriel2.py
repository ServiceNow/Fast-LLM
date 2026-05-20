"""Apriel2 text-only checkpoint format converter."""

import functools
import typing

from transformers import PretrainedConfig

from fast_llm.config import Config
from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    BlockSequenceWeightConverter,
    ConfigSectionConverter,
    ConstantImportConfigConverter,
    CustomConfigConverter,
    DispatchConfigConverter,
    DispatchWeightConverter,
    IgnoredConfigConverter,
    KeyValueWeightConverter,
    LinearWeightConverter,
    NestedConfigConverter,
    NestedWeightConverter,
    OptionalConfigConverter,
    OutputProjectionWeightConverter,
    RenameConfigConverter,
    SplitWeightConverter,
    TransposeSplitWeightConverter,
    TypedDictContainerConfigConverter,
    TypedDictWeightConverter,
    WeightConverter,
)
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.attention.rotary.config import DefaultRotaryConfig, Llama3RotaryConfig, YarnRotaryConfig
from fast_llm.layers.block.config import FixedBlockSequenceConfig, PatternBlockSequenceConfig
from fast_llm.layers.common.normalization.config import (
    LayerNormalizationConfig,
    NoNormalizationConfig,
    RMSNormalizationConfig,
)
from fast_llm.layers.decoder.config import DecoderBlockConfig, StochasticMixerConfig, StochasticMixerSamplingStrategy
from fast_llm.layers.decoder.mlp.config import MLPConfig
from fast_llm.layers.language_model.config import LanguageModelHeadConfig
from fast_llm.layers.ssm.config import GatedDeltaNetConfig, KimiDeltaAttentionConfig, MambaConfig
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.models.gpt.conversion.config import Apriel2TextCheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    LlamaEmbeddingsConverter,
    LlamaNormalizationConverter,
    assert_no_peft,
)
from fast_llm.models.gpt.conversion.llama import effective_bias as _effective_bias
from fast_llm.models.gpt.model import GPTModel
from fast_llm.utils import Assert, safe_merge_dicts

# ============================================================
# Helpers
# ============================================================


def _per_layer_bias_export(config: Config, layer_names: tuple[str, ...]) -> dict:
    """Emit per-layer ``{layer: {"bias": {"enabled": bool}}}`` only for layers whose bias is explicitly set."""
    out: dict = {}
    for layer_name in layer_names:
        layer = getattr(config, layer_name)
        if layer.bias.enabled is not None:
            out[(layer_name,)] = {"bias": {"enabled": layer.bias.enabled}}
    return out


def _per_layer_bias_import(hf_dict: dict, layer_names: tuple[str, ...]) -> dict:
    """Pass through HF ``{layer: {"bias": {...}}}`` entries to the Fast-LLM dict."""
    out: dict = {}
    for layer_name in layer_names:
        if layer_name in hf_dict:
            out[(layer_name,)] = hf_dict[layer_name]
    return out


def _per_layer_bias_converter(layer_names: tuple[str, ...]) -> CustomConfigConverter:
    """Per-layer ``bias.enabled`` round-trip for the named sub-layers of an attention or MLP config:
    emits/consumes the HF ``{layer: {"bias": {"enabled": ...}}}`` tree."""
    return CustomConfigConverter(
        fast_llm_paths=tuple((name,) for name in layer_names),
        hf_paths=tuple((name,) for name in layer_names),
        export_fn=lambda c: _per_layer_bias_export(c, layer_names),
        import_fn=lambda hf: _per_layer_bias_import(hf, layer_names),
        fast_llm_recurses=True,
    )


def _apriel2_kernel_size_only_conv_converter() -> CustomConfigConverter:
    """Round-trip Apriel2's flat ``convolution_layer.kernel_size`` against the Fast-LLM
    ``convolution_layer`` sub-config. Shared between :class:`Apriel2GatedDeltaNetConverter` and
    :class:`Apriel2KimiDeltaAttentionConverter`."""
    return CustomConfigConverter(
        fast_llm_paths=(("convolution_layer",), ("convolution_layer", "kernel_size")),
        hf_paths=(("convolution_layer",),),
        export_fn=lambda c: {("convolution_layer",): {"kernel_size": c.convolution_layer.kernel_size}},
        import_fn=lambda hf: ({("convolution_layer",): hf["convolution_layer"]} if "convolution_layer" in hf else {}),
    )


# ============================================================
# Mixer converters
# ============================================================


def _apriel2_attention_rotary_export(config: AttentionConfig) -> dict:
    """Emit Apriel2's typed rotary subdict.

    Asymmetric with the Fast-LLM type only for the default→``mistral_1d`` rename; ``llama3``/``yarn`` round-trip
    by name. The scale parameters of ``llama3``/``yarn`` are emitted under their Fast-LLM field names since
    the matching :func:`_apriel2_attention_rotary_import` is a wholesale pass-through of ``hf_dict["rotary"]``.
    """
    rotary = config.rotary
    if type(rotary) is DefaultRotaryConfig:
        return {("rotary",): {"type": "mistral_1d", "theta": rotary.theta}}
    if type(rotary) is Llama3RotaryConfig:
        return {
            ("rotary",): {
                "type": "llama3",
                "theta": rotary.theta,
                "scale_factor": rotary.scale_factor,
                "low_frequency_factor": rotary.low_frequency_factor,
                "high_frequency_factor": rotary.high_frequency_factor,
                "original_context_length": rotary.original_context_length,
            }
        }
    if type(rotary) is YarnRotaryConfig:
        return {
            ("rotary",): {
                "type": "yarn",
                "theta": rotary.theta,
                "scale_factor": rotary.scale_factor,
                "attention_factor": rotary.attention_factor,
                "beta_fast": rotary.beta_fast,
                "beta_slow": rotary.beta_slow,
                "original_context_length": rotary.original_context_length,
            }
        }
    raise NotImplementedError(f"Unsupported rotary type: {type(rotary).__name__}")


def _apriel2_attention_rotary_import(hf_dict: dict) -> dict:
    rotary = dict(hf_dict["rotary"])
    if rotary.get("type") == "mistral_1d":
        rotary["type"] = "default"
    return {("rotary",): rotary}


class Apriel2AttentionConverter(ConfigSectionConverter):
    fast_llm_config_class = AttentionConfig
    hf_type_name = "attention"

    @classmethod
    def _create_config_converters(cls) -> dict:
        layer_names = ("query_layer", "key_layer", "value_layer", "dense_layer")
        return {
            "heads": RenameConfigConverter(("heads",), ("heads",)),
            "head_groups": RenameConfigConverter(("head_groups",), ("head_groups",)),
            "head_size": RenameConfigConverter(("head_size",), ("head_size",)),
            "rotary": CustomConfigConverter(
                fast_llm_paths=(("rotary",),),
                hf_paths=(("rotary",),),
                export_fn=_apriel2_attention_rotary_export,
                import_fn=_apriel2_attention_rotary_import,
                fast_llm_recurses=True,
            ),
            # Apriel2 emits add_linear_biases only when False; the True default is implicit.
            "add_linear_biases": OptionalConfigConverter(
                ("add_linear_biases",), ("add_linear_biases",), sentinel=True
            ),
            "window_size": OptionalConfigConverter(("window_size",), ("window_size",)),
            "linear_layers": _per_layer_bias_converter(layer_names),
            "causal": IgnoredConfigConverter(("causal",)),
            "softmax_scale_power": IgnoredConfigConverter(("softmax_scale_power",)),
            "query_norm": ConstantImportConfigConverter(("query_norm",), None),
            "key_norm": ConstantImportConfigConverter(("key_norm",), None),
            "value_norm": ConstantImportConfigConverter(("value_norm",), None),
            "shared_key_value": ConstantImportConfigConverter(("shared_key_value",), False),
        }

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        # Each linear layer carries its own ``bias.enabled`` override; the default falls back to the
        # mixer-wide ``add_linear_biases`` via :func:`_effective_bias`. ``key_value`` is biased only when
        # both K and V agree (Fast-LLM packs them as a single tensor).
        return {
            "query": LinearWeightConverter(
                "query", "q_proj", bias_fn=lambda c: _effective_bias(c.query_layer, c.add_linear_biases)
            ),
            "key_value": LinearWeightConverter(
                "key_value",
                ("k_proj", "v_proj"),
                transform=KeyValueWeightConverter,
                bias_fn=lambda c: (
                    _effective_bias(c.key_layer, c.add_linear_biases)
                    and _effective_bias(c.value_layer, c.add_linear_biases)
                ),
            ),
            "dense": LinearWeightConverter(
                "dense", "o_proj", bias_fn=lambda c: _effective_bias(c.dense_layer, c.add_linear_biases)
            ),
        }


def _apriel2_mamba_aux_export(config: MambaConfig) -> dict:
    """Emit Apriel2's mamba-specific HF auxiliaries (``d_conv`` from convolution kernel size, plus the
    convolution and dt-projection effective bias flags). These have no flat Fast-LLM analogue."""
    return {
        ("d_conv",): config.convolution_layer.kernel_size,
        ("conv_bias",): config.convolution_layer.bias.enabled,
        ("dt_proj_bias",): config.dt_layer.bias.enabled,
    }


def _apriel2_mamba_aux_import(hf_dict: dict) -> dict:
    """Reverse of :func:`_apriel2_mamba_aux_export`. ``conv_bias`` / ``dt_proj_bias`` can diverge from the
    mixer-wide ``add_linear_biases`` flag, so they must populate the per-layer ``bias.enabled`` directly;
    dropping them on import would silently mask HF bias weights when the weight loader checks the
    per-layer flag."""
    out: dict = {}
    if "d_conv" in hf_dict:
        out[("convolution_layer", "kernel_size")] = hf_dict["d_conv"]
    if "conv_bias" in hf_dict:
        out[("convolution_layer", "bias", "enabled")] = hf_dict["conv_bias"]
    if "dt_proj_bias" in hf_dict:
        out[("dt_layer", "bias", "enabled")] = hf_dict["dt_proj_bias"]
    return out


class Apriel2MambaConverter(ConfigSectionConverter):
    fast_llm_config_class = MambaConfig
    hf_type_name = "mamba"

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "state_size": RenameConfigConverter(("state_size",), ("state_size",)),
            "d_inner": RenameConfigConverter(("d_inner",), ("d_inner",)),
            "add_linear_biases": RenameConfigConverter(("add_linear_biases",), ("add_linear_biases",)),
            "d_xb": RenameConfigConverter(("d_xb",), ("d_xb",)),
            "dt_rank": RenameConfigConverter(("dt_rank",), ("dt_rank",)),
            "aux": CustomConfigConverter(
                fast_llm_paths=(("convolution_layer",), ("dt_layer",)),
                hf_paths=(("d_conv",), ("conv_bias",), ("dt_proj_bias",)),
                export_fn=_apriel2_mamba_aux_export,
                import_fn=_apriel2_mamba_aux_import,
                fast_llm_recurses=True,
            ),
            # Architecture fields with no HF counterpart; they round-trip at their Fast-LLM defaults.
            "layers_unmapped": IgnoredConfigConverter(
                ("z_layer",),
                ("x_layer",),
                ("b_layer",),
                ("c_layer",),
                ("output_layer",),
                ("dt_input_layer",),
                ("a_log_weight",),
                ("d_weight",),
                ("repeat_kv_before_conv",),
            ),
        }

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        # dt_proj and convolution read per-layer ``bias.enabled`` directly (no fallback to the mixer-wide
        # flag — Apriel2's HF surfaces these biases via the dedicated ``dt_proj_bias`` / ``conv_bias``
        # auxiliary keys rather than via ``add_linear_biases``).
        return {
            "in_proj": LinearWeightConverter("in_proj", "in_proj"),
            "dt_in_proj": LinearWeightConverter("dt_in_proj", "dt_in_proj"),
            "dt_proj": LinearWeightConverter("dt_proj", "dt_proj", bias_fn=lambda c: c.dt_layer.bias.enabled),
            "convolution": LinearWeightConverter(
                "convolution", "conv1d", bias_fn=lambda c: c.convolution_layer.bias.enabled
            ),
            "A_log": WeightConverter("A_log", "A_log"),
            "D": WeightConverter("D", "D"),
            "out_proj": LinearWeightConverter("out_proj", "out_proj"),
        }


class Apriel2GatedDeltaNetConverter(ConfigSectionConverter):
    fast_llm_config_class = GatedDeltaNetConfig
    hf_type_name = "gdn"

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "value_heads": RenameConfigConverter(("value_heads",), ("value_heads",)),
            "key_heads": RenameConfigConverter(("key_heads",), ("key_heads",)),
            "key_head_dim": RenameConfigConverter(("key_head_dim",), ("key_head_dim",)),
            "value_head_dim": RenameConfigConverter(("value_head_dim",), ("value_head_dim",)),
            "convolution_layer_kernel": _apriel2_kernel_size_only_conv_converter(),
            # CausalConv1dConfig sub-fields the Apriel2 HF format does not surface (weight rides the tensor
            # side; bias/activation round-trip at their Fast-LLM defaults).
            "convolution_layer_unmapped": IgnoredConfigConverter(
                ("convolution_layer", "weight"),
                ("convolution_layer", "bias"),
                ("convolution_layer", "activation"),
            ),
            # Architecture fields not surfaced in HF; round-trip at default.
            "layers_unmapped": IgnoredConfigConverter(
                ("normalization",),
                ("qkv_projection_layer",),
                ("ba_projection_layer",),
                ("output_layer",),
                ("dt_bias_weight",),
                ("a_log_weight",),
            ),
        }

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        no_bias = lambda c: False
        return {
            "in_proj_qkvz": LinearWeightConverter("in_proj_qkvz", "in_proj_qkvz", bias_fn=no_bias),
            "in_proj_ba": LinearWeightConverter("in_proj_ba", "in_proj_ba", bias_fn=no_bias),
            "convolution": LinearWeightConverter(
                "convolution", "convolution", bias_fn=lambda c: c.convolution_layer.bias.enabled
            ),
            "out_proj": LinearWeightConverter("out_proj", "out_proj", bias_fn=no_bias),
            "dt_bias": WeightConverter("dt_bias", "dt_bias"),
            "A_log": WeightConverter("A_log", "A_log"),
            "norm": NestedWeightConverter("norm", "norm", LlamaNormalizationConverter, config_attr="normalization"),
        }


class Apriel2KimiDeltaAttentionConverter(ConfigSectionConverter):
    fast_llm_config_class = KimiDeltaAttentionConfig
    hf_type_name = "kda"

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "heads": RenameConfigConverter(("heads",), ("heads",)),
            "head_dim": RenameConfigConverter(("head_dim",), ("head_dim",)),
            "convolution_layer_kernel": _apriel2_kernel_size_only_conv_converter(),
            # CausalConv1dConfig sub-fields not surfaced in HF (same as :class:`Apriel2GatedDeltaNetConverter`).
            "convolution_layer_unmapped": IgnoredConfigConverter(
                ("convolution_layer", "weight"),
                ("convolution_layer", "bias"),
                ("convolution_layer", "activation"),
            ),
            "normalization_epsilon": CustomConfigConverter(
                fast_llm_paths=(("normalization",), ("normalization", "epsilon")),
                hf_paths=(("normalization",),),
                export_fn=lambda c: {("normalization",): {"epsilon": c.normalization.epsilon}},
                import_fn=lambda hf: ({("normalization",): hf["normalization"]} if "normalization" in hf else {}),
            ),
            # Other GatedRMSNormalizationConfig architecture fields are dropped on the HF side.
            "normalization_unmapped": IgnoredConfigConverter(
                ("normalization", "weight"),
                ("normalization", "zero_centered"),
            ),
            # Architecture fields not surfaced in HF; round-trip at default.
            "layers_unmapped": IgnoredConfigConverter(
                ("q_projection_layer",),
                ("k_projection_layer",),
                ("v_projection_layer",),
                ("f_a_projection_layer",),
                ("f_b_projection_layer",),
                ("g_a_projection_layer",),
                ("g_b_projection_layer",),
                ("beta_projection_layer",),
                ("output_projection_layer",),
                ("dt_bias_weight",),
                ("a_log_weight",),
            ),
        }

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        no_bias = lambda c: False
        proj_names = (
            "q_proj",
            "k_proj",
            "v_proj",
            "q_conv",
            "k_conv",
            "v_conv",
            "f_a_proj",
            "f_b_proj",
            "g_a_proj",
            "g_b_proj",
            "beta_proj",
            "o_proj",
        )
        return {
            **{name: LinearWeightConverter(name, name, bias_fn=no_bias) for name in proj_names},
            "A_log": WeightConverter("A_log", "A_log"),
            "dt_bias": WeightConverter("dt_bias", "dt_bias"),
            "norm": NestedWeightConverter("norm", "norm", LlamaNormalizationConverter, config_attr="normalization"),
        }


# Mixer dispatch registry — used inside StochasticMixer (no nested-stochastic) and at the block level.
APRIEL2_LEAF_MIXER_REGISTRY: dict[type[Config], type[ConfigSectionConverter]] = {
    AttentionConfig: Apriel2AttentionConverter,
    MambaConfig: Apriel2MambaConverter,
    GatedDeltaNetConfig: Apriel2GatedDeltaNetConverter,
    KimiDeltaAttentionConfig: Apriel2KimiDeltaAttentionConverter,
}


class Apriel2StochasticMixerConverter(ConfigSectionConverter):
    fast_llm_config_class = StochasticMixerConfig
    hf_type_name = "stochastic"

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "mixers": TypedDictContainerConfigConverter(
                fast_llm_path=("mixers",),
                hf_path=("mixers",),
                registry=APRIEL2_LEAF_MIXER_REGISTRY,
            ),
            "main_mixer_name": RenameConfigConverter(("main_mixer_name",), ("main_mixer_name",)),
            "sampling_strategy": OptionalConfigConverter(
                ("sampling_strategy",),
                ("sampling_strategy",),
                sentinel=StochasticMixerSamplingStrategy.uniform,
            ),
        }

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {"mixers": TypedDictWeightConverter("mixers", "mixers", APRIEL2_LEAF_MIXER_REGISTRY)}


# Block-level mixer registry includes StochasticMixer (which can wrap leaf mixers).
APRIEL2_BLOCK_MIXER_REGISTRY: dict[type[Config], type[ConfigSectionConverter]] = {
    **APRIEL2_LEAF_MIXER_REGISTRY,
    StochasticMixerConfig: Apriel2StochasticMixerConverter,
}


# ============================================================
# Normalization converters
# ============================================================


class Apriel2RMSNormConverter(LlamaNormalizationConverter):
    """Apriel2's typed ``{type: "rms_norm", epsilon: ...}`` form. Identical to
    :class:`LlamaNormalizationConverter` except the HF epsilon key is ``epsilon`` (not ``rms_norm_eps``)
    and the type discriminator is auto-injected by NestedConfigConverter/DispatchConfigConverter via
    ``hf_type_name`` rather than declared as a flat ``ConstantImport``.
    """

    hf_type_name = "rms_norm"

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            **super()._create_config_converters(),
            "epsilon": RenameConfigConverter(("epsilon",), ("epsilon",)),
        }


class Apriel2LayerNormConverter(ConfigSectionConverter):
    fast_llm_config_class = LayerNormalizationConfig
    hf_type_name = "layer_norm"

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "epsilon": RenameConfigConverter(("epsilon",), ("epsilon",)),
            "weight": IgnoredConfigConverter(("weight",)),
            "bias": IgnoredConfigConverter(("bias",)),
            "zero_centered": ConstantImportConfigConverter(("zero_centered",), False),
        }


class Apriel2NoNormConverter(ConfigSectionConverter):
    """No-op normalization. NoNormalizationConfig carries no fields, so the converter dict is empty —
    the class exists solely as a registry entry for :class:`APRIEL2_NORM_REGISTRY` dispatch."""

    fast_llm_config_class = NoNormalizationConfig
    hf_type_name = "none"

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {}


APRIEL2_NORM_REGISTRY: dict[type[Config], type[ConfigSectionConverter]] = {
    RMSNormalizationConfig: Apriel2RMSNormConverter,
    LayerNormalizationConfig: Apriel2LayerNormConverter,
    NoNormalizationConfig: Apriel2NoNormConverter,
}


# ============================================================
# MLP, Block, Decoder, Head
# ============================================================


class Apriel2MLPConverter(ConfigSectionConverter):
    fast_llm_config_class = MLPConfig
    hf_type_name = "mlp"

    @classmethod
    def _create_config_converters(cls) -> dict:
        layer_names = ("layer_1", "layer_2")
        return {
            "intermediate_size": RenameConfigConverter(("intermediate_size",), ("intermediate_size",)),
            "gated": RenameConfigConverter(("gated",), ("gated",)),
            "add_linear_biases": RenameConfigConverter(("add_linear_biases",), ("add_linear_biases",)),
            "activation": CustomConfigConverter(
                fast_llm_paths=(("activation",),),
                hf_paths=(("activation",),),
                export_fn=lambda c: {("activation",): c.activation.hf_name},
                import_fn=lambda hf: {("activation",): ActivationType.from_hf_name(hf["activation"])},
            ),
            "layers": _per_layer_bias_converter(layer_names),
            "pre_norm": ConstantImportConfigConverter(("pre_norm",), None),
            "post_norm": ConstantImportConfigConverter(("post_norm",), None),
        }

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        # ``layer_1`` splits into ``(gate_proj, up_proj)`` when gated, but stays as a single ``up_proj``
        # otherwise. The transform and HF prefix both depend on the live config; resolve at emit time.
        return {
            "layer_1": LinearWeightConverter(
                "layer_1",
                lambda c: ("gate_proj", "up_proj") if c.gated else ("up_proj",),
                transform=SplitWeightConverter,
                bias_fn=lambda c: _effective_bias(c.layer_1, c.add_linear_biases),
            ),
            "layer_2": LinearWeightConverter(
                "layer_2",
                "down_proj",
                transform=TransposeSplitWeightConverter,
                bias_fn=lambda c: _effective_bias(c.layer_2, c.add_linear_biases),
            ),
        }


class Apriel2BlockConverter(ConfigSectionConverter):
    fast_llm_config_class = DecoderBlockConfig

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "mixer": DispatchConfigConverter(
                fast_llm_path=("mixer",),
                hf_path=("mixer",),
                registry=APRIEL2_BLOCK_MIXER_REGISTRY,
            ),
            "mlp": NestedConfigConverter(("mlp",), Apriel2MLPConverter, hf_path=("mlp",)),
            "normalization": DispatchConfigConverter(
                fast_llm_path=("normalization",),
                hf_path=("normalization",),
                registry=APRIEL2_NORM_REGISTRY,
            ),
            "pre_mixer_normalization": ConstantImportConfigConverter(("pre_mixer_normalization",), None),
            "pre_mlp_normalization": ConstantImportConfigConverter(("pre_mlp_normalization",), None),
            "post_mixer_normalization": ConstantImportConfigConverter(("post_mixer_normalization",), None),
            "post_mlp_normalization": ConstantImportConfigConverter(("post_mlp_normalization",), None),
            "output_scale": IgnoredConfigConverter(("output_scale",)),
        }

    @classmethod
    def _validate_export(cls, config: DecoderBlockConfig) -> None:
        # Apriel2 HF format only represents plain MLP. ``NestedConfigConverter`` dispatches by fixed class
        # (``Apriel2MLPConverter`` registered against ``MLPConfig``) and would silently descend into a
        # ``MoEMLPConfig`` via MRO, dropping every MoE-specific architecture field.
        # Strict type to reject MoEMLPConfig subclass — not isinstance.
        Assert.is_(type(config.mlp), MLPConfig)
        # The config side dispatches normalization through APRIEL2_NORM_REGISTRY (RMS/Layer/None), but the
        # weight side below hardcodes LlamaNormalizationConverter (RMS-only). Fail loudly here so a
        # LayerNorm/NoNorm block config doesn't silently produce phantom norm_1.weight/norm_2.weight.
        Assert.is_(type(config.normalization), RMSNormalizationConfig)
        Assert.custom(lambda v: not v, config.output_scale.enabled)

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "mixer": DispatchWeightConverter("mixer", "mixer", APRIEL2_BLOCK_MIXER_REGISTRY),
            "mlp": NestedWeightConverter("mlp", "mlp", Apriel2MLPConverter),
            # The two state-dict norms (norm_1/norm_2) share the block's single ``normalization`` config.
            "norm_1": NestedWeightConverter(
                "norm_1", "input_layernorm", LlamaNormalizationConverter, config_attr="normalization"
            ),
            "norm_2": NestedWeightConverter(
                "norm_2", "post_attention_layernorm", LlamaNormalizationConverter, config_attr="normalization"
            ),
        }


class Apriel2FixedDecoderConverter(ConfigSectionConverter):
    fast_llm_config_class = FixedBlockSequenceConfig
    hf_type_name = "fixed"
    block_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = Apriel2BlockConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "num_blocks": RenameConfigConverter(("num_blocks",), ("num_blocks",)),
            "block": NestedConfigConverter(("block",), cls.block_converter_class, hf_path=("block",)),
        }

    # The block fan-out lives on the base-model converter, which uses :class:`BlockSequenceWeightConverter`
    # directly (Fixed/Pattern dispatch and block iteration share one primitive). The Fixed/Pattern decoder
    # section converters exist for the config side (dispatch via :class:`DispatchConfigConverter`) and
    # contribute no weights of their own.


class Apriel2PatternDecoderConverter(ConfigSectionConverter):
    fast_llm_config_class = PatternBlockSequenceConfig
    hf_type_name = "pattern"
    block_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = Apriel2BlockConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "num_blocks": RenameConfigConverter(("num_blocks",), ("num_blocks",)),
            "pattern": RenameConfigConverter(("pattern",), ("pattern",)),
            "blocks": TypedDictContainerConfigConverter(
                fast_llm_path=("blocks",),
                hf_path=("blocks",),
                registry={DecoderBlockConfig: cls.block_converter_class},
            ),
        }

    # See note on :class:`Apriel2FixedDecoderConverter` — block fan-out lives on the base-model converter.


APRIEL2_DECODER_REGISTRY: dict[type[Config], type[ConfigSectionConverter]] = {
    FixedBlockSequenceConfig: Apriel2FixedDecoderConverter,
    PatternBlockSequenceConfig: Apriel2PatternDecoderConverter,
}


def get_apriel2_decoder_converter(
    decoder_config: FixedBlockSequenceConfig | PatternBlockSequenceConfig,
) -> type[ConfigSectionConverter]:
    """Look up the Apriel2 per-shape decoder converter for a given decoder config instance."""
    converter_class = APRIEL2_DECODER_REGISTRY.get(type(decoder_config))
    if converter_class is None:
        raise NotImplementedError(f"Unsupported decoder type: {type(decoder_config).__name__}")
    return converter_class


class Apriel2HeadConverter(ConfigSectionConverter):
    fast_llm_config_class = LanguageModelHeadConfig

    normalization_converter_class: typing.ClassVar[type[LlamaNormalizationConverter]] = LlamaNormalizationConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "normalization": DispatchConfigConverter(
                fast_llm_path=("normalization",),
                hf_path=("normalization",),
                registry=APRIEL2_NORM_REGISTRY,
            ),
            "output_weight": IgnoredConfigConverter(("output_weight",)),
            # Apriel2 HF format does not support multi-token prediction; pin to 1 so any non-default value
            # fails on export instead of silently round-tripping.
            "prediction_heads": ConstantImportConfigConverter(("prediction_heads",), 1),
            "final_logit_softcap": ConstantImportConfigConverter(("final_logit_softcap",), None),
        }

    @classmethod
    def _validate_export(cls, config: LanguageModelHeadConfig) -> None:
        # The config side dispatches normalization through APRIEL2_NORM_REGISTRY (RMS/Layer/None), but the
        # weight side below hardcodes ``normalization_converter_class`` (RMSNorm-only). Fail loudly here so a
        # LayerNorm/NoNorm head config doesn't silently round-trip through the wrong weight conversion.
        Assert.is_(type(config.normalization), RMSNormalizationConfig)

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "final_norm": NestedWeightConverter(
                "final_norm", "model.norm", cls.normalization_converter_class, config_attr="normalization"
            ),
            "output_weights": OutputProjectionWeightConverter("output_weights", "lm_head.weight"),
        }


class Apriel2BaseModelConverter(ConfigSectionConverter):
    fast_llm_config_class = GPTBaseModelConfig

    embeddings_converter_class: typing.ClassVar[type[LlamaEmbeddingsConverter]] = LlamaEmbeddingsConverter
    head_converter_class: typing.ClassVar[type[Apriel2HeadConverter]] = Apriel2HeadConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "embeddings": NestedConfigConverter(("embeddings",), cls.embeddings_converter_class),
            "decoder": DispatchConfigConverter(
                fast_llm_path=("decoder",),
                hf_path=("decoder",),
                registry=APRIEL2_DECODER_REGISTRY,
            ),
            "head": NestedConfigConverter(("head",), cls.head_converter_class, hf_path=("head",)),
            "hidden_size": RenameConfigConverter(("hidden_size",), ("hidden_size",)),
            "tied_embedding_weight": RenameConfigConverter(("tied_embedding_weight",), ("tie_word_embeddings",)),
            "peft": IgnoredConfigConverter(("peft",)),
            # ``Apriel2TextConfig`` default-injects ``{"embeddings": {"max_position_embeddings": 2048}}``
            # the Fast-LLM converter doesn't use — vocab_size rides at top level via the flat-merged
            # ``LlamaEmbeddingsConverter``. Claim only the specific injected leaf so any future field
            # transformers adds to the same subdict trips the HF coverage check.
            "embeddings_subdict_unmapped": IgnoredConfigConverter(
                hf_paths=(("embeddings", "max_position_embeddings"),)
            ),
        }

    @classmethod
    def _validate_export(cls, config: GPTBaseModelConfig) -> None:
        assert_no_peft(config)

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "embeddings": NestedWeightConverter("embeddings", "model", cls.embeddings_converter_class),
            "decoder": BlockSequenceWeightConverter("decoder", "model.decoder.blocks", Apriel2BlockConverter),
            "head": NestedWeightConverter("head", "", cls.head_converter_class),
        }

    @classmethod
    def get_converters(cls, config: GPTBaseModelConfig, exported_config: dict) -> list[WeightConverter]:
        return cls.emit_weight_converters(config, "", "")


class Apriel2HuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    _model: GPTModel
    _model_class: typing.ClassVar[type] = GPTModelConfig
    format: typing.ClassVar[type[CheckpointFormat]] = Apriel2TextCheckpointFormat
    architecture: typing.ClassVar[str] = "Apriel2ForCausalLM"
    base_model_converter_class: typing.ClassVar[type[Apriel2BaseModelConverter]] = Apriel2BaseModelConverter

    @classmethod
    def get_huggingface_model_type(cls) -> str:
        return "apriel2_text"

    @classmethod
    def get_transformers_configuration_class(cls) -> type[PretrainedConfig]:
        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2TextConfig

        return Apriel2TextConfig

    @classmethod
    def get_model_files(cls) -> tuple[str, str, str | None]:
        from fast_llm_external_models.apriel2 import configuration_apriel2, modeling_apriel2

        return configuration_apriel2.__file__, modeling_apriel2.__file__, None

    @classmethod
    def _export_config(cls, config: GPTModelConfig) -> dict[str, typing.Any]:
        base_model = config.base_model
        return safe_merge_dicts(
            cls.base_model_converter_class.export_config(base_model),
            {
                "architectures": [cls.architecture],
                "model_type": cls.get_huggingface_model_type(),
                "auto_map": {
                    "AutoConfig": "configuration_apriel2.Apriel2TextConfig",
                    "AutoModel": "modeling_apriel2.Apriel2TextModel",
                    "AutoModelForCausalLM": "modeling_apriel2.Apriel2ForCausalLM",
                },
            },
        )

    @classmethod
    def _import_config(cls, config: dict[str, typing.Any]) -> dict[str, typing.Any]:
        cls._check_hf_coverage(config)
        return {"base_model": cls.base_model_converter_class.import_config(config)}

    @classmethod
    def _get_weight_converters(cls, config: GPTModelConfig, export_config: dict) -> list[WeightConverter]:
        return cls.base_model_converter_class.get_converters(config.base_model, export_config)
