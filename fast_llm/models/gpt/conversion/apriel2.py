"""Apriel2 text-only checkpoint format converter."""

import typing

from transformers import PretrainedConfig

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    ConfigSectionConverter,
    ConstantImportConfigConverter,
    CustomConfigConverter,
    DispatchConfigConverter,
    IgnoredConfigConverter,
    NestedConfigConverter,
    OptionalConfigConverter,
    RenameConfigConverter,
    TypedDictContainerConfigConverter,
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
    KeyValueWeightConverter,
    LlamaEmbeddingsConverter,
    LlamaNormalizationConverter,
    MLPLayer2Converter,
    SplitWeightConverter,
    assert_no_peft,
    effective_bias,
    get_parameter_converter,
    get_weight_and_bias_converters,
)
from fast_llm.models.gpt.model import GPTModel
from fast_llm.utils import safe_merge_dicts

# ============================================================
# Helpers
# ============================================================


def _per_layer_bias_export(config, layer_names: tuple[str, ...]) -> dict:
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
                recurses=True,
            ),
            # Apriel2 emits add_linear_biases only when False; the True default is implicit.
            "add_linear_biases": OptionalConfigConverter(
                ("add_linear_biases",), ("add_linear_biases",), sentinel=True
            ),
            "window_size": OptionalConfigConverter(("window_size",), ("window_size",)),
            "linear_layers": CustomConfigConverter(
                fast_llm_paths=tuple((name,) for name in layer_names),
                hf_paths=tuple((name,) for name in layer_names),
                export_fn=lambda c: _per_layer_bias_export(c, layer_names),
                import_fn=lambda hf: _per_layer_bias_import(hf, layer_names),
                recurses=True,
            ),
            "causal": IgnoredConfigConverter(("causal",)),
            "softmax_scale_power": IgnoredConfigConverter(("softmax_scale_power",)),
        }

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(
        cls,
        config: AttentionConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        q_bias = effective_bias(config.query_layer, config.add_linear_biases)
        k_bias = effective_bias(config.key_layer, config.add_linear_biases)
        v_bias = effective_bias(config.value_layer, config.add_linear_biases)
        o_bias = effective_bias(config.dense_layer, config.add_linear_biases)
        # k_proj and v_proj are merged in Fast-LLM's key_value layer; treat as biased only if both sides agree.
        kv_bias = k_bias and v_bias

        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.query",
                f"{hf_prefix}.q_proj",
                q_bias,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.key_value",
                (f"{hf_prefix}.k_proj", f"{hf_prefix}.v_proj"),
                kv_bias,
                KeyValueWeightConverter,
                config,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.dense",
                f"{hf_prefix}.o_proj",
                o_bias,
                drop_on_export=drop_on_export,
            ),
        ]


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
                recurses=True,
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

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(
        cls,
        config: MambaConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.in_proj",
                f"{hf_prefix}.in_proj",
                config.add_linear_biases,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.dt_in_proj",
                f"{hf_prefix}.dt_in_proj",
                config.add_linear_biases,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.dt_proj",
                f"{hf_prefix}.dt_proj",
                config.dt_layer.bias.enabled,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.convolution",
                f"{hf_prefix}.conv1d",
                config.convolution_layer.bias.enabled,
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.A_log",
                f"{hf_prefix}.A_log",
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.D",
                f"{hf_prefix}.D",
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.out_proj",
                f"{hf_prefix}.out_proj",
                config.add_linear_biases,
                drop_on_export=drop_on_export,
            ),
        ]


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
            "convolution_layer_kernel": CustomConfigConverter(
                fast_llm_paths=(("convolution_layer",), ("convolution_layer", "kernel_size")),
                hf_paths=(("convolution_layer",),),
                export_fn=lambda c: {("convolution_layer",): {"kernel_size": c.convolution_layer.kernel_size}},
                import_fn=lambda hf: (
                    {("convolution_layer",): hf["convolution_layer"]} if "convolution_layer" in hf else {}
                ),
            ),
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

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(
        cls,
        config: GatedDeltaNetConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.in_proj_qkvz",
                f"{hf_prefix}.in_proj_qkvz",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.in_proj_ba",
                f"{hf_prefix}.in_proj_ba",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.convolution",
                f"{hf_prefix}.convolution",
                config.convolution_layer.bias.enabled,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.out_proj",
                f"{hf_prefix}.out_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.dt_bias",
                f"{hf_prefix}.dt_bias",
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.A_log",
                f"{hf_prefix}.A_log",
                drop_on_export=drop_on_export,
            ),
            *LlamaNormalizationConverter.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.norm",
                f"{hf_prefix}.norm",
                drop_on_export=drop_on_export,
            ),
        ]


class Apriel2KimiDeltaAttentionConverter(ConfigSectionConverter):
    fast_llm_config_class = KimiDeltaAttentionConfig
    hf_type_name = "kda"

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "heads": RenameConfigConverter(("heads",), ("heads",)),
            "head_dim": RenameConfigConverter(("head_dim",), ("head_dim",)),
            "convolution_layer_kernel": CustomConfigConverter(
                fast_llm_paths=(("convolution_layer",), ("convolution_layer", "kernel_size")),
                hf_paths=(("convolution_layer",),),
                export_fn=lambda c: {("convolution_layer",): {"kernel_size": c.convolution_layer.kernel_size}},
                import_fn=lambda hf: (
                    {("convolution_layer",): hf["convolution_layer"]} if "convolution_layer" in hf else {}
                ),
            ),
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

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(
        cls,
        config: KimiDeltaAttentionConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.q_proj",
                f"{hf_prefix}.q_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.k_proj",
                f"{hf_prefix}.k_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.v_proj",
                f"{hf_prefix}.v_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.q_conv",
                f"{hf_prefix}.q_conv",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.k_conv",
                f"{hf_prefix}.k_conv",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.v_conv",
                f"{hf_prefix}.v_conv",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.f_a_proj",
                f"{hf_prefix}.f_a_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.f_b_proj",
                f"{hf_prefix}.f_b_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.g_a_proj",
                f"{hf_prefix}.g_a_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.g_b_proj",
                f"{hf_prefix}.g_b_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.beta_proj",
                f"{hf_prefix}.beta_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.o_proj",
                f"{hf_prefix}.o_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.A_log",
                f"{hf_prefix}.A_log",
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.dt_bias",
                f"{hf_prefix}.dt_bias",
                drop_on_export=drop_on_export,
            ),
            *LlamaNormalizationConverter.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.norm",
                f"{hf_prefix}.norm",
                drop_on_export=drop_on_export,
            ),
        ]


# Mixer dispatch registry — used inside StochasticMixer (no nested-stochastic) and at the block level.
APRIEL2_LEAF_MIXER_REGISTRY: dict = {
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

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(
        cls,
        config: StochasticMixerConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        converters = []
        for name, sub_mixer in config.mixers.items():
            converter_class = APRIEL2_LEAF_MIXER_REGISTRY.get(type(sub_mixer))
            if converter_class is None:
                raise ValueError(f"Unknown sub-mixer type: {type(sub_mixer)}")
            converters.extend(
                converter_class.get_converters(
                    sub_mixer,
                    f"{fast_llm_prefix}.mixers.{name}",
                    f"{hf_prefix}.mixers.{name}",
                    drop_on_export=drop_on_export,
                )
            )
        return converters


# Block-level mixer registry includes StochasticMixer (which can wrap leaf mixers).
APRIEL2_BLOCK_MIXER_REGISTRY: dict = {
    **APRIEL2_LEAF_MIXER_REGISTRY,
    StochasticMixerConfig: Apriel2StochasticMixerConverter,
}


# ============================================================
# Normalization converters
# ============================================================


class Apriel2RMSNormConverter(ConfigSectionConverter):
    fast_llm_config_class = RMSNormalizationConfig
    hf_type_name = "rms_norm"

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "epsilon": RenameConfigConverter(("epsilon",), ("epsilon",)),
            "weight": IgnoredConfigConverter(("weight",)),
            "zero_centered": ConstantImportConfigConverter(("zero_centered",), False),
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
    fast_llm_config_class = NoNormalizationConfig
    hf_type_name = "none"

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {}


APRIEL2_NORM_REGISTRY: dict = {
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
            "layers": CustomConfigConverter(
                fast_llm_paths=tuple((name,) for name in layer_names),
                hf_paths=tuple((name,) for name in layer_names),
                export_fn=lambda c: _per_layer_bias_export(c, layer_names),
                import_fn=lambda hf: _per_layer_bias_import(hf, layer_names),
                recurses=True,
            ),
        }

    @classmethod
    def get_converters(
        cls,
        config: MLPConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        layer_1_bias = effective_bias(config.layer_1, config.add_linear_biases)
        layer_2_bias = effective_bias(config.layer_2, config.add_linear_biases)
        if config.gated:
            return [
                *get_weight_and_bias_converters(
                    f"{fast_llm_prefix}.layer_1",
                    (f"{hf_prefix}.gate_proj", f"{hf_prefix}.up_proj"),
                    layer_1_bias,
                    SplitWeightConverter,
                    drop_on_export=drop_on_export,
                ),
                *get_weight_and_bias_converters(
                    f"{fast_llm_prefix}.layer_2",
                    f"{hf_prefix}.down_proj",
                    layer_2_bias,
                    MLPLayer2Converter,
                    drop_on_export=drop_on_export,
                ),
            ]
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_1",
                f"{hf_prefix}.up_proj",
                layer_1_bias,
                WeightConverter,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_2",
                f"{hf_prefix}.down_proj",
                layer_2_bias,
                MLPLayer2Converter,
                drop_on_export=drop_on_export,
            ),
        ]


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
        }

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(
        cls,
        config: DecoderBlockConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        mixer_converter_class = APRIEL2_BLOCK_MIXER_REGISTRY.get(type(config.mixer))
        if mixer_converter_class is None:
            raise ValueError(f"Unknown mixer type: {type(config.mixer)}")
        converters: list[WeightConverter] = list(
            mixer_converter_class.get_converters(
                config.mixer,
                f"{fast_llm_prefix}.mixer",
                f"{hf_prefix}.mixer",
                drop_on_export=drop_on_export,
            )
        )
        converters.extend(
            Apriel2MLPConverter.get_converters(
                config.mlp,
                f"{fast_llm_prefix}.mlp",
                f"{hf_prefix}.mlp",
                drop_on_export=drop_on_export,
            )
        )
        converters.extend(
            [
                *LlamaNormalizationConverter.get_converters(
                    config.normalization,
                    f"{fast_llm_prefix}.norm_1",
                    f"{hf_prefix}.input_layernorm",
                    drop_on_export=drop_on_export,
                ),
                *LlamaNormalizationConverter.get_converters(
                    config.normalization,
                    f"{fast_llm_prefix}.norm_2",
                    f"{hf_prefix}.post_attention_layernorm",
                    drop_on_export=drop_on_export,
                ),
            ]
        )
        return converters


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

    @classmethod
    def get_converters(
        cls,
        config: FixedBlockSequenceConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        converters: list[WeightConverter] = []
        for block_index in range(config.num_blocks):
            converters += cls.block_converter_class.get_converters(
                config.block,
                f"{fast_llm_prefix}.{block_index}",
                f"{hf_prefix}.{block_index}",
                drop_on_export=drop_on_export,
            )
        return converters


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

    @classmethod
    def get_converters(
        cls,
        config: PatternBlockSequenceConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        converters: list[WeightConverter] = []
        for block_index in range(config.num_blocks):
            block_config = config.blocks[config.pattern[block_index % len(config.pattern)]]
            converters += cls.block_converter_class.get_converters(
                block_config,
                f"{fast_llm_prefix}.{block_index}",
                f"{hf_prefix}.{block_index}",
                drop_on_export=drop_on_export,
            )
        return converters


APRIEL2_DECODER_REGISTRY: dict = {
    FixedBlockSequenceConfig: Apriel2FixedDecoderConverter,
    PatternBlockSequenceConfig: Apriel2PatternDecoderConverter,
}


def get_apriel2_decoder_converter(decoder_config) -> "type[ConfigSectionConverter]":
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
            "prediction_heads": IgnoredConfigConverter(("prediction_heads",)),
        }

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(
        cls,
        config,
        exported_config: dict,
        fast_llm_prefix: str,
    ) -> list[WeightConverter]:
        return [
            *cls.normalization_converter_class.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.final_norm",
                "model.norm",
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.output_weights",
                "lm_head.weight",
                drop_on_import=exported_config.get("tie_word_embeddings", False),
                drop_on_export=exported_config.get("tie_word_embeddings", False),
            ),
        ]


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
            # ``Apriel2TextConfig`` default-injects an ``embeddings`` HF subdict
            # (``{"max_position_embeddings": 2048}``) the Fast-LLM converter doesn't use — vocab_size
            # rides at top level via the flat-merged ``LlamaEmbeddingsConverter``. Claim the injected
            # subdict so the HF coverage check doesn't flag it.
            "embeddings_subdict_unmapped": IgnoredConfigConverter(hf_paths=(("embeddings",),)),
        }

    @classmethod
    def _validate_export(cls, config: GPTBaseModelConfig) -> None:
        assert_no_peft(config)

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(cls, config: GPTBaseModelConfig, exported_config: dict) -> list[WeightConverter]:
        return [
            *cls.embeddings_converter_class.get_converters(config.embeddings, "embeddings", "model"),
            *get_apriel2_decoder_converter(config.decoder).get_converters(
                config.decoder, "decoder", "model.decoder.blocks"
            ),
            *cls.head_converter_class.get_converters(config.head, exported_config, "head"),
        ]


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
