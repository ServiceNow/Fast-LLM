"""Apriel2 text-only checkpoint format converter."""

import typing

from transformers import PretrainedConfig

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import WeightConverter
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig, StochasticMixerConfig
from fast_llm.layers.ssm.config import GatedDeltaNetConfig, KimiDeltaAttentionConfig, MambaConfig
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.models.gpt.conversion.config import Apriel2TextCheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    KeyValueWeightConverter,
    LlamaEmbeddingsConverter,
    LlamaNormalizationConverter,
    MLPLayer2Converter,
    QueryWeightConverter,
    SplitWeightConverter,
    get_parameter_converter,
    get_weight_and_bias_converters,
)
from fast_llm.models.gpt.model import GPTModel
from fast_llm.utils import Assert, safe_merge_dicts


class Apriel2AttentionConverter:
    @classmethod
    def import_config(cls, config: dict) -> dict:
        rotary = config["rotary"]
        # Map Apriel2 HuggingFace rotary type to Fast-LLM internal type
        if rotary.get("type") == "mistral_1d":
            rotary = {**rotary, "type": "default"}
        result = {
            "type": "attention",
            "heads": config["heads"],
            "head_groups": config["head_groups"],
            "head_size": config["head_size"],
            "rotary": rotary,
            "add_linear_biases": config["add_linear_biases"],
        }
        if "window_size" in config:
            result["window_size"] = config["window_size"]
        return result

    @classmethod
    def export_config(cls, config: AttentionConfig) -> dict:
        from fast_llm.layers.attention.rotary.config import DefaultRotaryConfig, Llama3RotaryConfig, YarnRotaryConfig

        if type(config.rotary) is DefaultRotaryConfig:
            rotary_type = "mistral_1d"
        elif type(config.rotary) is Llama3RotaryConfig:
            rotary_type = "llama3"
        elif type(config.rotary) is YarnRotaryConfig:
            rotary_type = "yarn"
        else:
            raise NotImplementedError(f"Unsupported rotary type: {type(config.rotary).__name__}")

        return {
            "type": "attention",
            "heads": config.heads,
            "head_groups": config.head_groups,
            "head_size": config.head_size,
            "add_linear_biases": config.add_linear_biases,
            "rotary": {
                "type": rotary_type,
                "theta": config.rotary.theta,
            },
            "window_size": config.window_size,
        }

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


class Apriel2MambaConverter:
    @classmethod
    def import_config(cls, config: dict) -> dict:
        result = {
            "type": "mamba",
            "state_size": config["state_size"],
            "d_inner": config["d_inner"],
            "add_linear_biases": config["add_linear_biases"],
        }
        if "d_xb" in config:
            result["d_xb"] = config["d_xb"]
        if "dt_rank" in config:
            result["dt_rank"] = config["dt_rank"]
        return result

    @classmethod
    def export_config(cls, config: MambaConfig) -> dict:
        exported = {
            "type": "mamba",
            "state_size": config.state_size,
            "d_inner": config.d_inner,
            "d_conv": config.convolution_layer.kernel_size,
            "add_linear_biases": config.add_linear_biases,
            "conv_bias": config.convolution_layer.bias.enabled,
            "dt_proj_bias": config.dt_layer.bias.enabled,
        }

        if config.d_xb is not None:
            exported["d_xb"] = config.d_xb

        if config.dt_rank != "auto":
            exported["dt_rank"] = config.dt_rank

        return exported

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


class Apriel2GatedDeltaNetConverter:
    @classmethod
    def import_config(cls, config: dict) -> dict:
        result = {
            "type": "gdn",
            "value_heads": config["value_heads"],
            "key_heads": config["key_heads"],
            "key_head_dim": config["key_head_dim"],
            "value_head_dim": config["value_head_dim"],
        }
        if "convolution_layer" in config:
            result["convolution_layer"] = config["convolution_layer"]
        return result

    @classmethod
    def export_config(cls, config: GatedDeltaNetConfig) -> dict:
        return {
            "type": "gdn",
            "value_heads": config.value_heads,
            "key_heads": config.key_heads,
            "key_head_dim": config.key_head_dim,
            "value_head_dim": config.value_head_dim,
            "convolution_layer": {
                "kernel_size": config.convolution_layer.kernel_size,
            },
        }

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


class Apriel2KimiDeltaAttentionConverter:
    @classmethod
    def import_config(cls, config: dict) -> dict:
        result = {
            "type": "kda",
            "heads": config["heads"],
            "head_dim": config["head_dim"],
        }
        if "convolution_layer" in config:
            result["convolution_layer"] = config["convolution_layer"]
        if "normalization" in config:
            result["normalization"] = config["normalization"]
        return result

    @classmethod
    def export_config(cls, config: KimiDeltaAttentionConfig) -> dict:
        return {
            "type": "kda",
            "heads": config.heads,
            "head_dim": config.head_dim,
            "convolution_layer": {
                "kernel_size": config.convolution_layer.kernel_size,
            },
            "normalization": {
                "epsilon": config.normalization.epsilon,
            },
        }

    @classmethod
    def get_converters(
        cls,
        config: KimiDeltaAttentionConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        # Fast-LLM KDA uses abbreviated names matching the external module:
        # q_proj, k_proj, v_proj, q_conv, k_conv, v_conv, f_a_proj, f_b_proj,
        # g_a_proj, g_b_proj, beta_proj, o_proj, A_log, dt_bias, norm
        return [
            # Q/K/V projections
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
            # Convolutions (Q, K, V)
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
            # Gate projections (f_a, f_b, g_a, g_b)
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
            # Beta projection
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.beta_proj",
                f"{hf_prefix}.beta_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            # Output projection
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.o_proj",
                f"{hf_prefix}.o_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            # Learnable parameters
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
            # Normalization
            *LlamaNormalizationConverter.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.norm",
                f"{hf_prefix}.norm",
                drop_on_export=drop_on_export,
            ),
        ]


class Apriel2StochasticMixerConverter:
    @classmethod
    def import_config(cls, config: dict) -> dict:
        mixers = {}
        for name, sub_mixer_config in config["mixers"].items():
            mixer_type = sub_mixer_config["type"]
            if mixer_type == "attention":
                mixers[name] = Apriel2AttentionConverter.import_config(sub_mixer_config)
            elif mixer_type == "mamba":
                mixers[name] = Apriel2MambaConverter.import_config(sub_mixer_config)
            elif mixer_type == "gdn":
                mixers[name] = Apriel2GatedDeltaNetConverter.import_config(sub_mixer_config)
            elif mixer_type == "kda":
                mixers[name] = Apriel2KimiDeltaAttentionConverter.import_config(sub_mixer_config)
            else:
                raise ValueError(f"Unknown sub-mixer type: {mixer_type}")

        result = {
            "type": "stochastic",
            "mixers": mixers,
            "main_mixer_name": config["main_mixer_name"],
        }
        if "sampling_strategy" in config:
            result["sampling_strategy"] = config["sampling_strategy"]
        return result

    @classmethod
    def export_config(cls, config: StochasticMixerConfig) -> dict:
        mixers = {}
        for name, sub_mixer in config.mixers.items():
            mixer_type = type(sub_mixer)
            if mixer_type is AttentionConfig:
                mixers[name] = Apriel2AttentionConverter.export_config(sub_mixer)
            elif mixer_type is MambaConfig:
                mixers[name] = Apriel2MambaConverter.export_config(sub_mixer)
            elif mixer_type is GatedDeltaNetConfig:
                mixers[name] = Apriel2GatedDeltaNetConverter.export_config(sub_mixer)
            elif mixer_type is KimiDeltaAttentionConfig:
                mixers[name] = Apriel2KimiDeltaAttentionConverter.export_config(sub_mixer)
            else:
                raise ValueError(f"Unknown sub-mixer type: {mixer_type}")

        return {
            "type": "stochastic",
            "mixers": mixers,
            "main_mixer_name": config.main_mixer_name,
            "sampling_strategy": config.sampling_strategy.value,
        }

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
            mixer_type = type(sub_mixer)
            if mixer_type is AttentionConfig:
                converter_class = Apriel2AttentionConverter
                hf_sub_mixer_prefix = f"{hf_prefix}.mixers.{name}"
            elif mixer_type is MambaConfig:
                converter_class = Apriel2MambaConverter
                hf_sub_mixer_prefix = f"{hf_prefix}.mixers.{name}"
            elif mixer_type is GatedDeltaNetConfig:
                converter_class = Apriel2GatedDeltaNetConverter
                hf_sub_mixer_prefix = f"{hf_prefix}.mixers.{name}"
            elif mixer_type is KimiDeltaAttentionConfig:
                converter_class = Apriel2KimiDeltaAttentionConverter
                hf_sub_mixer_prefix = f"{hf_prefix}.mixers.{name}"
            else:
                raise ValueError(f"Unknown sub-mixer type: {mixer_type}")
            converters.extend(
                converter_class.get_converters(
                    sub_mixer,
                    f"{fast_llm_prefix}.mixers.{name}",
                    hf_sub_mixer_prefix,
                    drop_on_export=drop_on_export,
                )
            )

        return converters


class Apriel2BlockConverter:
    @classmethod
    def import_config(cls, config: dict, block_config: dict) -> dict:
        mixer_config = block_config["mixer"]
        mixer_type = mixer_config["type"]

        if mixer_type == "attention":
            mixer = Apriel2AttentionConverter.import_config(mixer_config)
        elif mixer_type == "mamba":
            mixer = Apriel2MambaConverter.import_config(mixer_config)
        elif mixer_type == "stochastic":
            mixer = Apriel2StochasticMixerConverter.import_config(mixer_config)
        elif mixer_type == "gdn":
            mixer = Apriel2GatedDeltaNetConverter.import_config(mixer_config)
        elif mixer_type == "kda":
            mixer = Apriel2KimiDeltaAttentionConverter.import_config(mixer_config)
        else:
            raise ValueError(f"Unknown mixer type: {mixer_type}")

        from fast_llm.functional.config import ActivationType

        mlp_config = block_config["mlp"]
        mlp = {
            "type": "mlp",
            "intermediate_size": mlp_config["intermediate_size"],
            "activation": ActivationType.from_hf_name(mlp_config["activation"]),
            "gated": True,
            "add_linear_biases": mlp_config["add_linear_biases"],
        }

        normalization = block_config["normalization"]

        return {
            "mixer": mixer,
            "mlp": mlp,
            "normalization": normalization,
        }

    @classmethod
    def export_config(cls, config: DecoderBlockConfig) -> dict:
        from fast_llm.layers.common.normalization.config import (
            LayerNormalizationConfig,
            NoNormalizationConfig,
            RMSNormalizationConfig,
        )

        mixer_type = type(config.mixer)

        if mixer_type is AttentionConfig:
            mixer = Apriel2AttentionConverter.export_config(config.mixer)
        elif mixer_type is MambaConfig:
            mixer = Apriel2MambaConverter.export_config(config.mixer)
        elif mixer_type is StochasticMixerConfig:
            mixer = Apriel2StochasticMixerConverter.export_config(config.mixer)
        elif mixer_type is GatedDeltaNetConfig:
            mixer = Apriel2GatedDeltaNetConverter.export_config(config.mixer)
        elif mixer_type is KimiDeltaAttentionConfig:
            mixer = Apriel2KimiDeltaAttentionConverter.export_config(config.mixer)
        else:
            raise ValueError(f"Unknown mixer type: {mixer_type}")

        norm_type = type(config.normalization)
        if norm_type is RMSNormalizationConfig:
            norm_type_str = "rms_norm"
        elif norm_type is LayerNormalizationConfig:
            norm_type_str = "layer_norm"
        elif norm_type is NoNormalizationConfig:
            norm_type_str = "none"
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

        from fast_llm.layers.decoder.mlp.config import MLPConfig

        if not isinstance(config.mlp, MLPConfig):
            raise ValueError(f"Unsupported MLP type: {type(config.mlp)}")

        mlp = {
            "type": "mlp",
            "intermediate_size": config.mlp.intermediate_size,
            "activation": config.mlp.activation.value,
            "gated": config.mlp.gated,
            "add_linear_biases": config.mlp.add_linear_biases,
        }

        normalization = {"type": norm_type_str, "epsilon": config.normalization.epsilon}

        return {
            "mixer": mixer,
            "mlp": mlp,
            "normalization": normalization,
        }

    @classmethod
    def get_converters(
        cls,
        config: DecoderBlockConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        converters = []
        mixer_type = type(config.mixer)
        if mixer_type is AttentionConfig:
            converter_class = Apriel2AttentionConverter
            hf_mixer_prefix = f"{hf_prefix}.mixer"
        elif mixer_type is MambaConfig:
            converter_class = Apriel2MambaConverter
            hf_mixer_prefix = f"{hf_prefix}.mixer"
        elif mixer_type is StochasticMixerConfig:
            converter_class = Apriel2StochasticMixerConverter
            hf_mixer_prefix = f"{hf_prefix}.mixer"
        elif mixer_type is GatedDeltaNetConfig:
            converter_class = Apriel2GatedDeltaNetConverter
            hf_mixer_prefix = f"{hf_prefix}.mixer"
        elif mixer_type is KimiDeltaAttentionConfig:
            converter_class = Apriel2KimiDeltaAttentionConverter
            hf_mixer_prefix = f"{hf_prefix}.mixer"
        else:
            raise ValueError(f"Unknown mixer type: {mixer_type}")

        converters.extend(
            converter_class.get_converters(
                config.mixer,
                f"{fast_llm_prefix}.mixer",
                hf_mixer_prefix,
                drop_on_export=drop_on_export,
            )
        )

        converters.extend(
            [
                *get_weight_and_bias_converters(
                    f"{fast_llm_prefix}.mlp.layer_1",
                    (f"{hf_prefix}.mlp.gate_proj", f"{hf_prefix}.mlp.up_proj"),
                    config.mlp.add_linear_biases,
                    SplitWeightConverter,
                    drop_on_export=drop_on_export,
                ),
                *get_weight_and_bias_converters(
                    f"{fast_llm_prefix}.mlp.layer_2",
                    f"{hf_prefix}.mlp.down_proj",
                    config.mlp.add_linear_biases,
                    MLPLayer2Converter,
                    drop_on_export=drop_on_export,
                ),
            ]
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


class Apriel2DecoderConverter:
    block_converter_class: typing.ClassVar[type[Apriel2BlockConverter]] = Apriel2BlockConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        decoder_config = config["decoder"]
        decoder_type = decoder_config["type"]

        if decoder_type == "fixed":
            block_config = decoder_config["block"]
            imported_block = cls.block_converter_class.import_config(config, block_config)

            return {
                "type": "fixed",
                "num_blocks": decoder_config["num_blocks"],
                "block": imported_block,
            }

        elif decoder_type == "pattern":
            blocks = {}
            for name, block_config in decoder_config["blocks"].items():
                blocks[name] = cls.block_converter_class.import_config(config, block_config)

            return {
                "type": "pattern",
                "blocks": blocks,
                "pattern": decoder_config["pattern"],
                "num_blocks": decoder_config["num_blocks"],
            }

        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

    @classmethod
    def export_config(cls, config) -> dict:
        from fast_llm.layers.block.config import FixedBlockSequenceConfig, PatternBlockSequenceConfig

        if isinstance(config, FixedBlockSequenceConfig):
            block_config = cls.block_converter_class.export_config(config.block)
            return {
                "decoder": {
                    "type": "fixed",
                    "num_blocks": config.num_blocks,
                    "block": block_config,
                }
            }

        elif isinstance(config, PatternBlockSequenceConfig):
            blocks = {}
            for name, block_config in config.blocks.items():
                blocks[name] = cls.block_converter_class.export_config(block_config)

            return {
                "decoder": {
                    "type": "pattern",
                    "blocks": blocks,
                    "pattern": config.pattern,
                    "num_blocks": config.num_blocks,
                }
            }

        else:
            raise ValueError(f"Unknown decoder config type: {type(config)}")

    @classmethod
    def get_converters(
        cls,
        config,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        from fast_llm.layers.block.config import FixedBlockSequenceConfig, PatternBlockSequenceConfig

        converters = []
        if type(config) is FixedBlockSequenceConfig:
            for block_index in range(config.num_blocks):
                converters += cls.block_converter_class.get_converters(
                    config.block,
                    f"{fast_llm_prefix}.{block_index}",
                    f"{hf_prefix}.{block_index}",
                    drop_on_export,
                )
        elif type(config) is PatternBlockSequenceConfig:
            for block_index in range(config.num_blocks):
                block_config = config.blocks[config.pattern[block_index % len(config.pattern)]]
                converters += cls.block_converter_class.get_converters(
                    block_config,
                    f"{fast_llm_prefix}.{block_index}",
                    f"{hf_prefix}.{block_index}",
                    drop_on_export,
                )
        else:
            raise NotImplementedError(f"Unsupported config type: {type(config).__name__}")
        return converters


class Apriel2HeadConverter:
    normalization_converter_class: typing.ClassVar[type[LlamaNormalizationConverter]] = LlamaNormalizationConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        norm_config = config["head"]["normalization"]
        return {"normalization": {"type": "rms_norm", "epsilon": norm_config["epsilon"]}}

    @classmethod
    def export_config(cls, config) -> dict:
        from fast_llm.layers.language_model.config import LanguageModelHeadConfig

        Assert.custom(isinstance, config, LanguageModelHeadConfig)
        return {
            "head": {
                "normalization": {
                    "type": "rms_norm",
                    "epsilon": config.normalization.epsilon,
                }
            }
        }

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


class Apriel2BaseModelConverter:
    decoder_converter_class: typing.ClassVar[type[Apriel2DecoderConverter]] = Apriel2DecoderConverter
    embeddings_converter_class: typing.ClassVar[type[LlamaEmbeddingsConverter]] = LlamaEmbeddingsConverter
    head_converter_class: typing.ClassVar[type[Apriel2HeadConverter]] = Apriel2HeadConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
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
            },
        )

    @classmethod
    def get_converters(cls, config: GPTBaseModelConfig, exported_config: dict) -> list[WeightConverter]:
        return [
            *cls.embeddings_converter_class.get_converters(config.embeddings, "embeddings", "model"),
            *cls.decoder_converter_class.get_converters(config.decoder, "decoder", "model.decoder.blocks"),
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
        exported = safe_merge_dicts(
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
        return exported

    @classmethod
    def _import_config(cls, config: dict[str, typing.Any]) -> dict[str, typing.Any]:
        return {"base_model": cls.base_model_converter_class.import_config(config)}

    @classmethod
    def _get_weight_converters(cls, config: GPTModelConfig, export_config: dict) -> list[WeightConverter]:
        return cls.base_model_converter_class.get_converters(config.base_model, export_config)
