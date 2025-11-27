"""
Apriel2 checkpoint format converter.

Apriel2 is a HuggingFace format that closely mirrors Fast-LLM's config structure,
making conversion straightforward.
"""

import typing

from transformers import PretrainedConfig

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import WeightConverter
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig, StochasticMixerConfig
from fast_llm.layers.ssm.config import Mamba2Config
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.models.gpt.conversion.config import Apriel2TextCheckpointFormat
from fast_llm.models.gpt.conversion.llama import get_parameter_converter, get_weight_and_bias_converters
from fast_llm.models.gpt.conversion.mistral import (
    MistralBaseModelConverter,
    MistralBlockConverter,
    MistralDecoderConverter,
    MistralHeadConverter,
    MistralHuggingfaceCheckpointHandler,
)
from fast_llm.utils import Assert, safe_merge_dicts


class Apriel2AttentionConverter:
    """Converter for attention mixers."""

    @classmethod
    def import_config(cls, config: dict) -> dict:
        """Import attention config from Apriel2 format."""
        return {
            "type": "attention",
            "heads": config.get("heads", 32),
            "head_groups": config.get("head_groups", config.get("heads", 32)),
            "head_size": config.get("head_size", None),
            "rotary": config.get("rotary", {"type": "default", "theta": 10000.0}),
            "add_linear_biases": config.get("add_linear_biases", False),
            "window_size": config.get("window_size", None),
        }

    @classmethod
    def export_config(cls, config: AttentionConfig) -> dict:
        """Export attention config to Apriel2 format."""
        from fast_llm.layers.attention.rotary.config import DefaultRotaryConfig, Llama3RotaryConfig, YarnRotaryConfig

        # Determine rotary type string
        if type(config.rotary) is DefaultRotaryConfig:
            rotary_type = "default"
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
        """Get weight converters for attention."""
        from fast_llm.models.gpt.conversion.llama import QueryWeightConverter, KeyValueWeightConverter

        # Use same weight names as Llama converter
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
    """Converter for Mamba mixers."""

    @classmethod
    def import_config(cls, config: dict) -> dict:
        """Import Mamba config from Apriel2 format."""
        return {
            "type": "mamba_2",
            "state_size": config.get("state_size", 16),
            "d_inner": config.get("d_inner"),
            "d_xb": config.get("d_xb", None),
            "dt_rank": config.get("dt_rank", "auto"),
            "add_linear_biases": config.get("add_linear_biases", False),
        }

    @classmethod
    def export_config(cls, config: Mamba2Config) -> dict:
        """Export Mamba config to Apriel2 format."""
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
        config: Mamba2Config,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        """Get weight converters for Mamba."""
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


# TODO: Add converters for GatedDeltaNet and KimiLinearAttention when implemented


class Apriel2StochasticMixerConverter:
    """Converter for stochastic mixers."""

    @classmethod
    def import_config(cls, config: dict) -> dict:
        """Import stochastic mixer config from Apriel2 format."""
        # Import each sub-mixer config
        mixers = {}
        for name, sub_mixer_config in config.get("mixers", {}).items():
            mixer_type = sub_mixer_config.get("type")
            if mixer_type == "attention":
                mixers[name] = Apriel2AttentionConverter.import_config(sub_mixer_config)
            elif mixer_type == "mamba":
                mixers[name] = Apriel2MambaConverter.import_config(sub_mixer_config)
            else:
                raise ValueError(f"Unknown sub-mixer type: {mixer_type}")

        return {
            "type": "stochastic",
            "mixers": mixers,
            "main_mixer_name": config.get("main_mixer_name"),
            "sampling_strategy": config.get("sampling_strategy", "uniform"),
        }

    @classmethod
    def export_config(cls, config: StochasticMixerConfig) -> dict:
        """Export stochastic mixer config to Apriel2 format."""
        # Export each sub-mixer config
        mixers = {}
        for name, sub_mixer in config.mixers.items():
            mixer_type = type(sub_mixer)
            if mixer_type is AttentionConfig:
                mixers[name] = Apriel2AttentionConverter.export_config(sub_mixer)
            elif mixer_type is Mamba2Config:
                mixers[name] = Apriel2MambaConverter.export_config(sub_mixer)
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
        """Get weight converters for stochastic mixer."""
        converters = []

        # Create converters for each sub-mixer
        for name, sub_mixer in config.mixers.items():
            mixer_type = type(sub_mixer)

            if mixer_type is AttentionConfig:
                converter_class = Apriel2AttentionConverter
                # Attention sub-mixers have .self_attn nested inside
                hf_sub_mixer_prefix = f"{hf_prefix}.mixers.{name}.self_attn"
            elif mixer_type is Mamba2Config:
                converter_class = Apriel2MambaConverter
                hf_sub_mixer_prefix = f"{hf_prefix}.mixers.{name}"
            else:
                raise ValueError(f"Unknown sub-mixer type: {mixer_type}")

            # Sub-mixers are stored in a ModuleDict with names as keys
            converters.extend(
                converter_class.get_converters(
                    sub_mixer,
                    f"{fast_llm_prefix}.mixers.{name}",
                    hf_sub_mixer_prefix,
                    drop_on_export=drop_on_export,
                )
            )

        return converters


class Apriel2BlockConverter(MistralBlockConverter):
    """Converter for decoder blocks."""

    @classmethod
    def import_config(cls, config: dict, block_config: dict) -> dict:
        """Import block config from Apriel2 format."""
        # Import mixer config
        mixer_config = block_config.get("mixer", {})
        mixer_type = mixer_config.get("type", "attention")

        if mixer_type == "attention":
            mixer = Apriel2AttentionConverter.import_config(mixer_config)
        elif mixer_type == "mamba":
            mixer = Apriel2MambaConverter.import_config(mixer_config)
        elif mixer_type == "stochastic":
            mixer = Apriel2StochasticMixerConverter.import_config(mixer_config)
        else:
            raise ValueError(f"Unknown mixer type: {mixer_type}")

        from fast_llm.functional.config import ActivationType

        mlp_config = block_config.get("mlp", {"type": "mlp"})
        mlp = {
            "type": "mlp",
            "intermediate_size": mlp_config.get("intermediate_size"),
            "activation": ActivationType.from_hf_name(mlp_config.get("activation", "silu")),
            "gated": True,
            "add_linear_biases": mlp_config.get("add_linear_biases", False),
        }

        normalization = block_config.get("normalization", {"type": "rms_norm"})

        return {
            "mixer": mixer,
            "mlp": mlp,
            "normalization": normalization,
        }

    @classmethod
    def export_config(cls, config: DecoderBlockConfig) -> dict:
        """Export block config to Apriel2 format."""
        from fast_llm.layers.common.normalization.config import (
            RMSNormalizationConfig,
            LayerNormalizationConfig,
            NoNormalizationConfig,
        )

        # Export mixer config
        mixer_type = type(config.mixer)

        if mixer_type is AttentionConfig:
            mixer = Apriel2AttentionConverter.export_config(config.mixer)
        elif mixer_type is Mamba2Config:
            mixer = Apriel2MambaConverter.export_config(config.mixer)
        elif mixer_type is StochasticMixerConfig:
            mixer = Apriel2StochasticMixerConverter.export_config(config.mixer)
        else:
            raise ValueError(f"Unknown mixer type: {mixer_type}")

        # Determine normalization type string
        norm_type = type(config.normalization)
        if norm_type is RMSNormalizationConfig:
            norm_type_str = "rms_norm"
        elif norm_type is LayerNormalizationConfig:
            norm_type_str = "layer_norm"
        elif norm_type is NoNormalizationConfig:
            norm_type_str = "none"
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

        # Export MLP
        from fast_llm.layers.decoder.mlp.config import MLPConfig

        if not isinstance(config.mlp, MLPConfig):
            raise ValueError(f"Unsupported MLP type: {type(config.mlp)}")

        mlp = {
            "type": "mlp",
            "intermediate_size": config.mlp.intermediate_size,
            "activation": config.mlp.activation.value,
        }

        # Export normalization
        normalization = {"type": norm_type_str}

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
        """Get weight converters for block."""
        converters = []

        # Mixer converters - all at .mixer with appropriate sub-paths
        mixer_type = type(config.mixer)
        if mixer_type is AttentionConfig:
            converter_class = Apriel2AttentionConverter
            hf_mixer_prefix = f"{hf_prefix}.mixer.self_attn"
        elif mixer_type is Mamba2Config:
            converter_class = Apriel2MambaConverter
            hf_mixer_prefix = f"{hf_prefix}.mixer"
        elif mixer_type is StochasticMixerConfig:
            converter_class = Apriel2StochasticMixerConverter
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

        # MLP converters - Fast-LLM uses layer_1 and layer_2
        from fast_llm.models.gpt.conversion.llama import SplitWeightConverter, MLPLayer2Converter

        converters.extend([
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
        ])

        # Normalization converters - Fast-LLM uses norm_1 and norm_2
        from fast_llm.models.gpt.conversion.llama import LlamaNormalizationConverter

        converters.extend([
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
        ])

        return converters


class Apriel2DecoderConverter(MistralDecoderConverter):
    """Converter for decoder."""

    block_converter_class: typing.ClassVar[type[Apriel2BlockConverter]] = Apriel2BlockConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        """Import decoder config from Apriel2 format."""
        decoder_config = config.get("decoder", {})
        decoder_type = decoder_config.get("type", "fixed")

        if decoder_type == "fixed":
            # Fixed decoder: single block config
            block_config = decoder_config.get("block", {})
            imported_block = cls.block_converter_class.import_config(config, block_config)

            return {
                "type": "fixed",
                "num_blocks": decoder_config.get("num_blocks", config.get("num_hidden_layers", 32)),
                "block": imported_block,
            }

        elif decoder_type == "pattern":
            # Pattern decoder: multiple named blocks
            blocks = {}
            for name, block_config in decoder_config.get("blocks", {}).items():
                blocks[name] = cls.block_converter_class.import_config(config, block_config)

            return {
                "type": "pattern",
                "blocks": blocks,
                "pattern": decoder_config.get("pattern", []),
                "num_blocks": decoder_config.get("num_blocks", config.get("num_hidden_layers", 32)),
            }

        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

    @classmethod
    def export_config(cls, config) -> dict:
        """Export decoder config to Apriel2 format."""
        from fast_llm.layers.block.config import FixedBlockSequenceConfig, PatternBlockSequenceConfig

        if isinstance(config, FixedBlockSequenceConfig):
            # Fixed decoder
            block_config = cls.block_converter_class.export_config(config.block)
            return {
                "decoder": {
                    "type": "fixed",
                    "num_blocks": config.num_blocks,
                    "block": block_config,
                }
            }

        elif isinstance(config, PatternBlockSequenceConfig):
            # Pattern decoder
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
        """Get weight converters for decoder."""
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


class Apriel2HeadConverter(MistralHeadConverter):
    block_converter_class: typing.ClassVar[type[Apriel2BlockConverter]] = Apriel2BlockConverter


class Apriel2BaseModelConverter(MistralBaseModelConverter):
    decoder_converter_class: typing.ClassVar[type[Apriel2DecoderConverter]] = Apriel2DecoderConverter
    head_converter_class: typing.ClassVar[type[Apriel2HeadConverter]] = Apriel2HeadConverter


class Apriel2HuggingfaceCheckpointHandler(MistralHuggingfaceCheckpointHandler):
    """HuggingFace checkpoint handler for Apriel2 format."""

    format: typing.ClassVar[type[CheckpointFormat]] = Apriel2TextCheckpointFormat
    architecture: typing.ClassVar[str] = "Apriel2ForCausalLM"
    base_model_converter_class: typing.ClassVar[type[Apriel2BaseModelConverter]] = Apriel2BaseModelConverter

    @classmethod
    def get_transformers_configuration_class(cls) -> type[PretrainedConfig]:
        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2TextConfig

        return Apriel2TextConfig

    @classmethod
    def get_model_files(cls) -> tuple[str, str, str | None]:
        from fast_llm_external_models.apriel2 import (
            configuration_apriel2,
            modeling_apriel2,
        )

        return configuration_apriel2.__file__, modeling_apriel2.__file__, None

    @classmethod
    def _export_config(cls, config: GPTModelConfig) -> dict[str, typing.Any]:
        return safe_merge_dicts(
            super()._export_config(config),
            {
                "auto_map": {
                    "AutoConfig": "configuration_apriel2.Apriel2TextConfig",
                    "AutoModel": "modeling_apriel2.Apriel2TextModel",
                    "AutoModelForCausalLM": "modeling_apriel2.Apriel2ForCausalLM",
                },
            },
        )
