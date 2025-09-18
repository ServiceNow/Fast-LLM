import math
import typing

from transformers import PretrainedConfig

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import WeightConverter
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.block.config import BlockSequenceConfig, FixedBlockSequenceConfig, PatternBlockSequenceConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.layers.ssm.config import DiscreteMamba2Config, Mamba2Config
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.models.gpt.conversion.config import AprielHybridSSMCheckpointFormat
from fast_llm.models.gpt.conversion.llama import get_parameter_converter, get_weight_and_bias_converters
from fast_llm.models.gpt.conversion.mistral import (
    MistralBaseModelConverter,
    MistralBlockConverter,
    MistralDecoderConverter,
    MistralHeadConverter,
    MistralHuggingfaceCheckpointHandler,
)
from fast_llm.utils import Assert, safe_merge_dicts


class AprielDiscreteMamba2Converter:
    @classmethod
    def import_config(cls, config: dict, hidden_size: int) -> dict:
        return {
            "type": "discrete_mamba_2",
            "state_size": config["ssm_cfg"]["d_state"],
            "d_inner": config["ssm_cfg"].get("d_inner") or hidden_size * config["ssm_cfg"].get("expand", 1),
            "add_linear_biases": config["ssm_cfg"]["bias"],
            "convolution_layer": {"bias": {"enabled": config["ssm_cfg"].get("conv_bias", True)}},
            "n_qk_heads": config["ssm_cfg"]["n_qk_heads"],
            "n_v_heads": config["ssm_cfg"]["n_v_heads"],
            "chunk_size": config["ssm_cfg"]["chunk_size"],
        }

    @classmethod
    def export_config(cls, config: DiscreteMamba2Config) -> dict:
        cls._check_config(config)
        return {
            "ssm_cfg": {
                "d_state": config.state_size,
                "d_inner": config.d_inner,
                "bias": config.add_linear_biases,
                "conv_bias": (
                    config.add_linear_biases
                    if config.convolution_layer.bias.enabled is None
                    else config.convolution_layer.bias.enabled
                ),
                "n_qk_heads": config.n_qk_heads,
                "n_v_heads": config.n_v_heads,
                "chunk_size": config.chunk_size,
            }
        }

    @classmethod
    def _check_config(cls, config: DiscreteMamba2Config) -> None:
        # Opportunity to make derived classes less constrained.
        Assert.is_(type(config), DiscreteMamba2Config)
        Assert.incl(config.z_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.x_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.b_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.c_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.output_layer.bias.enabled, (None, config.add_linear_biases))

    @classmethod
    def get_converters(
        cls,
        config: DiscreteMamba2Config,
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
                f"{fast_llm_prefix}.convolution",
                f"{hf_prefix}.conv1d",
                (
                    config.add_linear_biases
                    if config.convolution_layer.bias.enabled is None
                    else config.convolution_layer.bias.enabled
                ),
                drop_on_export=drop_on_export,
            ),
            *(
                []
                if config.add_linear_biases
                else [
                    get_parameter_converter(
                        f"{fast_llm_prefix}.z_bias",
                        f"{hf_prefix}.z_bias",
                        drop_on_export=drop_on_export,
                    )
                ]
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


class AprielMamba2Converter:
    @classmethod
    def import_config(cls, config: dict, hidden_size: int) -> dict:
        return {
            "type": "mamba_2",
            "state_size": config["ssm_cfg"]["d_state"],
            "d_inner": config["ssm_cfg"].get("d_inner") or hidden_size * config["ssm_cfg"].get("expand", 1),
            "add_linear_biases": config["ssm_cfg"]["bias"],
            "convolution_layer": {"bias": {"enabled": config["ssm_cfg"].get("conv_bias", True)}},
            "d_xb": config["ssm_cfg"].get("d_xb") or hidden_size,
            "dt_layer": {"bias": {"enabled": config["ssm_cfg"].get("dt_proj_bias", True)}},
            "dt_rank": (
                math.ceil(hidden_size)
                if config["ssm_cfg"].get("dt_rank", "auto") == "auto"
                else config["ssm_cfg"]["dt_rank"]
            ),
            "repeat_kv_before_conv": config["ssm_cfg"].get("repeat_kv_before_conv", True),
        }

    @classmethod
    def export_config(cls, config: Mamba2Config) -> dict:
        cls._check_config(config)
        return {
            "ssm_cfg": {
                "d_state": config.state_size,
                "d_inner": config.d_inner,
                "bias": config.add_linear_biases,
                "conv_bias": (
                    config.add_linear_biases
                    if config.convolution_layer.bias.enabled is None
                    else config.convolution_layer.bias.enabled
                ),
                "d_xb": config.d_xb,
                "dt_proj_bias": (
                    config.add_linear_biases if config.dt_layer.bias.enabled is None else config.dt_layer.bias.enabled
                ),
                "dt_rank": config.dt_rank,
                "repeat_kv_before_conv": config.repeat_kv_before_conv,
            }
        }

    @classmethod
    def _check_config(cls, config: Mamba2Config) -> None:
        # Opportunity to make derived classes less constrained.
        Assert.is_(type(config), Mamba2Config)
        Assert.incl(config.z_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.x_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.b_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.c_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.dt_input_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.output_layer.bias.enabled, (None, config.add_linear_biases))

    @classmethod
    def get_converters(
        cls,
        config: Mamba2Config,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return [
            # TODO: Conv
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
                config.add_linear_biases if config.dt_layer.bias.enabled is None else config.dt_layer.bias.enabled,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.convolution",
                f"{hf_prefix}.conv1d",
                (
                    config.add_linear_biases
                    if config.convolution_layer.bias.enabled is None
                    else config.convolution_layer.bias.enabled
                ),
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


class AprielDiscreteMamba2BlockConverter(MistralBlockConverter):
    mixer_converter_class: typing.ClassVar[type[AprielDiscreteMamba2Converter]] = AprielDiscreteMamba2Converter


class AprielMamba2BlockConverter(MistralBlockConverter):
    mixer_converter_class: typing.ClassVar[type[AprielMamba2Converter]] = AprielMamba2Converter


class AprielBlockConverter:
    layout_names = {
        AttentionConfig: "t",
        Mamba2Config: "m2",
        DiscreteMamba2Config: "m2d",
    }
    _converter_classes = {
        AttentionConfig: MistralBlockConverter,
        Mamba2Config: AprielMamba2BlockConverter,
        DiscreteMamba2Config: AprielDiscreteMamba2BlockConverter,
    }
    _config_classes = {value: key for key, value in layout_names.items()}

    @classmethod
    def import_config(cls, config: dict, hidden_size: int, layout_name: str = "t") -> dict:
        return cls._converter_classes[cls._config_classes[layout_name]].import_config(config, hidden_size)

    @classmethod
    def export_config(cls, config) -> dict:
        return cls._converter_classes[type(config.mixer)].export_config(config)

    @classmethod
    def get_converters(
        cls,
        config: DecoderBlockConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return cls._converter_classes[type(config.mixer)].get_converters(
            config, fast_llm_prefix, hf_prefix, drop_on_export=drop_on_export
        )


class AprielDecoderConverter(MistralDecoderConverter):
    block_converter_class: typing.ClassVar[type[AprielBlockConverter]] = AprielBlockConverter

    @classmethod
    def import_config(cls, config: dict, hidden_size: int) -> dict:
        layout = config["hybrid_block_layout"]
        if len(layout) == 1:
            return {
                "block": cls.block_converter_class.import_config(config, hidden_size, layout[0]),
                "num_blocks": config["num_hidden_layers"],
            }
        else:
            return {
                "type": "pattern",
                "blocks": {
                    layout_name: cls.block_converter_class.import_config(config, hidden_size, layout_name)
                    for layout_name in set(layout)
                },
                "pattern": layout,
                "num_blocks": config["num_hidden_layers"],
            }

    @classmethod
    def export_config(cls, config: BlockSequenceConfig) -> dict:
        if type(config) is FixedBlockSequenceConfig:
            block_configs = [config.block]
            pattern_block_configs = [config.block]
        elif type(config) is PatternBlockSequenceConfig:
            block_configs = config.blocks.values()
            pattern_block_configs = [config.blocks[block_name] for block_name in config.pattern]
        else:
            raise NotImplementedError()
        # There may be all sorts of blocks, but `safe_merge_dicts` ensures they are compatible.
        return safe_merge_dicts(
            *[cls.block_converter_class.export_config(block_config) for block_config in block_configs],
            {
                "num_hidden_layers": config.num_blocks,
                "hybrid_block_layout": [
                    cls.block_converter_class.layout_names[type(block_config.mixer)]
                    for block_config in pattern_block_configs
                ],
            },
        )

    @classmethod
    def get_converters(
        cls,
        config: PatternBlockSequenceConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
        fast_llm_layer_start: int = 1,
    ) -> list[WeightConverter]:
        converters = []
        for block_index in range(config.num_blocks):
            block_config = config.blocks[config.pattern[block_index % len(config.pattern)]]
            converters += cls.block_converter_class.get_converters(
                block_config,
                f"{fast_llm_prefix}.{block_index+fast_llm_layer_start}",
                f"{hf_prefix}.{block_index}",
                drop_on_export,
            )
        return converters


class AprielHeadConverter(MistralHeadConverter):
    block_converter_class: typing.ClassVar[type[AprielBlockConverter]] = AprielBlockConverter


class AprielBaseModelConverter(MistralBaseModelConverter):
    decoder_converter_class: typing.ClassVar[type[AprielDecoderConverter]] = AprielDecoderConverter
    head_converter_class: typing.ClassVar[type[AprielHeadConverter]] = AprielHeadConverter


class AprielHuggingfaceCheckpointHandler(MistralHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = AprielHybridSSMCheckpointFormat
    architecture: typing.ClassVar[str] = "AprielHybridSSMForCausalLM"
    base_model_converter_class: typing.ClassVar[type[AprielBaseModelConverter]] = AprielBaseModelConverter

    @classmethod
    def get_transformers_configuration_class(cls) -> type[PretrainedConfig]:
        from fast_llm_external_models.apriel_hybrid_ssm.configuration_apriel_hybrid_ssm import AprielHybridSSMConfig

        return AprielHybridSSMConfig

    @classmethod
    def get_model_files(cls) -> tuple[str, str, str | None]:
        from fast_llm_external_models.apriel_hybrid_ssm import (
            configuration_apriel_hybrid_ssm,
            modeling_apriel_hybrid_ssm,
        )

        return configuration_apriel_hybrid_ssm.__file__, modeling_apriel_hybrid_ssm.__file__, None

    @classmethod
    def _export_config(cls, config: GPTModelConfig) -> dict[str, typing.Any]:
        return safe_merge_dicts(
            super()._export_config(config),
            {
                "auto_map": {
                    "AutoConfig": "configuration_apriel_hybrid_ssm.AprielHybridSSMConfig",
                    "AutoModel": "modeling_apriel_hybrid_ssm.AprielHybridSSMModel",
                    "AutoModelForCausalLM": "modeling_apriel_hybrid_ssm.AprielHybridSSMForCausalLM",
                },
            },
        )
