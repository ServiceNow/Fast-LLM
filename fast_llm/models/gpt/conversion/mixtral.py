import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import SplitWeightConverter, WeightConverter
from fast_llm.layers.decoder.mlp.config import MoEMLPConfig
from fast_llm.models.gpt.conversion.config import MixtralCheckpointFormat
from fast_llm.models.gpt.conversion.llama import LlamaMLPConverter, MLPLayer2Converter, get_weight_and_bias_converters
from fast_llm.models.gpt.conversion.mistral import (
    MistralBaseModelConverter,
    MistralBlockConverter,
    MistralDecoderConverter,
    MistralHeadConverter,
    MistralHuggingfaceCheckpointHandler,
)
from fast_llm.utils import Assert, safe_merge_dicts


class MixtralMLPConverter(LlamaMLPConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        return safe_merge_dicts(
            super().import_config(config),
            {
                "type": "moe",
                "experts": config["num_local_experts"],
                "experts_per_token": config["num_experts_per_tok"],
            },
        )

    @classmethod
    def export_config(cls, config: MoEMLPConfig) -> dict:
        Assert.custom(isinstance, config, MoEMLPConfig)
        assert not config.add_linear_biases
        return safe_merge_dicts(
            super().export_config(config),
            {
                "num_local_experts": config.experts,
                "num_experts_per_tok": config.experts_per_token,
            },
        )

    @classmethod
    def get_converters(
        cls,
        config: MoEMLPConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.router",
                f"{hf_prefix}.gate",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_1",
                tuple(f"{hf_prefix}.experts.{i}.{w}" for i in range(config.experts) for w in ("w1", "w3")),
                False,
                SplitWeightConverter,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_2",
                tuple(f"{hf_prefix}.experts.{i}.w2" for i in range(config.experts)),
                False,
                MLPLayer2Converter,
                drop_on_export=drop_on_export,
            ),
        ]


class MixtralBlockConverter(MistralBlockConverter):
    hf_mlp_name: typing.ClassVar[str] = "block_sparse_moe"
    mlp_converter_class: typing.ClassVar[type[MixtralMLPConverter]] = MixtralMLPConverter


class MixtralDecoderConverter(MistralDecoderConverter):
    block_converter_class: typing.ClassVar[type[MixtralBlockConverter]] = MixtralBlockConverter


class MixtralHeadConverter(MistralHeadConverter):
    block_converter_class: typing.ClassVar[type[MixtralBlockConverter]] = MixtralBlockConverter


class MixtralBaseModelConverter(MistralBaseModelConverter):
    decoder_converter_class: typing.ClassVar[type[MixtralDecoderConverter]] = MixtralDecoderConverter
    head_converter_class: typing.ClassVar[type[MixtralHeadConverter]] = MixtralHeadConverter


class MixtralHuggingfaceCheckpointHandler(MistralHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = MixtralCheckpointFormat
    architecture: typing.ClassVar[str] = "MixtralForCausalLM"
    base_model_converter_class: typing.ClassVar[type[MixtralBaseModelConverter]] = MixtralBaseModelConverter

    @classmethod
    def get_transformers_configuration_class(cls):
        import transformers

        return transformers.MixtralConfig
