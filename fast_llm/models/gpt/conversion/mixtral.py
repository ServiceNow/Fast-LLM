import typing

import torch

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
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert, safe_merge_dicts


class MoEMLPLayer2Converter(WeightConverter):
    """
    Converter for MoE layer 2 (down projection) weights.

    HuggingFace format: Per-expert weights, each of shape [hidden_size, intermediate_size]
    Fast-LLM format: Weight of shape [num_experts * intermediate_size, hidden_size]

    Fast-LLM stores MoE layer 2 weights with input dimension (intermediate) flattened across experts.
    The output dimension (hidden) is NOT multiplied by experts - each expert outputs to the same hidden size.
    This matches the MoEAffineLinearConfig which extracts only the feature dimension for transposed weights.
    """

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        # Fast-LLM: [num_experts * intermediate_size, hidden_size]
        # HF needs: per-expert weights of [hidden_size, intermediate_size]
        (merged_weight,) = weight
        num_experts = len(self.export_name)
        hidden_size = merged_weight.shape[1]
        intermediate_size = merged_weight.shape[0] // num_experts

        # Reshape to [num_experts, intermediate_size, hidden_size]
        reshaped = merged_weight[:].reshape(num_experts, intermediate_size, hidden_size)

        # Transpose each expert to [hidden_size, intermediate_size] (HF format)
        return tuple(reshaped[i].t().contiguous() for i in range(num_experts))

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        # HF: per-expert weights, each [hidden_size, intermediate_size]
        # Need to create [num_experts * intermediate_size, hidden_size]
        num_experts = len(weight)

        # Materialize first weight to get dtype, device, and shape
        first_weight = weight[0][:]
        hidden_size, intermediate_size = first_weight.shape  # HF stores as [hidden, intermediate]

        # Transpose each expert's weights to [intermediate_size, hidden_size] and stack
        expert_weights = [weight[i][:].t() for i in range(num_experts)]

        # Concatenate along first dimension: [num_experts * intermediate_size, hidden_size]
        merged = torch.cat(expert_weights, dim=0)

        return (merged.contiguous(),)


class MixtralMLPConverter(LlamaMLPConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        return safe_merge_dicts(
            super().import_config(config),
            {
                "type": "moe",
                "experts": config["num_local_experts"],
                "experts_per_token": config["num_experts_per_tok"],
                # Use moe_affine_linear type for MoE expert layers to handle CompositeTensorDim correctly
                "layer_1": {
                    "type": "moe_affine_linear",
                },
                "layer_2": {
                    "type": "moe_affine_linear",
                },
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
                MoEMLPLayer2Converter,
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
