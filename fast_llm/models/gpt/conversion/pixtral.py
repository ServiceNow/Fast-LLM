import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import SplitWeightConverter, WeightConverter
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.rotary.config import Rotary2DConfig
from fast_llm.layers.common.normalization.config import LayerNormalizationConfig
from fast_llm.models.gpt.conversion.llama import KeyValueWeightConverter, MLPLayer2Converter, QueryWeightConverter
from fast_llm.utils import Assert


class PixtralNumHeadsConverter(ParamConverter):
    """
    Pixtral encoder uses Multi-Head Attention.
    Map `num_attention_heads` and `head_groups` to a single `num_heads` parameter.
    """

    def __post_init__(self):
        Assert.eq(len(self.fast_llm_names), 2)
        Assert.eq(len(self.export_names), 1)

    def export_params(self, fast_llm_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        (num_heads, head_groups) = fast_llm_values
        assert head_groups == num_heads, "Pixtral encoder expects num_heads == head_groups (MHA)"
        return (num_heads,)

    def import_params(self, export_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        (num_heads,) = export_values
        return (num_heads, num_heads)


class PixtralRotaryParamConverter(ParamConverter):
    """
    Pixtral encoder uses 2D Rotary Embeddings.
    Map `rope_theta` to a single `rotary` parameter. `rotary_scaling` is not needed.
    """

    def __init__(self, fast_llm_names, export_names):
        Assert.eq(len(fast_llm_names), 1)
        Assert.eq(len(export_names), 1)
        self.fast_llm_names = fast_llm_names
        self.export_names = export_names

    def export_params(self, fast_llm_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        (rotary_config,) = fast_llm_values
        if type(rotary_config) is Rotary2DConfig:
            return (rotary_config.theta,)
        else:
            raise ValueError(f"Unsupported rotary type: {type(rotary_config).__name__}")

    def import_params(self, export_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        (rotary_theta,) = export_values
        rotary_config = {
            "type": "rope_2d",
            "theta": rotary_theta,
        }
        return (rotary_config,)


class PixtralHuggingfaceCheckpointHandler(WeightAndBiasConverterMixin, HuggingfaceStateDictCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = PixtralGPTHuggingfaceCheckpointFormat
    _model_class: typing.ClassVar[FastLLMModelConfig] = FastLLMModelConfig

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantImportParamConverter(fast_llm_names=(("type",),), fast_llm_value="pixtral"),
            ConstantImportParamConverter(fast_llm_names=(("patch_norm", "type"),), fast_llm_value="rms_norm"),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "normalization", "type"),), fast_llm_value="rms_norm"
            ),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "type"),), fast_llm_value="image_encoder"),
            ConstantExportParamConverter(export_names=(("architectures",),), export_value=["PixtralVisionModel"]),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "causal"),), fast_llm_value=False),
            RenameParamConverter(
                fast_llm_names=(
                    (
                        "transformer",
                        "num_layers",
                    ),
                ),
                export_names=(("num_hidden_layers",),),
            ),
            RenameParamConverter(
                fast_llm_names=(
                    (
                        "transformer",
                        "hidden_size",
                    ),
                ),
                export_names=(("hidden_size",),),
            ),
            PixtralNumHeadsConverter(
                fast_llm_names=(
                    (
                        "transformer",
                        "num_attention_heads",
                    ),
                    (
                        "transformer",
                        "head_groups",
                    ),
                ),
                export_names=(("num_attention_heads",),),
            ),
            RenameParamConverter(
                fast_llm_names=(
                    (
                        "transformer",
                        "ffn_hidden_size",
                    ),
                ),
                export_names=(("intermediate_size",),),
            ),
            MappedConfigParamConverter(
                fast_llm_names=(("transformer", "activation_type"),),
                export_names=(("hidden_act",),),
                fast_llm_value=ActivationType.from_hf_name,
                export_value=lambda activation_type: activation_type.hf_name,
            ),
            RenameParamConverter(
                fast_llm_names=(
                    (
                        "transformer",
                        "kv_channels",
                    ),
                ),
                export_names=(("head_dim",),),
            ),
            # ConstantImportParamConverter(
            #     fast_llm_names=(("transformer", "rotary", "type"),), fast_llm_value=RotaryEmbeddingType.rope_2d
            # ),
            # RenameParamConverter(
            #     fast_llm_names=(
            #         (
            #             "transformer",
            #             "rotary",
            #             "theta",
            #         ),
            #     ),
            #     export_names=(("rope_theta",),),
            # ),
            PixtralRotaryParamConverter(
                fast_llm_names=(("transformer", "rotary"),),
                export_names=(("rope_theta",),),
            ),
            RenameParamConverter(fast_llm_names=(("patch_size",),), export_names=(("patch_size",),)),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "gated"),), fast_llm_value=True),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "add_linear_biases"),), fast_llm_value=False),
        ]

    def _get_transformer_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        return [
            SplitWeightConverter(
                f"{fast_llm_prefix}.mlp.layer_1.weight",
                (f"{hf_prefix}.feed_forward.gate_proj.weight", f"{hf_prefix}.feed_forward.up_proj.weight"),
            ),
            MLPLayer2Converter(
                f"{fast_llm_prefix}.mlp.layer_2.weight",
                f"{hf_prefix}.feed_forward.down_proj.weight",
                self._model.config.base_model,
            ),
        ]

    def _create_vision_transformer_layer_converters(
        self, transformer_layer_index: int, fast_llm_offset: int = 1, hf_base_prefix: str = ""
    ) -> list[WeightConverter]:
        # Vision transformer layer
        transformer_config = self._model.config.base_model.vision_encoder.transformer
        norm_bias: bool = isinstance(self._model.config.base_model.transformer.normalization, LayerNormalizationConfig)
        name_bias_cls = [
            # Self-attn
            (
                f"layers.{fast_llm_offset + transformer_layer_index}.self_attn.query",
                f"{hf_base_prefix}transformer.layers.{transformer_layer_index}.attention.q_proj",
                transformer_config.add_attn_qkv_bias,
                QueryWeightConverter,
            ),
            (
                f"layers.{fast_llm_offset + transformer_layer_index}.self_attn.key_value",
                (
                    f"{hf_base_prefix}transformer.layers.{transformer_layer_index}.attention.k_proj",
                    f"{hf_base_prefix}transformer.layers.{transformer_layer_index}.attention.v_proj",
                ),
                transformer_config.add_attn_qkv_bias,
                KeyValueWeightConverter,
            ),
            (
                f"layers.{fast_llm_offset + transformer_layer_index}.self_attn.dense",
                f"{hf_base_prefix}transformer.layers.{transformer_layer_index}.attention.o_proj",
                transformer_config.add_attn_dense_bias,
                WeightConverter,
            ),
            # Norm
            (
                f"layers.{fast_llm_offset + transformer_layer_index}.norm_1",
                f"{hf_base_prefix}transformer.layers.{transformer_layer_index}.attention_norm",
                norm_bias,
                WeightConverter,
            ),
            (
                f"layers.{fast_llm_offset + transformer_layer_index}.norm_2",
                f"{hf_base_prefix}transformer.layers.{transformer_layer_index}.ffn_norm",
                norm_bias,
                WeightConverter,
            ),
        ]
        converters = []
        for fast_llm_prefix, hf_prefix, use_bias, cls in name_bias_cls:
            converters += self._get_weight_and_bias_converters(
                fast_llm_prefix,
                hf_prefix,
                use_bias,
                cls,
            )
        # MLP
        converters += self._get_transformer_mlp_converters(
            f"layers.{fast_llm_offset + transformer_layer_index}",
            f"{hf_base_prefix}transformer.layers.{transformer_layer_index}",
        )
        return converters

    def _create_weight_converters(self, offset: int = 0, hf_base_prefix: str = "") -> list[WeightConverter]:
        converters = []
        norm_bias = isinstance(self._model.config.base_model.vision_encoder.patch_norm, LayerNormalizationConfig)
        converters.append(WeightConverter(f"layers.{offset}.weight", f"{hf_base_prefix}patch_conv.weight"))
        if self._model.config.base_model.vision_encoder.conv_bias:
            converters.append(WeightConverter(f"layers.{offset}.bias", f"{hf_base_prefix}patch_conv.bias"))
        converters.append(WeightConverter(f"layers.{offset}.norm.weight", f"{hf_base_prefix}ln_pre.weight"))
        if norm_bias:
            converters.append(WeightConverter(f"layers.{offset}.norm.bias", f"{hf_base_prefix}ln_pre.bias"))

        num_layers = self._model.config.base_model.vision_encoder.transformer.num_layers
        for i in range(num_layers):
            converters += self._create_vision_transformer_layer_converters(i, offset + 1, hf_base_prefix)

        converters.extend(
            [
                WeightConverter(
                    f"layers.{offset + num_layers + 1}.layer_1.weight", "multi_modal_projector.linear_1.weight"
                ),
                WeightConverter(
                    f"layers.{offset + num_layers + 1}.layer_2.weight", "multi_modal_projector.linear_2.weight"
                ),
            ]
        )
        if self._model.config.base_model.vision_encoder.adapter_bias:
            converters.extend(
                [
                    WeightConverter(
                        f"layers.{offset + num_layers + 1}.layer_1.bias", "multi_modal_projector.linear_1.bias"
                    ),
                    WeightConverter(
                        f"layers.{offset + num_layers + 1}.layer_2.bias", "multi_modal_projector.linear_2.bias"
                    ),
                ]
            )

        return converters

    @property
    def num_layers(self) -> int:
        # +2 for projector and conv layers
        return self._model.config.base_model.vision_encoder.transformer.num_layers + 2
