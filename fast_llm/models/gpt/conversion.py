import abc
import dataclasses
import logging
import typing

import torch

from fast_llm.config import DEFAULT, MISSING
from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    AutoStateDictCheckpointHandler,
    ConstantExportParamConverter,
    ConstantImportParamConverter,
    IgnoreExportWeightConverter,
    IgnoreImportParamConverter,
    IgnoreImportWeightConverter,
    MappedConfigParamConverter,
    ParamConverter,
    RenameParamConverter,
    SplitWeightConverter,
    WeightConverter,
)
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.functional.config import ActivationType
from fast_llm.functional.rotary import convert_rotary_complex_to_real, convert_rotary_real_to_complex
from fast_llm.layers.common.config import NormalizationType
from fast_llm.layers.transformer.config import RotaryEmbeddingType, RoutingType, TransformerConfig
from fast_llm.models.gpt.config import (
    GPTArchitectureConfig,
    GPTModelConfig,
    LlamaGPTHuggingfaceCheckpointFormat,
    MistralGPTHuggingfaceCheckpointFormat,
    MixtralGPTHuggingfaceCheckpointFormat,
    Qwen2GPTHuggingfaceCheckpointFormat,
    Starcoder2GPTHuggingfaceCheckpointFormat,
)
from fast_llm.models.gpt.model import GPTModel
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class QueryWeightConverter(WeightConverter):
    # Hf uses the real format for rotary embeddings.
    _config: GPTArchitectureConfig

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (query,) = weight
        if self._config.transformer.rotary.complex_format:
            query = convert_rotary_complex_to_real(query[:], self._config.transformer.kv_channels, 0)
        return (query,)

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (query,) = weight
        if self._config.transformer.rotary.complex_format:
            query = convert_rotary_real_to_complex(query[:], self._config.transformer.kv_channels, 0)
        return (query,)


class KeyValueWeightConverter(WeightConverter):
    # Hf uses the real format for rotary embeddings, and keeps the key and value separate.
    _config: GPTArchitectureConfig

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (key_value,) = weight
        key, value = key_value[:].chunk(2)
        if self._config.transformer.rotary.complex_format:
            key = convert_rotary_complex_to_real(key, self._config.transformer.kv_channels, 0)
        return key, value

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        key, value = weight
        if self._config.transformer.rotary.complex_format:
            key = convert_rotary_real_to_complex(key[:], self._config.transformer.kv_channels, 0)
        key_value = torch.cat([key[:], value[:]])
        return (key_value,)


class MLPLayer2Converter(WeightConverter):
    # Similar to SplitWeightConverter, but handles the optional MLP transpose.
    # Still ok for non-gated (trivial split) and biases (trivial 1d transpose)
    _config: GPTArchitectureConfig

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (merged_weight,) = weight
        return tuple(t.contiguous() for t in merged_weight[:].t().chunk(len(self.export_name), dim=-1))

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        merged_weight = torch.cat([weight_[:] for weight_ in weight], dim=-1)
        return (merged_weight.t().contiguous(),)


class CommonHuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    _model: GPTModel
    _model_class: typing.ClassVar[FastLLMModelConfig] = GPTModelConfig
    """
    Common converter for llama-based huggingface models (llama, starcoder2, mistral, mixtral)
    """

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantImportParamConverter(fast_llm_names=(("use_position_embeddings",),), fast_llm_value=False),
            RenameParamConverter(
                fast_llm_names=(("transformer", "rotary", "theta"),), export_names=(("rope_theta",),)
            ),
            MappedConfigParamConverter(
                fast_llm_names=(("transformer", "activation_type"),),
                export_names=(("hidden_act",),),
                fast_llm_value=ActivationType.from_hf_name,
                export_value=lambda activation_type: activation_type.hf_name,
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "num_layers"),),
                export_names=(("num_hidden_layers",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "hidden_size"),),
                export_names=(("hidden_size",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "num_attention_heads"),),
                export_names=(("num_attention_heads",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "head_groups"),),
                export_names=(("num_key_value_heads",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "ffn_hidden_size"),),
                export_names=(("intermediate_size",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("vocab_size",),),
                export_names=(("vocab_size",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("tie_word_embeddings",),),
                export_names=(("tie_word_embeddings",),),
            ),
        ]

    @abc.abstractmethod
    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        pass

    def _create_weight_converters(
        self,
    ) -> list[WeightConverter]:
        converters = []
        num_layers = self._model.config.base_model.transformer.num_layers

        # Embeddings
        converters.append(WeightConverter("layers.0.word_embeddings_weight", "model.embed_tokens.weight"))

        converters += self._create_lm_head_converters()

        for i in range(num_layers):
            converters += self._create_transformer_layer_converters(i)

        return converters

    def _create_transformer_layer_converters(self, i: int, ignore_export: bool = False) -> list[WeightConverter]:
        transformer_config: TransformerConfig = self._model.config.base_model.transformer
        norm_bias: bool = self._model.config.base_model.transformer.normalization.type == NormalizationType.layer_norm
        converters = []
        names_bias_cls = [
            # Self-attn
            (
                f"layers.{i+1}.self_attn.query",
                f"model.layers.{i}.self_attn.q_proj",
                transformer_config.add_attn_qkv_bias,
                QueryWeightConverter,
            ),
            (
                f"layers.{i+1}.self_attn.key_value",
                (f"model.layers.{i}.self_attn.k_proj", f"model.layers.{i}.self_attn.v_proj"),
                transformer_config.add_attn_qkv_bias,
                KeyValueWeightConverter,
            ),
            (
                f"layers.{i+1}.self_attn.dense",
                f"model.layers.{i}.self_attn.o_proj",
                transformer_config.add_attn_dense_bias,
                WeightConverter,
            ),
            # Norm
            (
                f"layers.{i+1}.norm_1",
                f"model.layers.{i}.input_layernorm",
                norm_bias,
                WeightConverter,
            ),
            (
                f"layers.{i+1}.norm_2",
                f"model.layers.{i}.post_attention_layernorm",
                norm_bias,
                WeightConverter,
            ),
        ]
        for fast_llm_prefix, hf_prefix, use_bias, cls in names_bias_cls:
            converters += self._get_weight_and_bias_converters(
                fast_llm_prefix,
                () if ignore_export else hf_prefix,
                use_bias,
                cls=IgnoreExportWeightConverter if ignore_export else cls,
            )

        # MLP
        if ignore_export:
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.mlp.layer_1", (), transformer_config.add_mlp_bias, cls=IgnoreExportWeightConverter
            )
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.mlp.layer_2", (), transformer_config.add_mlp_bias, cls=IgnoreExportWeightConverter
            )
            converters += [IgnoreExportWeightConverter(f"layers.{i+1}.mlp.router.weight", ())]
        else:
            converters += self._get_mlp_converters(f"layers.{i+1}", f"model.layers.{i}")
        return converters

    def _create_lm_head_converters(self) -> list[WeightConverter]:
        num_layers = self._model.config.base_model.transformer.num_layers
        prediction_heads = self._model.config.base_model.prediction_heads
        norm_bias: bool = self._model.config.base_model.transformer.normalization.type == NormalizationType.layer_norm
        converters = []

        # Next-token prediction head
        # Final norm
        converters += self._get_weight_and_bias_converters(
            f"layers.{num_layers + 1}.final_norm", "model.norm", norm_bias
        )
        # Output weights
        if self._model.config.base_model.tie_word_embeddings:
            converters.append(IgnoreImportWeightConverter((), "lm_head.weight"))
        else:
            converters.append(WeightConverter(f"layers.{num_layers + 1}.output_weights", "lm_head.weight"))

        # MTP-heads > 0 are thrown away
        for i in range(1, prediction_heads):
            logger.warning(
                f"The model weights for the multi-token prediction head {i} are discarded during conversion."
            )
            mtp_transformer_layer_index = num_layers - 1 + 2 * i
            # MTP transformer layer
            converters += self._create_transformer_layer_converters(mtp_transformer_layer_index, ignore_export=True)
            # MTP output norm
            converters += self._get_weight_and_bias_converters(
                f"layers.{mtp_transformer_layer_index + 2}.final_norm", (), norm_bias, IgnoreExportWeightConverter
            )

        return converters

    def _get_weight_and_bias_converters(
        self,
        fast_llm_prefix: str | tuple[str, ...],
        hf_prefix: str | tuple[str, ...],
        use_bias: bool,
        cls=WeightConverter,
    ) -> list[WeightConverter]:
        if isinstance(fast_llm_prefix, str):
            fast_llm_prefix = (fast_llm_prefix,)
        if isinstance(hf_prefix, str):
            hf_prefix = (hf_prefix,)
        converters = [
            cls(
                tuple(f"{prefix}.weight" for prefix in fast_llm_prefix),
                tuple(f"{prefix}.weight" for prefix in hf_prefix),
                self._model.config.base_model,
            )
        ]
        if use_bias:
            converters.append(
                cls(
                    tuple(f"{prefix}.bias" for prefix in fast_llm_prefix),
                    tuple(f"{prefix}.bias" for prefix in hf_prefix),
                    self._model.config.base_model,
                )
            )
        return converters


class Starcoder2HuggingfaceCheckpointHandler(CommonHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = Starcoder2GPTHuggingfaceCheckpointFormat

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(export_names=(("architectures",),), export_value=["Starcoder2ForCausalLM"]),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "rotary", "type"),), fast_llm_value=RotaryEmbeddingType.default
            ),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "normalization", "type"),), fast_llm_value=NormalizationType.layer_norm
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "normalization", "epsilon"),), export_names=(("norm_epsilon",),)
            ),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "gated"),), fast_llm_value=False),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "add_linear_biases"),), fast_llm_value=True),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        transformer_config: TransformerConfig = self._model.config.base_model.transformer
        return [
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_1", f"{hf_prefix}.mlp.c_fc", transformer_config.add_mlp_bias
            ),
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_2",
                f"{hf_prefix}.mlp.c_proj",
                transformer_config.add_mlp_bias,
                MLPLayer2Converter,
            ),
        ]


class CommonLlamaHuggingfaceCheckpointHandler(CommonHuggingfaceCheckpointHandler, abc.ABC):
    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "normalization", "type"),), fast_llm_value=NormalizationType.rms_norm
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "normalization", "epsilon"),), export_names=(("rms_norm_eps",),)
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "kv_channels"),),
                export_names=(("head_dim"),),
            ),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "gated"),), fast_llm_value=True),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "add_linear_biases"),), fast_llm_value=False),
            RopeScalingParamConverter(
                fast_llm_names=(
                    ("transformer", "rotary", "type"),
                    ("transformer", "rotary", "scale_factor"),
                    ("transformer", "rotary", "low_frequency_factor"),
                    ("transformer", "rotary", "high_frequency_factor"),
                    ("transformer", "rotary", "original_context_length"),
                    ("transformer", "rotary", "attention_factor"),
                    ("transformer", "rotary", "beta_fast"),
                    ("transformer", "rotary", "beta_slow"),
                ),
                export_names=(("rope_scaling",),),
            ),
        ]


@dataclasses.dataclass
class RopeScalingParamConverter(ParamConverter):
    _HUGGINGFACE_NAMES = (
        "rope_type",
        "factor",
        "low_freq_factor",
        "high_freq_factor",
        "original_max_position_embeddings",
        "attention_factor",
        "beta_fast",
        "beta_slow",
    )

    def __post_init__(self):
        Assert.eq(len(self.fast_llm_names), 8)
        Assert.eq(len(self.export_names), 1)

    def export_params(self, fast_llm_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        rope_type, *parameters = fast_llm_values
        if rope_type == RotaryEmbeddingType.default:
            return (None,)
        elif rope_type == RotaryEmbeddingType.llama3:
            return ({key: value for key, value in zip(self._HUGGINGFACE_NAMES, ("llama3", *parameters), strict=True)},)
        elif rope_type == RotaryEmbeddingType.yarn:
            return ({key: value for key, value in zip(self._HUGGINGFACE_NAMES, ("yarn", *parameters), strict=True)},)
        else:
            raise ValueError(f"Unsupported rotary scaling type: {rope_type}")

    def import_params(self, export_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        (export_value,) = export_values
        if export_value is None or (rope_type := export_value[self._HUGGINGFACE_NAMES[0]]) == "default":
            return (RotaryEmbeddingType.default,) + (DEFAULT,) * 7
        elif rope_type == RotaryEmbeddingType.llama3:
            return ("llama3", *[export_value.get(key, DEFAULT) for key in self._HUGGINGFACE_NAMES[1:]])
        elif rope_type == RotaryEmbeddingType.yarn:
            return ("yarn", *[export_value.get(key, DEFAULT) for key in self._HUGGINGFACE_NAMES[1:]])
        else:
            raise ValueError(f"Unsupported rotary scaling type: {rope_type}")


class LlamaHuggingfaceCheckpointHandler(CommonLlamaHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = LlamaGPTHuggingfaceCheckpointFormat

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(export_names=(("architectures",),), export_value=["LlamaForCausalLM"]),
            # TODO: Llama supports biases
            ConstantExportParamConverter(export_names=(("attention_bias",),), export_value=False),
            ConstantExportParamConverter(export_names=(("mlp_bias",),), export_value=False),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        transformer_config: TransformerConfig = self._model.config.base_model.transformer
        return [
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_1",
                (f"{hf_prefix}.mlp.gate_proj", f"{hf_prefix}.mlp.up_proj"),
                transformer_config.add_mlp_bias,
                SplitWeightConverter,
            ),
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_2",
                f"{hf_prefix}.mlp.down_proj",
                transformer_config.add_mlp_bias,
                MLPLayer2Converter,
            ),
        ]


@dataclasses.dataclass
class IgnoreImportQwen2SlidingWindowParamsConverter(ParamConverter):
    def __post_init__(self):
        Assert.eq(len(self.fast_llm_names), 0)
        Assert.eq(len(self.export_names), 0)
        self.export_names = (("use_sliding_window",), ("sliding_window",), ("max_window_layers",))

    def export_params(self, fast_llm_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        return (MISSING, MISSING, MISSING)

    def import_params(self, export_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        # Default value for use_sliding_window in Qwen2 HF config is False
        if export_values[0] != MISSING and export_values[0] == True:
            logger.warning(
                f"The configuration parameters `{self.export_names[0]}={export_values[0]}`,"
                f" `{self.export_names[1]}={export_values[1]}`, `{self.export_names[2]}={export_values[2]}`"
                f" are ignored during conversion."
                f" If you intend to use them in Fast-LLM, make sure to set them explicitly in the model configuration."
            )
        return ()


class Qwen2HuggingfaceCheckpointHandler(CommonHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = Qwen2GPTHuggingfaceCheckpointFormat

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(export_names=(("architectures",),), export_value=["Qwen2ForCausalLM"]),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "normalization", "type"),), fast_llm_value=NormalizationType.rms_norm
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "normalization", "epsilon"),), export_names=(("rms_norm_eps",),)
            ),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "gated"),), fast_llm_value=True),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "add_linear_biases"),), fast_llm_value="only_attn_qkv"
            ),
            RopeScalingParamConverter(
                fast_llm_names=(
                    ("transformer", "rotary", "type"),
                    ("transformer", "rotary", "scale_factor"),
                    ("transformer", "rotary", "low_frequency_factor"),
                    ("transformer", "rotary", "high_frequency_factor"),
                    ("transformer", "rotary", "original_context_length"),
                    ("transformer", "rotary", "attention_factor"),
                    ("transformer", "rotary", "beta_fast"),
                    ("transformer", "rotary", "beta_slow"),
                ),
                export_names=(("rope_scaling",),),
            ),
            IgnoreImportQwen2SlidingWindowParamsConverter(),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        transformer_config: TransformerConfig = self._model.config.base_model.transformer
        return [
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_1",
                (f"{hf_prefix}.mlp.gate_proj", f"{hf_prefix}.mlp.up_proj"),
                transformer_config.add_mlp_bias,
                SplitWeightConverter,
            ),
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_2",
                f"{hf_prefix}.mlp.down_proj",
                transformer_config.add_mlp_bias,
                MLPLayer2Converter,
            ),
        ]


class MistralHuggingfaceCheckpointHandler(CommonLlamaHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = MistralGPTHuggingfaceCheckpointFormat

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(export_names=(("architectures",),), export_value=["MistralForCausalLM"]),
            IgnoreImportParamConverter(export_names=(("sliding_window",),), ignore_export_value=None),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        return [
            SplitWeightConverter(
                f"{fast_llm_prefix}.mlp.layer_1.weight",
                (f"{hf_prefix}.mlp.gate_proj.weight", f"{hf_prefix}.mlp.up_proj.weight"),
            ),
            MLPLayer2Converter(
                f"{fast_llm_prefix}.mlp.layer_2.weight",
                f"{hf_prefix}.mlp.down_proj.weight",
                self._model.config.base_model,
            ),
        ]


class MixtralHuggingfaceCheckpointHandler(CommonLlamaHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = MixtralGPTHuggingfaceCheckpointFormat

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(export_names=(("architectures",),), export_value=["MixtralForCausalLM"]),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "expert_routing_type"),), fast_llm_value=RoutingType.topk
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "num_experts"),), export_names=(("num_local_experts",),)
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "num_experts_per_token"),), export_names=(("num_experts_per_tok",),)
            ),
            IgnoreImportParamConverter(export_names=(("sliding_window",),), ignore_export_value=None),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        num_experts = self._model.config.base_model.transformer.num_experts
        return [
            WeightConverter(f"{fast_llm_prefix}.mlp.router.weight", f"{hf_prefix}.block_sparse_moe.gate.weight"),
            SplitWeightConverter(
                f"{fast_llm_prefix}.mlp.layer_1.weight",
                tuple(
                    f"{hf_prefix}.block_sparse_moe.experts.{i}.{w}.weight"
                    for i in range(num_experts)
                    for w in ("w1", "w3")
                ),
            ),
            MLPLayer2Converter(
                f"{fast_llm_prefix}.mlp.layer_2.weight",
                tuple(f"{hf_prefix}.block_sparse_moe.experts.{i}.w2.weight" for i in range(num_experts)),
                self._model.config.base_model,
            ),
        ]


class AutoGPTHuggingfaceCheckpointHandler(
    AutoStateDictCheckpointHandler, HuggingfaceStateDictCheckpointHandler, abc.ABC
):

    handler_map = {
        Starcoder2GPTHuggingfaceCheckpointFormat.name: Starcoder2HuggingfaceCheckpointHandler,
        LlamaGPTHuggingfaceCheckpointFormat.name: LlamaHuggingfaceCheckpointHandler,
        Qwen2GPTHuggingfaceCheckpointFormat.name: Qwen2HuggingfaceCheckpointHandler,
        MistralGPTHuggingfaceCheckpointFormat.name: MistralHuggingfaceCheckpointHandler,
        MixtralGPTHuggingfaceCheckpointFormat.name: MixtralHuggingfaceCheckpointHandler,
    }
