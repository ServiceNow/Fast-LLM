import abc
import dataclasses
import typing

import torch

from fast_llm.config import DEFAULT
from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    AutoStateDictCheckpointHandler,
    ConstantExportParamConverter,
    ConstantImportParamConverter,
    HuggingfaceStateDictCheckpointHandler,
    IgnoreImportParamConverter,
    IgnoreWeightConverter,
    MappedConfigParamConverter,
    ParamConverter,
    RenameParamConverter,
    SplitWeightConverter,
    WeightConverter,
)
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.functional.config import ActivationType
from fast_llm.functional.rotary import convert_rotary_complex_to_real, convert_rotary_real_to_complex
from fast_llm.layers.common.config import NormalizationType
from fast_llm.layers.transformer.config import RotaryEmbeddingType, RoutingType
from fast_llm.models.gpt.config import (
    GPTArchitectureConfig,
    GPTModelConfig,
    LlamaGPTHuggingfaceCheckpointFormat,
    MistralGPTHuggingfaceCheckpointFormat,
    MixtralGPTHuggingfaceCheckpointFormat,
    Starcoder2GPTHuggingfaceCheckpointFormat,
)
from fast_llm.models.gpt.model import GPTModel
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    pass


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

    @abc.abstractmethod
    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str):
        pass

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

    def _create_weight_converters(self) -> list[WeightConverter]:
        converters = []
        num_layers = self._model.base_model_config.transformer.num_layers
        norm_bias: bool = self._model.base_model_config.transformer.normalization.type == NormalizationType.layer_norm
        linear_bias: bool = self._model.base_model_config.transformer.add_linear_biases

        # Embedding and output
        if self._model.base_model_config.tie_word_embeddings:
            converters.append(WeightConverter("layers.0.word_embeddings_weight", "model.embed_tokens.weight"))
            converters.append(IgnoreWeightConverter((), "lm_head.weight"))
        else:
            converters.append(WeightConverter("layers.0.word_embeddings_weight", "model.embed_tokens.weight"))
            converters.append(WeightConverter(f"layers.{num_layers + 1}.output_weights", "lm_head.weight"))

        # Final norm
        converters += self._get_weight_and_bias_converters(
            f"layers.{num_layers + 1}.final_norm", "model.norm", norm_bias
        )

        for i in range(num_layers):
            # Self-attn
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.self_attn.query",
                f"model.layers.{i}.self_attn.q_proj",
                linear_bias,
                QueryWeightConverter,
            )
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.self_attn.key_value",
                (f"model.layers.{i}.self_attn.k_proj", f"model.layers.{i}.self_attn.v_proj"),
                linear_bias,
                KeyValueWeightConverter,
            )
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.self_attn.dense", f"model.layers.{i}.self_attn.o_proj", linear_bias
            )

            # Norm
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.norm_1", f"model.layers.{i}.input_layernorm", norm_bias
            )
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.norm_2", f"model.layers.{i}.post_attention_layernorm", norm_bias
            )

            # MLP
            converters += self._get_mlp_converters(f"layers.{i+1}", f"model.layers.{i}")

        return converters

    def _get_weight_and_bias_converters(
        self,
        fast_llm_prefix: str | tuple[str, ...],
        hf_prefix: str | tuple[str, ...],
        use_bias: bool,
        cls=WeightConverter,
    ):
        if isinstance(fast_llm_prefix, str):
            fast_llm_prefix = (fast_llm_prefix,)
        if isinstance(hf_prefix, str):
            hf_prefix = (hf_prefix,)
        converters = [
            cls(
                tuple(f"{prefix}.weight" for prefix in fast_llm_prefix),
                tuple(f"{prefix}.weight" for prefix in hf_prefix),
                self._model.base_model_config,
            )
        ]
        if use_bias:
            converters.append(
                cls(
                    tuple(f"{prefix}.bias" for prefix in fast_llm_prefix),
                    tuple(f"{prefix}.bias" for prefix in hf_prefix),
                    self._model.base_model_config,
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

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str):
        linear_bias: bool = self._model.base_model_config.transformer.add_linear_biases
        return [
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_1", f"{hf_prefix}.mlp.c_fc", linear_bias
            ),
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_2", f"{hf_prefix}.mlp.c_proj", linear_bias, MLPLayer2Converter
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
            ConstantImportParamConverter(fast_llm_names=(("transformer", "gated"),), fast_llm_value=True),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "add_linear_biases"),), fast_llm_value=False),
        ]


@dataclasses.dataclass
class RopeScalingParamConverter(ParamConverter):
    _HUGGINGFACE_NAMES = (
        "rope_type",
        "factor",
        "low_freq_factor",
        "high_freq_factor",
        "original_max_position_embeddings",
    )

    def __post_init__(self):
        Assert.eq(len(self.fast_llm_names), 5)
        Assert.eq(len(self.export_names), 1)

    def export_params(self, fast_llm_values):
        rope_type, *parameters = fast_llm_values
        if rope_type == RotaryEmbeddingType.default:
            return (None,)
        elif rope_type == RotaryEmbeddingType.llama3:
            return ({key: value for key, value in zip(self._HUGGINGFACE_NAMES, ("llama3", *parameters), strict=True)},)
        else:
            raise ValueError(f"Unsupported rotary scaling type: {rope_type}")

    def import_params(self, export_values):
        (export_value,) = export_values
        if export_value is None or (rope_type := export_value[self._HUGGINGFACE_NAMES[0]]) == "default":
            return (RotaryEmbeddingType.default,) + (DEFAULT,) * 4
        elif rope_type == RotaryEmbeddingType.llama3:
            # TODO: Is it safe to assume all values are provided?
            return ("llama3", *[export_value[key] for key in self._HUGGINGFACE_NAMES[1:]])
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
            RopeScalingParamConverter(
                fast_llm_names=(
                    ("transformer", "rotary", "type"),
                    ("transformer", "rotary", "scale_factor"),
                    ("transformer", "rotary", "low_frequency_factor"),
                    ("transformer", "rotary", "high_frequency_factor"),
                    ("transformer", "rotary", "original_context_length"),
                ),
                export_names=(("rope_scaling",),),
            ),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str):
        linear_bias: bool = self._model.base_model_config.transformer.add_linear_biases
        return [
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_1",
                (f"{hf_prefix}.mlp.gate_proj", f"{hf_prefix}.mlp.up_proj"),
                linear_bias,
                SplitWeightConverter,
            ),
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_2",
                f"{hf_prefix}.mlp.down_proj",
                linear_bias,
                MLPLayer2Converter,
            ),
        ]


class MistralHuggingfaceCheckpointHandler(CommonLlamaHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = MistralGPTHuggingfaceCheckpointFormat

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(export_names=(("architectures",),), export_value=["MistralForCausalLM"]),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "rotary", "type"),), fast_llm_value=RotaryEmbeddingType.default
            ),
            IgnoreImportParamConverter(export_names=(("sliding_window",),), ignore_export_value=None),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str):
        return [
            SplitWeightConverter(
                f"{fast_llm_prefix}.mlp.layer_1.weight",
                (f"{hf_prefix}.mlp.gate_proj.weight", f"{hf_prefix}.mlp.up_proj.weight"),
            ),
            MLPLayer2Converter(
                f"{fast_llm_prefix}.mlp.layer_2.weight",
                f"{hf_prefix}.mlp.down_proj.weight",
                self._model.base_model_config,
            ),
        ]


class MixtralHuggingfaceCheckpointHandler(CommonLlamaHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = MixtralGPTHuggingfaceCheckpointFormat

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(export_names=(("architectures",),), export_value=["MixtralForCausalLM"]),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "rotary", "type"),), fast_llm_value=RotaryEmbeddingType.default
            ),
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

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str):
        num_experts = self._model.base_model_config.transformer.num_experts
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
                self._model.base_model_config,
            ),
        ]


class AutoGPTHuggingfaceCheckpointHandler(
    AutoStateDictCheckpointHandler, HuggingfaceStateDictCheckpointHandler, abc.ABC
):

    handler_map = {
        Starcoder2GPTHuggingfaceCheckpointFormat.name: Starcoder2HuggingfaceCheckpointHandler,
        LlamaGPTHuggingfaceCheckpointFormat.name: LlamaHuggingfaceCheckpointHandler,
        MistralGPTHuggingfaceCheckpointFormat.name: MistralHuggingfaceCheckpointHandler,
        MixtralGPTHuggingfaceCheckpointFormat.name: MixtralHuggingfaceCheckpointHandler,
    }
