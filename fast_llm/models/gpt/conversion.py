import abc
import dataclasses
import logging
import typing

import torch
from transformers.configuration_utils import PretrainedConfig

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
from fast_llm.engine.checkpoint.huggingface import CustomModelingExportMixin, HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.rotary.config import DefaultRotaryConfig, Llama3RotaryConfig, YarnRotaryConfig
from fast_llm.layers.attention.rotary.rotary import convert_rotary_complex_to_real, convert_rotary_real_to_complex
from fast_llm.layers.block.config import BlockConfig
from fast_llm.layers.block.mlp.config import RoutingType
from fast_llm.layers.common.normalization.config import LayerNormalizationConfig
from fast_llm.models.gpt.config import (
    DiffusionDreamGPTHuggingfaceCheckpointFormat,
    DiffusionLlamaGPTHuggingfaceCheckpointFormat,
    GPTBaseModelConfig,
    GPTModelConfig,
    LlamaGPTHuggingfaceCheckpointFormat,
    MistralGPTHuggingfaceCheckpointFormat,
    MixtralGPTHuggingfaceCheckpointFormat,
    MTPLlamaGPTHuggingfaceCheckpointFormat,
    Qwen2GPTHuggingfaceCheckpointFormat,
    Starcoder2GPTHuggingfaceCheckpointFormat,
)
from fast_llm.models.gpt.external.diffusion_dream.configuration_dream import DreamConfig
from fast_llm.models.gpt.external.diffusion_llama.configuration_diffusion_llama import DiffusionLlamaConfig
from fast_llm.models.gpt.external.mtp_llama.configuration_mtp_llama import MTPLlamaConfig
from fast_llm.models.gpt.model import GPTModel
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert, div

if typing.TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class HiddenSizeParamConverter(ParamConverter):
    """
    Some HF models don't have a `head_dim` parameter, and instead use hidden_size // heads
    """

    def __post_init__(self):
        Assert.eq(len(self.fast_llm_names), 3)
        Assert.eq(len(self.export_names), 2)

    def export_params(self, fast_llm_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        hidden_size, heads, head_size = fast_llm_values
        Assert.eq(head_size * heads, hidden_size)
        return hidden_size, heads

    def import_params(self, export_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        hidden_size, heads = export_values
        return hidden_size, heads, div(hidden_size, heads)


class QueryWeightConverter(WeightConverter):
    # Hf uses the real format for rotary embeddings.
    _config: GPTBaseModelConfig

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (query,) = weight
        if self._config.transformer.mixer.rotary.complex_format:
            query = convert_rotary_complex_to_real(query[:], self._config.transformer.mixer.head_size, 0)
        return (query,)

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (query,) = weight
        if self._config.transformer.mixer.rotary.complex_format:
            query = convert_rotary_real_to_complex(query[:], self._config.transformer.mixer.head_size, 0)
        return (query,)


class KeyValueWeightConverter(WeightConverter):
    # Hf uses the real format for rotary embeddings, and keeps the key and value separate.
    _config: GPTBaseModelConfig

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (key_value,) = weight
        key, value = key_value[:].chunk(2)
        if self._config.transformer.mixer.rotary.complex_format:
            key = convert_rotary_complex_to_real(key, self._config.transformer.mixer.head_size, 0)
        return key, value

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        key, value = weight
        if self._config.transformer.mixer.rotary.complex_format:
            key = convert_rotary_real_to_complex(key[:], self._config.transformer.mixer.head_size, 0)
        key_value = torch.cat([key[:], value[:]])
        return (key_value,)


class MLPLayer2Converter(WeightConverter):
    # Similar to SplitWeightConverter, but handles the optional MLP transpose.
    # Still ok for non-gated (trivial split) and biases (trivial 1d transpose)
    _config: GPTBaseModelConfig

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
    architecture: typing.ClassVar[str]
    """
    Common converter for llama-based huggingface models (llama, starcoder2, mistral, mixtral)
    """

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(export_names=(("architectures",),), export_value=[cls.architecture]),
            ConstantImportParamConverter(
                fast_llm_names=(
                    (
                        "embeddings_layer",
                        "position_embeddings",
                        "enabled",
                    ),
                ),
                fast_llm_value=False,
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "mixer", "rotary", "theta"),), export_names=(("rope_theta",),)
            ),
            MappedConfigParamConverter(
                fast_llm_names=(("transformer", "mlp", "activation"),),
                export_names=(("hidden_act",),),
                fast_llm_value=ActivationType.from_hf_name,
                export_value=lambda activation_type: activation_type.hf_name,
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "num_layers"),),
                export_names=(("num_hidden_layers",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "mixer", "head_groups"),),
                export_names=(("num_key_value_heads",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "mlp", "intermediate_size"),),
                export_names=(("intermediate_size",),),
            ),
            RenameParamConverter(
                fast_llm_names=(
                    (
                        "embeddings_layer",
                        "vocab_size",
                    ),
                ),
                export_names=(("vocab_size",),),
            ),
            RenameParamConverter(
                fast_llm_names=(
                    (
                        "output_layer",
                        "tied_weight",
                    ),
                ),
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
            converters += self._create_transformer_layer_converters(f"layers.{i+1}", f"model.layers.{i}")

        return converters

    def _create_transformer_layer_converters(
        self, fast_llm_layer_name: str, hf_layer_name: str, ignore_export: bool = False
    ) -> list[WeightConverter]:
        transformer_config: BlockConfig = self._model.config.base_model.transformer
        norm_bias: bool = isinstance(self._model.config.base_model.transformer.normalization, LayerNormalizationConfig)
        converters = []
        names_bias_cls = [
            # Self-attn
            (
                f"{fast_llm_layer_name}.mixer.query",
                f"{hf_layer_name}.self_attn.q_proj",
                # TODO: Fix
                transformer_config.mixer.add_linear_biases,
                QueryWeightConverter,
            ),
            (
                f"{fast_llm_layer_name}.mixer.key_value",
                (f"{hf_layer_name}.self_attn.k_proj", f"{hf_layer_name}.self_attn.v_proj"),
                # TODO: Fix
                transformer_config.mixer.add_linear_biases,
                KeyValueWeightConverter,
            ),
            (
                f"{fast_llm_layer_name}.mixer.dense",
                f"{hf_layer_name}.self_attn.o_proj",
                # TODO: Fix
                transformer_config.mixer.add_linear_biases,
                WeightConverter,
            ),
            # Norm
            (
                f"{fast_llm_layer_name}.norm_1",
                f"{hf_layer_name}.input_layernorm",
                norm_bias,
                WeightConverter,
            ),
            (
                f"{fast_llm_layer_name}.norm_2",
                f"{hf_layer_name}.post_attention_layernorm",
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
                f"{fast_llm_layer_name}.mlp.layer_1",
                (),
                # TODO: Fix
                transformer_config.mlp.add_linear_biases,
                cls=IgnoreExportWeightConverter,
            )
            converters += self._get_weight_and_bias_converters(
                f"{fast_llm_layer_name}.mlp.layer_2",
                (),
                # TODO: Fix
                transformer_config.mlp.add_linear_biases,
                cls=IgnoreExportWeightConverter,
            )
            converters += [IgnoreExportWeightConverter(f"{fast_llm_layer_name}.mlp.router.weight", ())]
        else:
            converters += self._get_mlp_converters(f"{fast_llm_layer_name}", f"{hf_layer_name}")
        return converters

    def _create_lm_head_converters(self) -> list[WeightConverter]:
        num_layers = self._model.config.base_model.transformer.num_layers
        prediction_heads = self._model.config.base_model.output_layer.prediction_heads
        norm_bias: bool = isinstance(self._model.config.base_model.transformer.normalization, LayerNormalizationConfig)
        converters = []

        # Next-token prediction head
        # Final norm
        converters += self._get_weight_and_bias_converters(
            f"layers.{num_layers + 1}.final_norm", "model.norm", norm_bias
        )
        # Output weights
        if self._model.config.base_model.output_layer.tied_weight:
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
            converters += self._create_transformer_layer_converters(
                f"layers.{mtp_transformer_layer_index + 1}", "", ignore_export=True
            )
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
    architecture: typing.ClassVar[str] = "Starcoder2ForCausalLM"

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            HiddenSizeParamConverter(
                fast_llm_names=(
                    ("transformer", "hidden_size"),
                    ("transformer", "mixer", "heads"),
                    ("transformer", "mixer", "head_size"),
                ),
                export_names=(("hidden_size",), ("num_attention_heads",)),
            ),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "mixer", "rotary", "type"),),
                fast_llm_value=DefaultRotaryConfig.dynamic_type_name,
            ),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "mixer", "add_linear_biases"),), fast_llm_value=True
            ),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "normalization", "type"),),
                fast_llm_value="layer_norm",
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "normalization", "epsilon"),), export_names=(("norm_epsilon",),)
            ),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "mlp", "gated"),), fast_llm_value=False),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "mlp", "add_linear_biases"),), fast_llm_value=True
            ),
            ConstantImportParamConverter(
                fast_llm_names=(("output_layer", "normalization", "type"),),
                fast_llm_value="layer_norm",
            ),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        transformer_config: BlockConfig = self._model.config.base_model.transformer
        return [
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_1",
                f"{hf_prefix}.mlp.c_fc",
                # TODO: Fix
                transformer_config.mlp.add_linear_biases,
            ),
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_2",
                f"{hf_prefix}.mlp.c_proj",
                # TODO: Fix
                transformer_config.mlp.add_linear_biases,
                MLPLayer2Converter,
            ),
        ]


class CommonLlamaHuggingfaceCheckpointHandler(CommonHuggingfaceCheckpointHandler, abc.ABC):
    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "normalization", "type"),),
                fast_llm_value="rms_norm",
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "normalization", "epsilon"),), export_names=(("rms_norm_eps",),)
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "hidden_size"),),
                export_names=(("hidden_size",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "mixer", "heads"),),
                export_names=(("num_attention_heads",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "mixer", "head_size"),),
                export_names=(("head_dim",),),
            ),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "mixer", "add_linear_biases"),), fast_llm_value=False
            ),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "mlp", "gated"),), fast_llm_value=True),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "mlp", "add_linear_biases"),), fast_llm_value=False
            ),
            LLamaRotaryParamConverter(
                fast_llm_names=(("transformer", "mixer", "rotary"),),
                export_names=(
                    ("rope_theta",),
                    ("rope_scaling",),
                ),
            ),
            ConstantImportParamConverter(
                fast_llm_names=(("output_layer", "normalization", "type"),),
                fast_llm_value="rms_norm",
            ),
        ]


@dataclasses.dataclass
class LLamaRotaryParamConverter(ParamConverter):
    def __post_init__(self):
        Assert.eq(len(self.fast_llm_names), 1)
        Assert.eq(len(self.export_names), 2)

    def export_params(self, fast_llm_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        (rotary_config,) = fast_llm_values
        if type(rotary_config) is DefaultRotaryConfig:
            rotary_scaling = {
                "rope_type": "default",
            }
        elif type(rotary_config) is Llama3RotaryConfig:
            rotary_scaling = {
                "rope_type": "llama3",
                "factor": rotary_config.scale_factor,
                "low_freq_factor": rotary_config.low_frequency_factor,
                "high_freq_factor": rotary_config.high_frequency_factor,
                "original_max_position_embeddings": rotary_config.original_context_length,
            }
        elif type(rotary_config) is YarnRotaryConfig:
            rotary_scaling = {
                "rope_type": "yarn",
                "attention_factor": rotary_config.attention_factor,
                "beta_fast": rotary_config.beta_fast,
                "beta_slow": rotary_config.beta_slow,
                "original_max_position_embeddings": rotary_config.original_context_length,
            }
        else:
            raise ValueError(f"Unsupported rotary type: {type(rotary_config).__name__}")

        return rotary_config.theta, rotary_scaling

    def import_params(self, export_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]:
        rotary_theta, rope_scaling = export_values
        rotary_type = "default" if rope_scaling in (None, MISSING) else rope_scaling.get("rope_type", "default")
        rotary_config = {
            "type": rotary_type,
            "theta": rotary_theta,
        }
        if rotary_type == "default":
            pass
        elif rotary_type == "llama3":
            rotary_config.update(
                {
                    "scale_factor": rope_scaling.get("factor", DEFAULT),
                    "low_frequency_factor": rope_scaling.get("low_freq_factor", DEFAULT),
                    "high_frequency_factor": rope_scaling.get("high_freq_factor", DEFAULT),
                    "original_context_length": rope_scaling.get("original_max_position_embeddings", DEFAULT),
                }
            )
        elif rotary_type == "yarn":
            rotary_config.update(
                {
                    "attention_factor": rope_scaling.get("attention_factor", DEFAULT),
                    "beta_fast": rope_scaling.get("beta_fast", DEFAULT),
                    "beta_slow": rope_scaling.get("beta_slow", DEFAULT),
                    "original_context_length": rope_scaling.get("original_max_position_embeddings", DEFAULT),
                }
            )
        return (rotary_config,)  # RotaryConfig.from_dict(rotary_config)


class LlamaHuggingfaceCheckpointHandler(CommonLlamaHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = LlamaGPTHuggingfaceCheckpointFormat
    architecture: typing.ClassVar[str] = "LlamaForCausalLM"

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        transformer_config: BlockConfig = self._model.config.base_model.transformer
        return [
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_1",
                (f"{hf_prefix}.mlp.gate_proj", f"{hf_prefix}.mlp.up_proj"),
                # TODO: Fix
                transformer_config.mlp.add_linear_biases,
                SplitWeightConverter,
            ),
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_2",
                f"{hf_prefix}.mlp.down_proj",
                # TODO: Fix
                transformer_config.mlp.add_linear_biases,
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
    architecture: typing.ClassVar[str] = "Qwen2ForCausalLM"

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "normalization", "type"),),
                fast_llm_value="rms_norm",
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "normalization", "epsilon"),), export_names=(("rms_norm_eps",),)
            ),
            HiddenSizeParamConverter(
                fast_llm_names=(
                    ("transformer", "hidden_size"),
                    ("transformer", "mixer", "heads"),
                    ("transformer", "mixer", "head_size"),
                ),
                export_names=(("hidden_size",), ("num_attention_heads",)),
            ),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "mlp", "gated"),), fast_llm_value=True),
            # TODO: Fix
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "add_linear_biases"),), fast_llm_value="only_attn_qkv"
            ),
            LLamaRotaryParamConverter(
                fast_llm_names=(("transformer", "mixer", "rotary"),),
                export_names=(
                    ("rope_theta",),
                    ("rope_scaling",),
                ),
            ),
            ConstantImportParamConverter(
                fast_llm_names=(("output_layer", "normalization", "type"),),
                fast_llm_value="rms_norm",
            ),
            IgnoreImportQwen2SlidingWindowParamsConverter(),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        transformer_config: BlockConfig = self._model.config.base_model.transformer
        return [
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_1",
                (f"{hf_prefix}.mlp.gate_proj", f"{hf_prefix}.mlp.up_proj"),
                # TODO: Fix
                transformer_config.mlp.add_linear_biases,
                SplitWeightConverter,
            ),
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_2",
                f"{hf_prefix}.mlp.down_proj",
                # TODO: Fix
                transformer_config.mlp.add_linear_biases,
                MLPLayer2Converter,
            ),
        ]


class MistralHuggingfaceCheckpointHandler(CommonLlamaHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = MistralGPTHuggingfaceCheckpointFormat
    architecture: typing.ClassVar[str] = "MistralForCausalLM"

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
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
    architecture: typing.ClassVar[str] = "MixtralForCausalLM"

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantImportParamConverter(fast_llm_names=(("transformer", "mlp", "type"),), fast_llm_value="moe"),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "mlp", "routing"),), fast_llm_value=RoutingType.topk
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "mlp", "experts"),), export_names=(("num_local_experts",),)
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "mlp", "experts_per_token"),),
                export_names=(("num_experts_per_tok",),),
            ),
            IgnoreImportParamConverter(export_names=(("sliding_window",),), ignore_export_value=None),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        num_experts = self._model.config.base_model.transformer.mlp.experts
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


class MTPLlamaHuggingfaceCheckpointHandler(CustomModelingExportMixin, CommonLlamaHuggingfaceCheckpointHandler):
    from fast_llm.models.gpt.external.mtp_llama import configuration_mtp_llama, modeling_mtp_llama

    format: typing.ClassVar[type[CheckpointFormat]] = MTPLlamaGPTHuggingfaceCheckpointFormat
    architecture: typing.ClassVar[str] = "MTPLlamaForCausalLM"
    modeling_file = modeling_mtp_llama.__file__
    configuration_file = configuration_mtp_llama.__file__
    configuration_cls: typing.ClassVar[type[PretrainedConfig]] = MTPLlamaConfig

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(
                export_names=(("auto_map",),),
                export_value={
                    "AutoConfig": "configuration_mtp_llama.MTPLlamaConfig",
                    "AutoModel": "modeling_mtp_llama.MTPLlamaModel",
                    "AutoModelForCausalLM": "modeling_mtp_llama.MTPLlamaForCausalLM",
                },
            ),
            RenameParamConverter(
                fast_llm_names=(
                    (
                        "output_layer",
                        "prediction_heads",
                    ),
                ),
                export_names=(("prediction_heads",),),
            ),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        transformer_config: BlockConfig = self._model.config.base_model.transformer
        return [
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_1",
                (f"{hf_prefix}.mlp.gate_proj", f"{hf_prefix}.mlp.up_proj"),
                # TODO: Fix
                transformer_config.mlp.add_linear_biases,
                SplitWeightConverter,
            ),
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_2",
                f"{hf_prefix}.mlp.down_proj",
                # TODO: Fix
                transformer_config.mlp.add_linear_biases,
                MLPLayer2Converter,
            ),
        ]

    # Override base method to handle the MTP heads
    def _create_lm_head_converters(self) -> list[WeightConverter]:
        num_layers = self._model.config.base_model.transformer.num_layers
        prediction_heads = self._model.config.base_model.output_layer.prediction_heads
        norm_bias: bool = isinstance(self._model.config.base_model.transformer.normalization, LayerNormalizationConfig)
        converters = []

        # Next-token prediction head
        # Transformer layer is already handled in the transformer layer converters
        # Final norm
        converters += self._get_weight_and_bias_converters(
            f"layers.{num_layers + 1}.final_norm", "model.mtp_norms.0", norm_bias
        )
        # Multi-token prediction head
        for i in range(1, prediction_heads):
            mtp_transformer_layer_index = num_layers - 1 + 2 * i
            # MTP transformer layer
            converters += self._create_transformer_layer_converters(
                f"layers.{mtp_transformer_layer_index + 1}",
                f"model.mtp_heads.{i - 1}",
            )
            # MTP output norm
            converters += self._get_weight_and_bias_converters(
                f"layers.{mtp_transformer_layer_index + 2}.final_norm",
                f"model.mtp_norms.{i}",
                norm_bias,
            )
        # Output weights
        if self._model.config.base_model.output_layer.tied_weight:
            converters.append(IgnoreImportWeightConverter((), "lm_head.weight"))
        else:
            converters.append(WeightConverter(f"layers.{num_layers + 1}.output_weights", "lm_head.weight"))

        return converters


class DiffusionDreamHuggingfaceCheckpointHandler(CustomModelingExportMixin, Qwen2HuggingfaceCheckpointHandler):
    """
    Handler for DiffusionDream Huggingface checkpoints.
    Inherits from Qwen2HuggingfaceCheckpointHandler (and CustomModelingExportMixin),
    but overrides _create_config_converters to update architectures and auto_map.
    """

    from fast_llm.models.gpt.external.diffusion_dream import configuration_dream, generation_utils, modeling_dream

    format: typing.ClassVar[type[CheckpointFormat]] = DiffusionDreamGPTHuggingfaceCheckpointFormat
    architecture: typing.ClassVar[str] = "DreamModel"
    modeling_file = modeling_dream.__file__
    configuration_file = configuration_dream.__file__
    generation_utils_file = generation_utils.__file__
    configuration_cls: typing.ClassVar[type[PretrainedConfig]] = DreamConfig

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(
                export_names=(("auto_map",),),
                export_value={
                    "AutoConfig": "configuration_dream.DreamConfig",
                    "AutoModel": "modeling_dream.DreamModel",
                },
            ),
        ]


class DiffusionLlamaHuggingfaceCheckpointHandler(CustomModelingExportMixin, LlamaHuggingfaceCheckpointHandler):

    from fast_llm.models.gpt.external.diffusion_llama import (
        configuration_diffusion_llama,
        generation_utils,
        modeling_diffusion_llama,
    )

    format: typing.ClassVar[type[CheckpointFormat]] = DiffusionLlamaGPTHuggingfaceCheckpointFormat
    architecture: typing.ClassVar[str] = "DiffusionLlamaModel"
    modeling_file = modeling_diffusion_llama.__file__
    configuration_file = configuration_diffusion_llama.__file__
    generation_utils_file = generation_utils.__file__
    configuration_cls: typing.ClassVar[type[PretrainedConfig]] = DiffusionLlamaConfig

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(
                export_names=(("auto_map",),),
                export_value={
                    "AutoConfig": "configuration_diffusion_llama.DiffusionLlamaConfig",
                    "AutoModel": "modeling_diffusion_llama.DiffusionLlamaModel",
                },
            ),
            # TODO: include when the mask diffusion training is implemented;
            # since the imported model (llama) for CPT doesn't have it but the exported model (diffusion llama) does need to have this token.
            # RenameParamConverter(
            #     fast_llm_names=(("mask_token_id",),),
            #     export_names=(("mask_token_id",),),
            # ),
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
        MTPLlamaGPTHuggingfaceCheckpointFormat.name: MTPLlamaHuggingfaceCheckpointHandler,
        DiffusionDreamGPTHuggingfaceCheckpointFormat.name: DiffusionDreamHuggingfaceCheckpointHandler,
        DiffusionLlamaGPTHuggingfaceCheckpointFormat.name: DiffusionLlamaHuggingfaceCheckpointHandler,
    }
