import abc
import math
import typing

import torch

from fast_llm.engine.multi_stage.conversion import (
    AutoModelConverter,
    ConstantExportParamConverter,
    ConstantImportParamConverter,
    HuggingfaceModelConverter,
    IgnoreImportParamConverter,
    IgnoreWeightConverter,
    MappedConfigParamConverter,
    ParamConverter,
    SplitWeightConverter,
    WeightConverter,
)
from fast_llm.functional.config import ActivationType
from fast_llm.functional.rotary import convert_rotary_complex_to_real, convert_rotary_real_to_complex
from fast_llm.layers.common.config import NormalizationType
from fast_llm.layers.transformer.config import RoutingType
from fast_llm.models.stardoc.config import StarDocArchitectureConfig, StarDocBaseModelConfig, HuggingfaceModelType
from fast_llm.tensor import SafeTensorSlice

if typing.TYPE_CHECKING:
    pass


class QueryWeightConverter(WeightConverter):
    # Hf uses the real format for rotary embeddings.
    _config: StarDocArchitectureConfig

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (query,) = weight
        if self._config.transformer.complex_rotary_embeddings:
            query = convert_rotary_complex_to_real(query[:], self._config.transformer.kv_channels, 0)
        return (query,)

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (query,) = weight
        if self._config.transformer.complex_rotary_embeddings:
            query = convert_rotary_real_to_complex(query[:], self._config.transformer.kv_channels, 0)
        return (query,)


class KeyValueWeightConverter(WeightConverter):
    # Hf uses the real format for rotary embeddings, and keeps the key and value separate.
    _config: StarDocArchitectureConfig

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (key_value,) = weight
        key, value = key_value[:].chunk(2)
        if self._config.transformer.complex_rotary_embeddings:
            key = convert_rotary_complex_to_real(key, self._config.transformer.kv_channels, 0)
        return key, value

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        key, value = weight
        if self._config.transformer.complex_rotary_embeddings:
            key = convert_rotary_real_to_complex(key[:], self._config.transformer.kv_channels, 0)
        key_value = torch.cat([key[:], value[:]])
        return (key_value,)


class MLPLayer2Converter(WeightConverter):
    # Similar to SplitWeightConverter, but handles the optional MLP transpose.
    # Still ok for non-gated (trivial split) and biases (trivial 1d transpose)
    _config: StarDocArchitectureConfig

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


class CommonHuggingfaceConverter(HuggingfaceModelConverter):
    config: StarDocArchitectureConfig
    _base_model_cls = StarDocBaseModelConfig
    """
    Common converter for llama-based huggingface models (llama, starcoder2, mistral, mixtral)
    """

    @abc.abstractmethod
    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str):
        pass

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantImportParamConverter(("multimodal_model", "image_encoder_hidden_size",), None, 1024),
            ConstantImportParamConverter(("multimodal_model", "num_image_tokens",), None, 256),
            ConstantImportParamConverter(("multimodal_model", "max_num_images",), None, 10),
            ConstantImportParamConverter(("use_position_embeddings",), None, False),
            ConstantImportParamConverter(("transformer", "use_rotary_embeddings"), None, True),
            MappedConfigParamConverter(
                ("transformer", "rotary_embedding_scale"), "rope_theta", lambda x: -math.log(x), lambda x: math.exp(-x)
            ),
            MappedConfigParamConverter(
                ("transformer", "activation_type"),
                "hidden_act",
                ActivationType.from_hf_name,
                lambda activation_type: activation_type.hf_name,
            ),
            ParamConverter(("transformer", "num_layers"), "num_hidden_layers"),
            ParamConverter(("transformer", "hidden_size"), "hidden_size"),
            ParamConverter(("transformer", "num_attention_heads"), "num_attention_heads"),
            ParamConverter(("transformer", "head_groups"), "num_key_value_heads"),
            ParamConverter(("transformer", "ffn_hidden_size"), "intermediate_size"),
            ParamConverter(("vocab_size",), "vocab_size"),
            ParamConverter(("tie_word_embeddings",), "tie_word_embeddings"),
        ]

    def _create_weight_converters(self) -> list[WeightConverter]:
        converters = []
        num_layers = self.config.transformer.num_layers
        norm_bias: bool = self.config.transformer.normalization.normalization_type == NormalizationType.layer_norm
        linear_bias: bool = self.config.transformer.add_linear_biases

        # Vision encoder
        converters.append(WeightConverter("layers.0.visual_encoder.class_embedding", "visual_encoder.class_embedding")) 
        converters.append(WeightConverter("layers.0.visual_encoder.positional_embedding", "visual_encoder.positional_embedding")) 
        converters.append(WeightConverter("layers.0.visual_encoder.proj", "visual_encoder.proj")) 
        converters.append(WeightConverter("layers.0.visual_encoder.conv1.weight", "visual_encoder.conv1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.ln_pre.weight", "visual_encoder.ln_pre.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.ln_pre.bias", "visual_encoder.ln_pre.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.0.ln_1.weight", "visual_encoder.transformer.resblocks.0.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.0.ln_1.bias", "visual_encoder.transformer.resblocks.0.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.0.attn.in_proj_weight", "visual_encoder.transformer.resblocks.0.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.0.attn.in_proj_bias", "visual_encoder.transformer.resblocks.0.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.0.attn.out_proj.weight", "visual_encoder.transformer.resblocks.0.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.0.attn.out_proj.bias", "visual_encoder.transformer.resblocks.0.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.0.ln_2.weight", "visual_encoder.transformer.resblocks.0.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.0.ln_2.bias", "visual_encoder.transformer.resblocks.0.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.0.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.0.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.0.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.0.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.0.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.0.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.0.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.0.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.1.ln_1.weight", "visual_encoder.transformer.resblocks.1.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.1.ln_1.bias", "visual_encoder.transformer.resblocks.1.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.1.attn.in_proj_weight", "visual_encoder.transformer.resblocks.1.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.1.attn.in_proj_bias", "visual_encoder.transformer.resblocks.1.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.1.attn.out_proj.weight", "visual_encoder.transformer.resblocks.1.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.1.attn.out_proj.bias", "visual_encoder.transformer.resblocks.1.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.1.ln_2.weight", "visual_encoder.transformer.resblocks.1.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.1.ln_2.bias", "visual_encoder.transformer.resblocks.1.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.1.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.1.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.1.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.1.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.1.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.1.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.1.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.1.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.2.ln_1.weight", "visual_encoder.transformer.resblocks.2.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.2.ln_1.bias", "visual_encoder.transformer.resblocks.2.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.2.attn.in_proj_weight", "visual_encoder.transformer.resblocks.2.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.2.attn.in_proj_bias", "visual_encoder.transformer.resblocks.2.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.2.attn.out_proj.weight", "visual_encoder.transformer.resblocks.2.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.2.attn.out_proj.bias", "visual_encoder.transformer.resblocks.2.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.2.ln_2.weight", "visual_encoder.transformer.resblocks.2.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.2.ln_2.bias", "visual_encoder.transformer.resblocks.2.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.2.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.2.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.2.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.2.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.2.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.2.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.2.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.2.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.3.ln_1.weight", "visual_encoder.transformer.resblocks.3.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.3.ln_1.bias", "visual_encoder.transformer.resblocks.3.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.3.attn.in_proj_weight", "visual_encoder.transformer.resblocks.3.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.3.attn.in_proj_bias", "visual_encoder.transformer.resblocks.3.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.3.attn.out_proj.weight", "visual_encoder.transformer.resblocks.3.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.3.attn.out_proj.bias", "visual_encoder.transformer.resblocks.3.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.3.ln_2.weight", "visual_encoder.transformer.resblocks.3.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.3.ln_2.bias", "visual_encoder.transformer.resblocks.3.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.3.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.3.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.3.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.3.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.3.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.3.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.3.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.3.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.4.ln_1.weight", "visual_encoder.transformer.resblocks.4.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.4.ln_1.bias", "visual_encoder.transformer.resblocks.4.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.4.attn.in_proj_weight", "visual_encoder.transformer.resblocks.4.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.4.attn.in_proj_bias", "visual_encoder.transformer.resblocks.4.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.4.attn.out_proj.weight", "visual_encoder.transformer.resblocks.4.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.4.attn.out_proj.bias", "visual_encoder.transformer.resblocks.4.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.4.ln_2.weight", "visual_encoder.transformer.resblocks.4.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.4.ln_2.bias", "visual_encoder.transformer.resblocks.4.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.4.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.4.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.4.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.4.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.4.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.4.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.4.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.4.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.5.ln_1.weight", "visual_encoder.transformer.resblocks.5.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.5.ln_1.bias", "visual_encoder.transformer.resblocks.5.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.5.attn.in_proj_weight", "visual_encoder.transformer.resblocks.5.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.5.attn.in_proj_bias", "visual_encoder.transformer.resblocks.5.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.5.attn.out_proj.weight", "visual_encoder.transformer.resblocks.5.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.5.attn.out_proj.bias", "visual_encoder.transformer.resblocks.5.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.5.ln_2.weight", "visual_encoder.transformer.resblocks.5.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.5.ln_2.bias", "visual_encoder.transformer.resblocks.5.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.5.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.5.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.5.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.5.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.5.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.5.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.5.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.5.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.6.ln_1.weight", "visual_encoder.transformer.resblocks.6.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.6.ln_1.bias", "visual_encoder.transformer.resblocks.6.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.6.attn.in_proj_weight", "visual_encoder.transformer.resblocks.6.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.6.attn.in_proj_bias", "visual_encoder.transformer.resblocks.6.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.6.attn.out_proj.weight", "visual_encoder.transformer.resblocks.6.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.6.attn.out_proj.bias", "visual_encoder.transformer.resblocks.6.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.6.ln_2.weight", "visual_encoder.transformer.resblocks.6.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.6.ln_2.bias", "visual_encoder.transformer.resblocks.6.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.6.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.6.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.6.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.6.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.6.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.6.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.6.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.6.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.7.ln_1.weight", "visual_encoder.transformer.resblocks.7.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.7.ln_1.bias", "visual_encoder.transformer.resblocks.7.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.7.attn.in_proj_weight", "visual_encoder.transformer.resblocks.7.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.7.attn.in_proj_bias", "visual_encoder.transformer.resblocks.7.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.7.attn.out_proj.weight", "visual_encoder.transformer.resblocks.7.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.7.attn.out_proj.bias", "visual_encoder.transformer.resblocks.7.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.7.ln_2.weight", "visual_encoder.transformer.resblocks.7.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.7.ln_2.bias", "visual_encoder.transformer.resblocks.7.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.7.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.7.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.7.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.7.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.7.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.7.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.7.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.7.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.8.ln_1.weight", "visual_encoder.transformer.resblocks.8.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.8.ln_1.bias", "visual_encoder.transformer.resblocks.8.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.8.attn.in_proj_weight", "visual_encoder.transformer.resblocks.8.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.8.attn.in_proj_bias", "visual_encoder.transformer.resblocks.8.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.8.attn.out_proj.weight", "visual_encoder.transformer.resblocks.8.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.8.attn.out_proj.bias", "visual_encoder.transformer.resblocks.8.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.8.ln_2.weight", "visual_encoder.transformer.resblocks.8.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.8.ln_2.bias", "visual_encoder.transformer.resblocks.8.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.8.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.8.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.8.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.8.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.8.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.8.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.8.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.8.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.9.ln_1.weight", "visual_encoder.transformer.resblocks.9.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.9.ln_1.bias", "visual_encoder.transformer.resblocks.9.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.9.attn.in_proj_weight", "visual_encoder.transformer.resblocks.9.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.9.attn.in_proj_bias", "visual_encoder.transformer.resblocks.9.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.9.attn.out_proj.weight", "visual_encoder.transformer.resblocks.9.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.9.attn.out_proj.bias", "visual_encoder.transformer.resblocks.9.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.9.ln_2.weight", "visual_encoder.transformer.resblocks.9.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.9.ln_2.bias", "visual_encoder.transformer.resblocks.9.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.9.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.9.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.9.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.9.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.9.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.9.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.9.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.9.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.10.ln_1.weight", "visual_encoder.transformer.resblocks.10.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.10.ln_1.bias", "visual_encoder.transformer.resblocks.10.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.10.attn.in_proj_weight", "visual_encoder.transformer.resblocks.10.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.10.attn.in_proj_bias", "visual_encoder.transformer.resblocks.10.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.10.attn.out_proj.weight", "visual_encoder.transformer.resblocks.10.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.10.attn.out_proj.bias", "visual_encoder.transformer.resblocks.10.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.10.ln_2.weight", "visual_encoder.transformer.resblocks.10.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.10.ln_2.bias", "visual_encoder.transformer.resblocks.10.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.10.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.10.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.10.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.10.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.10.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.10.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.10.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.10.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.11.ln_1.weight", "visual_encoder.transformer.resblocks.11.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.11.ln_1.bias", "visual_encoder.transformer.resblocks.11.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.11.attn.in_proj_weight", "visual_encoder.transformer.resblocks.11.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.11.attn.in_proj_bias", "visual_encoder.transformer.resblocks.11.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.11.attn.out_proj.weight", "visual_encoder.transformer.resblocks.11.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.11.attn.out_proj.bias", "visual_encoder.transformer.resblocks.11.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.11.ln_2.weight", "visual_encoder.transformer.resblocks.11.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.11.ln_2.bias", "visual_encoder.transformer.resblocks.11.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.11.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.11.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.11.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.11.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.11.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.11.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.11.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.11.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.12.ln_1.weight", "visual_encoder.transformer.resblocks.12.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.12.ln_1.bias", "visual_encoder.transformer.resblocks.12.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.12.attn.in_proj_weight", "visual_encoder.transformer.resblocks.12.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.12.attn.in_proj_bias", "visual_encoder.transformer.resblocks.12.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.12.attn.out_proj.weight", "visual_encoder.transformer.resblocks.12.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.12.attn.out_proj.bias", "visual_encoder.transformer.resblocks.12.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.12.ln_2.weight", "visual_encoder.transformer.resblocks.12.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.12.ln_2.bias", "visual_encoder.transformer.resblocks.12.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.12.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.12.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.12.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.12.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.12.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.12.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.12.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.12.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.13.ln_1.weight", "visual_encoder.transformer.resblocks.13.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.13.ln_1.bias", "visual_encoder.transformer.resblocks.13.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.13.attn.in_proj_weight", "visual_encoder.transformer.resblocks.13.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.13.attn.in_proj_bias", "visual_encoder.transformer.resblocks.13.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.13.attn.out_proj.weight", "visual_encoder.transformer.resblocks.13.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.13.attn.out_proj.bias", "visual_encoder.transformer.resblocks.13.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.13.ln_2.weight", "visual_encoder.transformer.resblocks.13.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.13.ln_2.bias", "visual_encoder.transformer.resblocks.13.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.13.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.13.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.13.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.13.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.13.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.13.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.13.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.13.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.14.ln_1.weight", "visual_encoder.transformer.resblocks.14.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.14.ln_1.bias", "visual_encoder.transformer.resblocks.14.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.14.attn.in_proj_weight", "visual_encoder.transformer.resblocks.14.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.14.attn.in_proj_bias", "visual_encoder.transformer.resblocks.14.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.14.attn.out_proj.weight", "visual_encoder.transformer.resblocks.14.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.14.attn.out_proj.bias", "visual_encoder.transformer.resblocks.14.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.14.ln_2.weight", "visual_encoder.transformer.resblocks.14.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.14.ln_2.bias", "visual_encoder.transformer.resblocks.14.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.14.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.14.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.14.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.14.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.14.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.14.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.14.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.14.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.15.ln_1.weight", "visual_encoder.transformer.resblocks.15.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.15.ln_1.bias", "visual_encoder.transformer.resblocks.15.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.15.attn.in_proj_weight", "visual_encoder.transformer.resblocks.15.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.15.attn.in_proj_bias", "visual_encoder.transformer.resblocks.15.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.15.attn.out_proj.weight", "visual_encoder.transformer.resblocks.15.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.15.attn.out_proj.bias", "visual_encoder.transformer.resblocks.15.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.15.ln_2.weight", "visual_encoder.transformer.resblocks.15.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.15.ln_2.bias", "visual_encoder.transformer.resblocks.15.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.15.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.15.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.15.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.15.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.15.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.15.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.15.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.15.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.16.ln_1.weight", "visual_encoder.transformer.resblocks.16.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.16.ln_1.bias", "visual_encoder.transformer.resblocks.16.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.16.attn.in_proj_weight", "visual_encoder.transformer.resblocks.16.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.16.attn.in_proj_bias", "visual_encoder.transformer.resblocks.16.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.16.attn.out_proj.weight", "visual_encoder.transformer.resblocks.16.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.16.attn.out_proj.bias", "visual_encoder.transformer.resblocks.16.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.16.ln_2.weight", "visual_encoder.transformer.resblocks.16.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.16.ln_2.bias", "visual_encoder.transformer.resblocks.16.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.16.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.16.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.16.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.16.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.16.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.16.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.16.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.16.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.17.ln_1.weight", "visual_encoder.transformer.resblocks.17.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.17.ln_1.bias", "visual_encoder.transformer.resblocks.17.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.17.attn.in_proj_weight", "visual_encoder.transformer.resblocks.17.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.17.attn.in_proj_bias", "visual_encoder.transformer.resblocks.17.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.17.attn.out_proj.weight", "visual_encoder.transformer.resblocks.17.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.17.attn.out_proj.bias", "visual_encoder.transformer.resblocks.17.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.17.ln_2.weight", "visual_encoder.transformer.resblocks.17.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.17.ln_2.bias", "visual_encoder.transformer.resblocks.17.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.17.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.17.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.17.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.17.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.17.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.17.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.17.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.17.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.18.ln_1.weight", "visual_encoder.transformer.resblocks.18.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.18.ln_1.bias", "visual_encoder.transformer.resblocks.18.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.18.attn.in_proj_weight", "visual_encoder.transformer.resblocks.18.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.18.attn.in_proj_bias", "visual_encoder.transformer.resblocks.18.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.18.attn.out_proj.weight", "visual_encoder.transformer.resblocks.18.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.18.attn.out_proj.bias", "visual_encoder.transformer.resblocks.18.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.18.ln_2.weight", "visual_encoder.transformer.resblocks.18.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.18.ln_2.bias", "visual_encoder.transformer.resblocks.18.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.18.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.18.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.18.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.18.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.18.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.18.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.18.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.18.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.19.ln_1.weight", "visual_encoder.transformer.resblocks.19.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.19.ln_1.bias", "visual_encoder.transformer.resblocks.19.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.19.attn.in_proj_weight", "visual_encoder.transformer.resblocks.19.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.19.attn.in_proj_bias", "visual_encoder.transformer.resblocks.19.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.19.attn.out_proj.weight", "visual_encoder.transformer.resblocks.19.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.19.attn.out_proj.bias", "visual_encoder.transformer.resblocks.19.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.19.ln_2.weight", "visual_encoder.transformer.resblocks.19.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.19.ln_2.bias", "visual_encoder.transformer.resblocks.19.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.19.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.19.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.19.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.19.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.19.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.19.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.19.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.19.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.20.ln_1.weight", "visual_encoder.transformer.resblocks.20.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.20.ln_1.bias", "visual_encoder.transformer.resblocks.20.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.20.attn.in_proj_weight", "visual_encoder.transformer.resblocks.20.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.20.attn.in_proj_bias", "visual_encoder.transformer.resblocks.20.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.20.attn.out_proj.weight", "visual_encoder.transformer.resblocks.20.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.20.attn.out_proj.bias", "visual_encoder.transformer.resblocks.20.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.20.ln_2.weight", "visual_encoder.transformer.resblocks.20.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.20.ln_2.bias", "visual_encoder.transformer.resblocks.20.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.20.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.20.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.20.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.20.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.20.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.20.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.20.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.20.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.21.ln_1.weight", "visual_encoder.transformer.resblocks.21.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.21.ln_1.bias", "visual_encoder.transformer.resblocks.21.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.21.attn.in_proj_weight", "visual_encoder.transformer.resblocks.21.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.21.attn.in_proj_bias", "visual_encoder.transformer.resblocks.21.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.21.attn.out_proj.weight", "visual_encoder.transformer.resblocks.21.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.21.attn.out_proj.bias", "visual_encoder.transformer.resblocks.21.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.21.ln_2.weight", "visual_encoder.transformer.resblocks.21.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.21.ln_2.bias", "visual_encoder.transformer.resblocks.21.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.21.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.21.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.21.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.21.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.21.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.21.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.21.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.21.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.22.ln_1.weight", "visual_encoder.transformer.resblocks.22.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.22.ln_1.bias", "visual_encoder.transformer.resblocks.22.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.22.attn.in_proj_weight", "visual_encoder.transformer.resblocks.22.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.22.attn.in_proj_bias", "visual_encoder.transformer.resblocks.22.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.22.attn.out_proj.weight", "visual_encoder.transformer.resblocks.22.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.22.attn.out_proj.bias", "visual_encoder.transformer.resblocks.22.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.22.ln_2.weight", "visual_encoder.transformer.resblocks.22.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.22.ln_2.bias", "visual_encoder.transformer.resblocks.22.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.22.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.22.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.22.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.22.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.22.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.22.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.22.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.22.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.23.ln_1.weight", "visual_encoder.transformer.resblocks.23.ln_1.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.23.ln_1.bias", "visual_encoder.transformer.resblocks.23.ln_1.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.23.attn.in_proj_weight", "visual_encoder.transformer.resblocks.23.attn.in_proj_weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.23.attn.in_proj_bias", "visual_encoder.transformer.resblocks.23.attn.in_proj_bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.23.attn.out_proj.weight", "visual_encoder.transformer.resblocks.23.attn.out_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.23.attn.out_proj.bias", "visual_encoder.transformer.resblocks.23.attn.out_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.23.ln_2.weight", "visual_encoder.transformer.resblocks.23.ln_2.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.23.ln_2.bias", "visual_encoder.transformer.resblocks.23.ln_2.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.23.mlp.c_fc.weight", "visual_encoder.transformer.resblocks.23.mlp.c_fc.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.23.mlp.c_fc.bias", "visual_encoder.transformer.resblocks.23.mlp.c_fc.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.23.mlp.c_proj.weight", "visual_encoder.transformer.resblocks.23.mlp.c_proj.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.transformer.resblocks.23.mlp.c_proj.bias", "visual_encoder.transformer.resblocks.23.mlp.c_proj.bias")) 
        converters.append(WeightConverter("layers.0.visual_encoder.ln_post.weight", "visual_encoder.ln_post.weight")) 
        converters.append(WeightConverter("layers.0.visual_encoder.ln_post.bias", "visual_encoder.ln_post.bias")) 
        converters.append(WeightConverter("layers.0.ln_vision.weight", "ln_vision.weight")) 
        converters.append(WeightConverter("layers.0.ln_vision.bias", "ln_vision.bias")) 
        
        # Adapter
        converters.append(WeightConverter("layers.1.adapter_fc.weight", "c_fc.weight")) 
        converters.append(WeightConverter("layers.1.adapter_fc.bias", "c_fc.bias")) 

        # Embedding and output
        if self.config.tie_word_embeddings:
            converters.append(WeightConverter("layers.2.word_embeddings_weight", "model.embed_tokens.weight"))
            converters.append(IgnoreWeightConverter((), "lm_head.weight"))
        else:
            converters.append(WeightConverter("layers.2.word_embeddings_weight", "model.embed_tokens.weight"))
            converters.append(WeightConverter(f"layers.{num_layers + 3}.output_weights", "lm_head.weight"))

        # Final norm
        converters += self._get_weight_and_bias_converters(
            f"layers.{num_layers + 3}.final_norm", "model.norm", norm_bias
        )

        for i in range(num_layers):
            # Self-attn
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+3}.self_attn.query",
                f"model.layers.{i}.self_attn.q_proj",
                linear_bias,
                QueryWeightConverter,
            )
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+3}.self_attn.key_value",
                (f"model.layers.{i}.self_attn.k_proj", f"model.layers.{i}.self_attn.v_proj"),
                linear_bias,
                KeyValueWeightConverter,
            )
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+3}.self_attn.dense", f"model.layers.{i}.self_attn.o_proj", linear_bias
            )

            # Norm
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+3}.norm_1", f"model.layers.{i}.input_layernorm", norm_bias
            )
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+3}.norm_2", f"model.layers.{i}.post_attention_layernorm", norm_bias
            )

            # MLP
            converters += self._get_mlp_converters(f"layers.{i+3}", f"model.layers.{i}")

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
                self.config,
            )
        ]
        if use_bias:
            converters.append(
                cls(
                    tuple(f"{prefix}.bias" for prefix in fast_llm_prefix),
                    tuple(f"{prefix}.bias" for prefix in hf_prefix),
                    self.config,
                )
            )
        return converters


class Starcoder2HuggingfaceConverter(CommonHuggingfaceConverter):
    model_type = HuggingfaceModelType.starcoder2

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(None, "architectures", ["Starcoder2ForCausalLM"]),
            ConstantImportParamConverter(
                ("transformer", "normalization", "normalization_type"), None, NormalizationType.layer_norm
            ),
            ParamConverter(("transformer", "normalization", "layer_norm_eps"), "norm_epsilon"),
            ConstantImportParamConverter(("transformer", "gated"), None, False),
            ConstantImportParamConverter(("transformer", "add_linear_biases"), None, True),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str):
        linear_bias: bool = self.config.transformer.add_linear_biases
        return [
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_1", f"{hf_prefix}.mlp.c_fc", linear_bias
            ),
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_2", f"{hf_prefix}.mlp.c_proj", linear_bias, MLPLayer2Converter
            ),
        ]


class CommonLlamaHuggingfaceConverter(CommonHuggingfaceConverter, abc.ABC):
    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantImportParamConverter(
                ("transformer", "normalization", "normalization_type"), None, NormalizationType.rms_norm
            ),
            ParamConverter(("transformer", "normalization", "layer_norm_eps"), "rms_norm_eps"),
            ConstantImportParamConverter(("transformer", "gated"), None, True),
            ConstantImportParamConverter(("transformer", "add_linear_biases"), None, False),
        ]


class LlamaHuggingfaceConverter(CommonLlamaHuggingfaceConverter):
    model_type = HuggingfaceModelType.llama

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(None, "architectures", ["LlamaForCausalLM"]),
            # TODO: Llama supports biases
            ConstantExportParamConverter(None, "attention_bias", False),
            ConstantExportParamConverter(None, "mlp_bias", False),
            ConstantExportParamConverter(None, "rope_scaling", False),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str):
        linear_bias: bool = self.config.transformer.add_linear_biases
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


class MistralHuggingfaceConverter(CommonLlamaHuggingfaceConverter):
    model_type = HuggingfaceModelType.mistral

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(None, "architectures", ["MistralForCausalLM"]),
            IgnoreImportParamConverter(None, "sliding_window", None),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str):
        return [
            SplitWeightConverter(
                f"{fast_llm_prefix}.mlp.layer_1.weight",
                (f"{hf_prefix}.mlp.gate_proj.weight", f"{hf_prefix}.mlp.up_proj.weight"),
            ),
            MLPLayer2Converter(
                f"{fast_llm_prefix}.mlp.layer_2.weight", f"{hf_prefix}.mlp.down_proj.weight", self.config
            ),
        ]


class MixtralHuggingfaceConverter(CommonLlamaHuggingfaceConverter):
    model_type = HuggingfaceModelType.mixtral

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(None, "architectures", ["MixtralForCausalLM"]),
            ConstantImportParamConverter(("transformer", "expert_routing_type"), None, RoutingType.topk),
            ParamConverter(("transformer", "num_experts"), "num_local_experts"),
            ParamConverter(("transformer", "num_experts_per_token"), "num_experts_per_tok"),
            IgnoreImportParamConverter(None, "sliding_window", None),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str):
        num_experts = self.config.transformer.num_experts
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
                self.config,
            ),
        ]


class AutoStarDocConverter(AutoModelConverter, HuggingfaceModelConverter, abc.ABC):
    converter_map = {
        HuggingfaceModelType.starcoder2: Starcoder2HuggingfaceConverter,
        HuggingfaceModelType.llama: LlamaHuggingfaceConverter,
        HuggingfaceModelType.mistral: MistralHuggingfaceConverter,
        HuggingfaceModelType.mixtral: MixtralHuggingfaceConverter,
    }