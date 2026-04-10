"""
Whisper audio encoder converter for multimodal models.

Maps weights from a HuggingFace Whisper encoder checkpoint to the
``audio_encoder`` portion of a Fast-LLM multimodal model.

Fast-LLM ↔ HuggingFace weight-name mapping:
  audio_encoder.conv.conv1_weight                          ← encoder.conv1.weight
  audio_encoder.conv.conv2_weight                          ← encoder.conv2.weight
  audio_encoder.conv.conv1_bias                            ← encoder.conv1.bias
  audio_encoder.conv.conv2_bias                            ← encoder.conv2.bias
  audio_encoder.conv.positional_embeddings                 ← encoder.embed_positions.weight
  audio_encoder.encoder.{i}.norm_1.{weight,bias}          ← encoder.layers.{i}.self_attn_layer_norm.*
  audio_encoder.encoder.{i}.norm_2.{weight,bias}          ← encoder.layers.{i}.final_layer_norm.*
  audio_encoder.encoder.{i}.mixer.query.{weight,bias}     ← encoder.layers.{i}.self_attn.q_proj.*
  audio_encoder.encoder.{i}.mixer.key_value.{weight,bias} ← encoder.layers.{i}.self_attn.{k,v}_proj.*
  audio_encoder.encoder.{i}.mixer.dense.{weight,bias}     ← encoder.layers.{i}.self_attn.out_proj.*
  audio_encoder.encoder.{i}.mlp.layer_1.{weight,bias}     ← encoder.layers.{i}.fc1.*
  audio_encoder.encoder.{i}.mlp.layer_2.{weight,bias}     ← encoder.layers.{i}.fc2.*
  audio_encoder.adapter.norm_1.{weight,bias}              ← encoder.layer_norm.*
  audio_encoder.adapter.norm_2.{weight,bias}              ← encoder_projector.layer_norm.*
  audio_encoder.adapter.layer_1.{weight,bias}             ← encoder_projector.linear1.*
  audio_encoder.adapter.layer_2.{weight,bias}             ← encoder_projector.linear2.*
"""

import typing

import torch

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import WeightConverter
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.layers.audio_encoder.config import AudioEncoderConfig, AudioEncoderType
from fast_llm.layers.block.config import FixedBlockSequenceConfig
from fast_llm.layers.common.normalization.config import LayerNormalizationConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.models.gpt.conversion.llama import (
    KeyValueWeightConverter,
    MLPLayer2Converter,
    get_weight_and_bias_converters,
)
from fast_llm.models.multimodal.config import MultiModalBaseModelConfig, MultiModalModelConfig
from fast_llm.models.multimodal.conversion.config import WhisperCheckpointFormat
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.utils import Assert


class WhisperLayerNormConverter:
    """Converts LayerNorm config from/to Whisper HF format."""

    @classmethod
    def import_config(cls, config: dict) -> dict:
        # Whisper config.json doesn't expose layer_norm_eps; default is 1e-5.
        return {"type": "layer_norm", "epsilon": config.get("layer_norm_eps", 1e-5)}

    @classmethod
    def export_config(cls, config: LayerNormalizationConfig) -> dict:
        Assert.custom(isinstance, config, LayerNormalizationConfig)
        return {"layer_norm_eps": config.epsilon}

    @classmethod
    def get_converters(
        cls,
        config: LayerNormalizationConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
    ) -> list[WeightConverter]:
        # LayerNorm has both weight and bias.
        return get_weight_and_bias_converters(fast_llm_prefix, hf_prefix, True)


class WhisperMLPConverter:
    """Non-gated MLP converter for Whisper (fc1 → layer_1, fc2 → layer_2)."""

    @classmethod
    def import_config(cls, config: dict) -> dict:
        from fast_llm.functional.config import ActivationType

        return {
            "intermediate_size": config["encoder_ffn_dim"],
            "add_linear_biases": True,
            "gated": False,
            "activation": ActivationType.from_hf_name(config.get("activation_function", "gelu")),
        }

    @classmethod
    def get_converters(
        cls,
        config,
        fast_llm_prefix: str,
        hf_prefix: str,
    ) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_1",
                f"{hf_prefix}.fc1",
                True,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_2",
                f"{hf_prefix}.fc2",
                True,
                MLPLayer2Converter,
            ),
        ]


class WhisperAttentionConverter:
    """
    Attention converter for Whisper encoder (no RoPE, biases enabled, causal=False,
    dense named 'out_proj' instead of 'o_proj').
    """

    @classmethod
    def import_config(cls, config: dict) -> dict:
        n_heads = config["encoder_attention_heads"]
        d_model = config["d_model"]
        return {
            "type": "attention",
            "heads": n_heads,
            "head_groups": n_heads,
            "head_size": d_model // n_heads,
            "add_linear_biases": True,
            "causal": False,
            "rotary": {"type": "none"},
        }

    @classmethod
    def get_converters(
        cls,
        config,
        fast_llm_prefix: str,
        hf_prefix: str,
    ) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.query",
                f"{hf_prefix}.q_proj",
                True,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.key_value",
                (f"{hf_prefix}.k_proj", f"{hf_prefix}.v_proj"),
                True,
                KeyValueWeightConverter,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.dense",
                f"{hf_prefix}.out_proj",
                True,
            ),
        ]


class WhisperBlockConverter:
    mixer_converter_class: typing.ClassVar = WhisperAttentionConverter
    mlp_converter_class: typing.ClassVar = WhisperMLPConverter
    normalization_converter_class: typing.ClassVar = WhisperLayerNormConverter
    hf_norm_1_name: typing.ClassVar[str] = "self_attn_layer_norm"
    hf_norm_2_name: typing.ClassVar[str] = "final_layer_norm"
    hf_mixer_name: typing.ClassVar[str] = "self_attn"
    hf_mlp_name: typing.ClassVar[str] = ""  # MLP converters use full hf_prefix

    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            "mixer": cls.mixer_converter_class.import_config(config),
            "mlp": cls.mlp_converter_class.import_config(config),
            "normalization": cls.normalization_converter_class.import_config(config),
        }

    @classmethod
    def get_converters(
        cls,
        config: DecoderBlockConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
    ) -> list[WeightConverter]:
        return [
            *cls.mixer_converter_class.get_converters(
                config.mixer,
                f"{fast_llm_prefix}.mixer",
                f"{hf_prefix}.{cls.hf_mixer_name}",
            ),
            *cls.mlp_converter_class.get_converters(
                config.mlp,
                f"{fast_llm_prefix}.mlp",
                hf_prefix,
            ),
            *cls.normalization_converter_class.get_converters(
                config.norm_1,
                f"{fast_llm_prefix}.norm_1",
                f"{hf_prefix}.{cls.hf_norm_1_name}",
            ),
            *cls.normalization_converter_class.get_converters(
                config.norm_2,
                f"{fast_llm_prefix}.norm_2",
                f"{hf_prefix}.{cls.hf_norm_2_name}",
            ),
        ]


class WhisperEncoderConverter:
    block_converter_class: typing.ClassVar = WhisperBlockConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            "num_blocks": config["num_hidden_layers"],
            "block": cls.block_converter_class.import_config(config),
        }

    @classmethod
    def get_converters(
        cls,
        config: FixedBlockSequenceConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
    ) -> list[WeightConverter]:
        converters = []
        block_config = config.block
        for i in range(config.num_blocks):
            converters += cls.block_converter_class.get_converters(
                block_config,
                f"{fast_llm_prefix}.{i}",
                f"{hf_prefix}.{i}",
            )
        return converters


class WhisperAudioConvConverter:
    @classmethod
    def get_converters(
        cls,
        config: AudioEncoderConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
    ) -> list[WeightConverter]:
        converters = [
            WeightConverter(f"{fast_llm_prefix}.conv1_weight", f"{hf_prefix}.conv1.weight"),
            WeightConverter(f"{fast_llm_prefix}.conv2_weight", f"{hf_prefix}.conv2.weight"),
        ]
        if config.conv_bias:
            converters += [
                WeightConverter(f"{fast_llm_prefix}.conv1_bias", f"{hf_prefix}.conv1.bias"),
                WeightConverter(f"{fast_llm_prefix}.conv2_bias", f"{hf_prefix}.conv2.bias"),
            ]
        converters.append(
            WeightConverter(f"{fast_llm_prefix}.positional_embeddings", f"{hf_prefix}.embed_positions.weight")
        )
        return converters


class WhisperAudioAdapterConverter:
    @classmethod
    def get_converters(
        cls,
        config: AudioEncoderConfig,
        fast_llm_prefix: str,
        hf_encoder_prefix: str,
        hf_projector_prefix: str,
    ) -> list[WeightConverter]:
        converters = [
            # norm_1 ← Whisper final encoder layer_norm
            WeightConverter(f"{fast_llm_prefix}.norm_1.weight", f"{hf_encoder_prefix}.layer_norm.weight"),
            WeightConverter(f"{fast_llm_prefix}.norm_1.bias", f"{hf_encoder_prefix}.layer_norm.bias"),
            # norm_2 ← Ayra encoder_projector layer_norm
            WeightConverter(f"{fast_llm_prefix}.norm_2.weight", f"{hf_projector_prefix}.layer_norm.weight"),
            WeightConverter(f"{fast_llm_prefix}.norm_2.bias", f"{hf_projector_prefix}.layer_norm.bias"),
        ]
        converters += get_weight_and_bias_converters(
            f"{fast_llm_prefix}.layer_1", f"{hf_projector_prefix}.linear1", config.adapter_bias
        )
        converters += get_weight_and_bias_converters(
            f"{fast_llm_prefix}.layer_2", f"{hf_projector_prefix}.linear2", config.adapter_bias, MLPLayer2Converter
        )
        return converters


class WhisperAudioEncoderConverter:
    """
    Top-level converter for the full audio encoder stack (conv + transformer + adapter).
    """

    conv_converter_class: typing.ClassVar = WhisperAudioConvConverter
    encoder_converter_class: typing.ClassVar = WhisperEncoderConverter
    adapter_converter_class: typing.ClassVar = WhisperAudioAdapterConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            "encoder_type": AudioEncoderType.whisper,
            "hidden_size": config["d_model"],
            "num_mel_bins": config.get("num_mel_bins", 80),
            "normalization": WhisperLayerNormConverter.import_config(config),
            "encoder": WhisperEncoderConverter.import_config(config),
        }

    @classmethod
    def get_converters(
        cls,
        config: AudioEncoderConfig,
        fast_llm_prefix: str = "audio_encoder",
        hf_encoder_prefix: str = "encoder",
        hf_projector_prefix: str = "encoder_projector",
    ) -> list[WeightConverter]:
        return [
            *cls.conv_converter_class.get_converters(
                config,
                f"{fast_llm_prefix}.conv",
                hf_encoder_prefix,
            ),
            *cls.encoder_converter_class.get_converters(
                config.encoder,
                f"{fast_llm_prefix}.encoder",
                f"{hf_encoder_prefix}.layers",
            ),
            *cls.adapter_converter_class.get_converters(
                config,
                f"{fast_llm_prefix}.adapter",
                hf_encoder_prefix,
                hf_projector_prefix,
            ),
        ]


class WhisperHuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    """
    Loads a standalone Whisper encoder checkpoint into the audio_encoder portion
    of a Fast-LLM multimodal model.
    """

    _model_class: typing.ClassVar[FastLLMModelConfig] = MultiModalModelConfig
    format: typing.ClassVar[type[CheckpointFormat]] = WhisperCheckpointFormat
    architecture: typing.ClassVar[str] = "WhisperForConditionalGeneration"
    audio_encoder_converter_class: typing.ClassVar = WhisperAudioEncoderConverter

    @classmethod
    def get_transformers_configuration_class(cls):
        import transformers

        return transformers.WhisperConfig

    @classmethod
    def _import_config(cls, config: dict) -> dict:
        return {
            "audio_encoder": cls.audio_encoder_converter_class.import_config(config),
        }

    @classmethod
    def _export_config(cls, config: MultiModalBaseModelConfig) -> dict:
        # Export is not fully supported; only audio encoder config is exported.
        return {}

    def _create_weight_converters(self) -> list[WeightConverter]:
        audio_encoder_config = self._model.config.base_model.audio_encoder
        return self.audio_encoder_converter_class.get_converters(audio_encoder_config)
