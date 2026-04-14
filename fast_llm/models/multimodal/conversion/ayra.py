"""
Ayra audio-language model converter.

The Ayra checkpoint is a combined model with:
  - ``text_config`` (LLM portion, compatible with standard Llama-like handlers)
  - ``audio_config`` (Whisper-compatible encoder config)
  - Top-level projector/adapter fields

Weight namespaces:
  - ``"encoder.*"``  → audio_encoder.conv / audio_encoder.encoder
  - ``"encoder_projector.*"`` → audio_encoder.adapter
  - ``"llm.*"`` → embeddings / decoder / head

Config:
  - ``cfg_dict["audio_config"]`` → AudioEncoderConfig fields
  - ``cfg_dict["text_config"]``  → LLM-side fields
  - Top-level keys (adapter_size, activation_function, encoder_projector_ds_rate) →
        AudioEncoderConfig adapter fields
"""

import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import WeightConverter
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.models.multimodal.config import MultiModalBaseModelConfig, MultiModalModelConfig
from fast_llm.models.multimodal.conversion.config import AyraCheckpointFormat
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.models.multimodal.conversion.whisper import WhisperAudioEncoderConverter


class AyraAudioEncoderConverter(WhisperAudioEncoderConverter):
    """
    Extends the Whisper converter with Ayra-specific config (top-level projector fields).
    """

    @classmethod
    def import_config(cls, audio_config: dict, top_level_config: dict) -> dict:
        from fast_llm.functional.config import ActivationType

        base = super().import_config(audio_config)
        # Override adapter settings from top-level config
        base.update(
            {
                "adapter_size": top_level_config.get("adapter_size", base.get("adapter_size", 5120)),
                "adapter_activation_type": ActivationType.from_hf_name(
                    top_level_config.get("activation_function", "gelu")
                ),
                "aud_downsampling_k": top_level_config.get("encoder_projector_ds_rate", 5),
            }
        )
        return base


class AyraHuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    """
    Loads an Ayra audio-language model checkpoint.

    The checkpoint directory contains a ``config.json`` with three sections:
    - ``text_config`` (LLM, handled by the text handler registered under ``text_name``)
    - ``audio_config`` (Whisper encoder)
    - top-level adapter/projector parameters

    Weights are prefixed:
    - ``"encoder.*"`` → audio encoder
    - ``"llm.*"`` → language model
    """

    _model_class: typing.ClassVar[FastLLMModelConfig] = MultiModalModelConfig
    format: typing.ClassVar[type[CheckpointFormat]] = AyraCheckpointFormat
    architecture: typing.ClassVar[str] = "AyraAudioModel"
    audio_encoder_converter_class: typing.ClassVar = AyraAudioEncoderConverter
    # Name of the text handler registered in the auto registry (e.g. "llama" or "mistral")
    text_handler_name: typing.ClassVar[str] = "llama"

    @classmethod
    def get_transformers_configuration_class(cls):
        import transformers

        return transformers.AutoConfig

    @classmethod
    def _load_config(cls, directory) -> dict:
        import transformers

        return transformers.AutoConfig.from_pretrained(directory, trust_remote_code=True).to_dict()

    @classmethod
    def _import_config(cls, config: dict) -> "FastLLMModelConfig":
        from fast_llm.models.gpt.conversion.auto import AutoGPTHuggingfaceCheckpointHandler

        audio_config = config.get("audio_config", {})
        text_config = config.get("text_config", {})

        # Import LLM config from text handler.
        # In main's API, _import_config returns a FastLLMModelConfig, not a dict.
        text_handler_cls = AutoGPTHuggingfaceCheckpointHandler.get_handler_class(
            text_config.get("model_type", cls.text_handler_name)
        )
        gpt_model_config = text_handler_cls._import_config(text_config)
        base_model_dict = gpt_model_config.base_model.to_dict()

        # Import audio encoder config and merge into base_model dict
        base_model_dict["audio_encoder"] = cls.audio_encoder_converter_class.import_config(audio_config, config)
        return cls._model_class.from_dict({"base_model": base_model_dict})

    @classmethod
    def _export_config(cls, config: MultiModalBaseModelConfig) -> dict:
        # Export not fully implemented.
        return {}

    def _create_weight_converters(self) -> list[WeightConverter]:
        from fast_llm.models.gpt.conversion.auto import AutoGPTHuggingfaceCheckpointHandler
        from fast_llm.models.gpt.conversion.llama import LlamaBaseModelConverter

        base_config = self._model.config.base_model
        audio_encoder_config = base_config.audio_encoder

        # Audio encoder weights: HF prefix "encoder."
        converters = self.audio_encoder_converter_class.get_converters(
            audio_encoder_config,
            fast_llm_prefix="audio_encoder",
            hf_encoder_prefix="encoder",
            hf_projector_prefix="encoder_projector",
        )

        # LLM weights: HF prefix "llm."
        text_handler_cls = AutoGPTHuggingfaceCheckpointHandler.get_handler_class(self.text_handler_name)
        text_handler = text_handler_cls(self._model)
        exported_config = text_handler._exported_config
        for converter in text_handler._create_weight_converters():
            # Re-prefix HF names from "" to "llm."
            new_export_names = tuple(
                f"llm.{name}" if name else name for name in converter.export_name
            )
            converter.export_name = new_export_names
            converters.append(converter)

        return converters
