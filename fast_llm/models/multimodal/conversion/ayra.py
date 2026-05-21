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
    - ``"encoder_projector.*"`` → audio projector / adapter
    - ``"llm.*"`` → language model

    Subclasses (e.g. Ultravox) override the three ``hf_*_prefix`` ClassVars
    plus ``architecture`` / ``format`` / ``audio_encoder_converter_class``.
    """

    _model_class: typing.ClassVar[FastLLMModelConfig] = MultiModalModelConfig
    format: typing.ClassVar[type[CheckpointFormat]] = AyraCheckpointFormat
    architecture: typing.ClassVar[str] = "AyraAudioModel"
    audio_encoder_converter_class: typing.ClassVar = AyraAudioEncoderConverter
    # Fallback text handler used when the checkpoint's ``text_config.model_type``
    # is missing or doesn't match any registered GPT handler's HF model_type.
    text_handler_name: typing.ClassVar[str] = "llama"
    # Cache of the most recent HF config loaded by ``_load_config``. ``__init__``
    # builds weight converters before the path/config is otherwise available, so
    # the cache bridges ``_load_config`` -> ``_create_weight_converters``. Set
    # sequentially per checkpoint load; not safe under concurrent loads.
    _last_loaded_hf_config: typing.ClassVar[dict | None] = None
    # HF state-dict prefixes for the three weight namespaces.
    hf_audio_encoder_prefix: typing.ClassVar[str] = "encoder"
    hf_audio_projector_prefix: typing.ClassVar[str] = "encoder_projector"
    hf_llm_prefix: typing.ClassVar[str] = "llm"

    @classmethod
    def get_transformers_configuration_class(cls):
        import transformers

        return transformers.AutoConfig

    @classmethod
    def _load_config(cls, directory) -> dict:
        import transformers

        config = transformers.AutoConfig.from_pretrained(directory, trust_remote_code=True).to_dict()
        cls._last_loaded_hf_config = config
        return config

    @classmethod
    def _resolve_text_handler_cls(cls, hf_model_type: str | None):
        """Pick the GPT handler whose ``get_huggingface_model_type()`` matches ``hf_model_type``.

        ``CheckpointFormat.name`` (handler_map key) and ``get_huggingface_model_type()`` can
        diverge (e.g. gemma4 vs gemma4_text), so direct map lookup is wrong in general.
        """
        from fast_llm.models.gpt.conversion.auto import AutoGPTHuggingfaceCheckpointHandler

        if hf_model_type is not None:
            for handler_cls in AutoGPTHuggingfaceCheckpointHandler.handler_map.values():
                if handler_cls.get_huggingface_model_type() == hf_model_type:
                    return handler_cls
        return AutoGPTHuggingfaceCheckpointHandler.get_handler_class(cls.text_handler_name)

    @classmethod
    def _import_config(cls, config: dict) -> "FastLLMModelConfig":
        audio_config = config.get("audio_config", {})
        text_config = config.get("text_config", {})

        text_handler_cls = cls._resolve_text_handler_cls(text_config.get("model_type"))
        gpt_model_config = text_handler_cls._import_config(text_config)
        base_model_dict = gpt_model_config.base_model.to_dict()

        base_model_dict["audio_encoder"] = cls.audio_encoder_converter_class.import_config(audio_config, config)
        return cls._model_class.from_dict({"base_model": base_model_dict})

    @classmethod
    def _export_config(cls, config: FastLLMModelConfig) -> dict[str, typing.Any]:
        # Export not fully implemented.
        return {}

    @classmethod
    def _save_config(cls, directory, config: dict) -> None:
        pass

    def _create_weight_converters(self) -> list[WeightConverter]:
        base_config = self._model.config.base_model
        audio_encoder_config = base_config.audio_encoder

        converters = self.audio_encoder_converter_class.get_converters(
            audio_encoder_config,
            fast_llm_prefix="audio_encoder",
            hf_encoder_prefix=self.hf_audio_encoder_prefix,
            hf_projector_prefix=self.hf_audio_projector_prefix,
        )

        cached_hf_config = self._last_loaded_hf_config or {}
        text_handler_cls = self._resolve_text_handler_cls(
            cached_hf_config.get("text_config", {}).get("model_type")
        )
        text_handler = text_handler_cls(self._model)
        for converter in text_handler._create_weight_converters():
            new_export_names = tuple(
                f"{self.hf_llm_prefix}.{name}" if name else name for name in converter.export_name
            )
            converter.export_name = new_export_names
            converters.append(converter)

        return converters

