"""
AudioMultiModalModel mixin — the audio-encoder counterpart of VisionMultiModalModel.

Inheriting from both ``LanguageModel`` and this mixin (via ``MultiModalBaseModel``) adds
an audio encoder stack whose output embeddings are injected into the language model's
token sequence at the positions specified by ``AudioKwargs.audio_positions``.
"""

import functools
import logging
import typing

import torch

from fast_llm.engine.base_model.base_model import Layer, LayerBaseWithNamespace
from fast_llm.engine.base_model.config import LossDef
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.audio_encoder.config import AudioKwargs, AudioMultiModalModelConfig
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.language_model.language_model import LanguageModel
from fast_llm.utils import safe_merge_dicts

logger = logging.getLogger(__name__)


class AudioMultiModalModel[ConfigType: AudioMultiModalModelConfig](LanguageModel[ConfigType]):
    """
    Mixin that attaches an ``AudioEncoder`` to a language model.

    Usage (multiple-inheritance with GPTBaseModel and VisionMultiModalModel):
    ::

        class MultiModalBaseModel(GPTBaseModel, VisionMultiModalModel, AudioMultiModalModel):
            pass
    """

    _config: ConfigType

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        hidden_dim: TensorDim | None = None,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ):
        super().__init__(
            config,
            distributed_config,
            hidden_dim=TensorDim("hidden", config.hidden_size),
            lr_scale=lr_scale,
            peft=peft,
        )
        if self._config.audio_encoder.enabled:
            self.audio_encoder = self._config.audio_encoder.get_layer(
                distributed_config,
                hidden_dim=self._hidden_dim,
                lr_scale=self._lr_scale,
                peft=self._peft,
            )
        else:
            self.audio_encoder = None

    def get_layers(self) -> list[Layer]:
        if self.audio_encoder is None:
            return super().get_layers()
        return self._audio_encoder_with_namespace.get_layers() + super().get_layers()

    def get_preprocessing_config(self) -> dict[str, typing.Any]:
        base = super().get_preprocessing_config()
        if self.audio_encoder is None:
            return base
        return safe_merge_dicts(
            {
                "audio_encoder": safe_merge_dicts(
                    self._audio_encoder_with_namespace.get_preprocessing_config(),
                    {"distributed": self._distributed_config, "namespace": self._audio_encoder_namespace},
                )
            },
            base,
        )

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        if self.audio_encoder is not None:
            self._audio_encoder_with_namespace.preprocess(kwargs)

            # Build embedding_map so LanguageModelEmbedding injects audio tokens at the
            # correct LM sequence positions.  audio_positions (set by AudioBatch.to_kwargs)
            # lives in the top-level kwargs; audio_token_lens was computed by
            # AudioPreprocessor.preprocess() and stored in the audio encoder's sub-namespace.
            audio_positions = kwargs.get(AudioKwargs.audio_positions)
            if audio_positions is not None and audio_positions.numel() > 0:
                namespace_kwargs = kwargs.get(self._audio_encoder_namespace, {})
                audio_token_lens = namespace_kwargs.get(AudioKwargs.audio_token_lens)
                if audio_token_lens is not None:
                    embedding_map = torch.cat(
                        [
                            torch.arange(
                                int(pos.item()),
                                int(pos.item()) + int(n.item()),
                                dtype=torch.long,
                                device=audio_positions.device,
                            )
                            for pos, n in zip(audio_positions, audio_token_lens)
                        ]
                    )
                    kwargs[LanguageModelKwargs.embedding_map] = embedding_map

        super().preprocess(kwargs)

    def get_loss_definitions(self) -> list[LossDef]:
        audio_losses = self.audio_encoder.get_loss_definitions() if self.audio_encoder is not None else []
        return audio_losses + super().get_loss_definitions()

    @functools.cached_property
    def _audio_encoder_namespace(self) -> str:
        return self.audio_encoder.module_name

    @functools.cached_property
    def _audio_encoder_with_namespace(self) -> LayerBaseWithNamespace:
        return LayerBaseWithNamespace(self.audio_encoder, self._audio_encoder_namespace)
