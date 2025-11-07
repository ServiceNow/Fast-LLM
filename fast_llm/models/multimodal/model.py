import logging
import typing

import torch

from fast_llm.data.sample.language_model import LanguageModelBatch
from fast_llm.engine.config_utils.tensor_dim import ConcatenatedTensorDim, TensorDim, scalar_dim
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.vision.vision_encoder import VisionMultiModalModel
from fast_llm.models.gpt.config import GPTBatchConfig
from fast_llm.models.gpt.model import GPTBaseModel, GPTModel
from fast_llm.models.multimodal.config import MultiModalBaseModelConfig, MultiModalBatchConfig, MultiModalModelConfig
from fast_llm.tensor import TensorMeta

logger = logging.getLogger(__name__)


class MultiModalBaseModel[ConfigType: MultiModalBaseModelConfig](
    VisionMultiModalModel[ConfigType], GPTBaseModel[ConfigType]
):
    """
    A transformer-based language model generalizing the GPT model architecture.
    """

    _config: ConfigType

    def preprocess_meta(
        self, batch_meta: GPTBatchConfig | torch.Tensor, phase: PhaseType
    ) -> list[tuple[TensorMeta, dict]]:
        preprocessed_meta = []
        for tokens, kwargs in super().preprocess_meta(batch_meta, phase):
            kwargs[LanguageModelKwargs.token_ids] = tokens
            image_patches = TensorMeta.from_dims(
                (
                    # We combine the batch and sequence dims to allow for variable sequence lengths.
                    # Gives the same result, assuming we disable cross-image attention (TODO: Enforce)
                    scalar_dim,
                    # TODO: Wrong (variable size).
                    ConcatenatedTensorDim("image_sequence", tokens.dims),
                    # TODO: Relate to tensor dims in patch convolution.
                    TensorDim("input_channels", self._config.vision_encoder.patch_convolution.input_channels),
                    TensorDim("patch_height", self._config.vision_encoder.patch_convolution.patch_height),
                    TensorDim("patch_width", self._config.vision_encoder.patch_convolution.patch_width),
                )
            )
            preprocessed_meta.append((image_patches, kwargs))

        return preprocessed_meta

    def preprocess_batch(
        self,
        batch: LanguageModelBatch,
        preprocessed_meta: list[tuple[TensorMeta, dict]] | None = None,
        *,
        phase: PhaseType,
        iteration: int,
        metrics: dict | None = None,
    ) -> list[tuple[torch.Tensor, dict]]:
        preprocessed = super().preprocess_batch(
            batch, preprocessed_meta, phase=phase, iteration=iteration, metrics=metrics
        )
        # TODO: Support micro-sequences.
        assert len(preprocessed) == 1, "Micro-sequences not supported for MultiModalModel."
        tokens, kwargs = preprocessed[0]

        kwargs[LanguageModelKwargs.token_ids] = tokens

        kwargs[LanguageModelKwargs.embedding_map] = batch.image_patches.token_map

        image_patches = batch.image_patches.patches
        sequence_length = image_patches.size(0)
        sequence_dim = TensorDim("image_sequence", sequence_length)

        hidden_dims = (
            (sequence_dim, scalar_dim, self.vision_encoder._hidden_dim)
            if (sequence_first := kwargs[LanguageModelKwargs.sequence_first])
            else (scalar_dim, sequence_dim, self.vision_encoder._hidden_dim)
        )
        kwargs[self._vision_encoder_namespace] = {
            LanguageModelKwargs.sequence_first: sequence_first,
            LanguageModelKwargs.position_ids: batch.image_patches.position_ids,
            LanguageModelKwargs.sequence_lengths: batch.image_patches.lengths,
            LanguageModelKwargs.sequence_length: sequence_length,
            LanguageModelKwargs.sequence_k_dim: sequence_dim,
            LanguageModelKwargs.sequence_q_dim: sequence_dim,
            LanguageModelKwargs.hidden_dims: hidden_dims,
        }
        super().preprocess(kwargs)

        return [(image_patches, kwargs)]

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        # Hack to delay preprocessing in super().preprocess_batch (TODO: Improve)
        pass


class MultiModalModel[ConfigType: MultiModalModelConfig](GPTModel[ConfigType]):
    # TODO: Can we drop class?
    pass


class MultiModalInferenceRunner(InferenceRunner):
    model_class: typing.ClassVar[type[MultiModalModel]] = MultiModalModel
    batch_config_class: typing.ClassVar[type[MultiModalBatchConfig]] = MultiModalBatchConfig
