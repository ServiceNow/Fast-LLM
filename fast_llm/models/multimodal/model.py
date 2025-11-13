import logging
import typing

import torch

from fast_llm.data.sample.language_model import LanguageModelBatch
from fast_llm.engine.config_utils.tensor_dim import TensorDim, scalar_dim
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.vision.config import VisionKwargs
from fast_llm.layers.vision.vision_encoder import VisionMultiModalModel
from fast_llm.models.gpt.config import GPTBatchConfig
from fast_llm.models.gpt.model import GPTBaseModel, GPTModel
from fast_llm.models.multimodal.config import MultiModalBaseModelConfig, MultiModalBatchConfig, MultiModalModelConfig
from fast_llm.tensor import TensorMeta

logger = logging.getLogger(__name__)


class MultiModalBaseModel[ConfigType: MultiModalBaseModelConfig](
    GPTBaseModel[ConfigType], VisionMultiModalModel[ConfigType]
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
            kwargs[LanguageModelKwargs.mask_inputs] = True
            image_patches = TensorMeta.from_dims(
                (
                    # We combine the batch and sequence dims to allow for variable sequence lengths.
                    # Gives the same result, assuming we disable cross-image attention (TODO: Enforce)
                    sequence_dim := TensorDim("image_sequence", tokens.numel(), variable_size=True),
                    # TODO: Relate to tensor dims in patch convolution.
                    TensorDim("input_channels", self._config.vision_encoder.patch_convolution.input_channels),
                    TensorDim("patch_height", self._config.vision_encoder.patch_convolution.patch_height),
                    TensorDim("patch_width", self._config.vision_encoder.patch_convolution.patch_width),
                )
            )

            hidden_dims = (
                (sequence_dim, scalar_dim, self.vision_encoder._hidden_dim)
                if (sequence_first := kwargs[LanguageModelKwargs.sequence_first])
                else (scalar_dim, sequence_dim, self.vision_encoder._hidden_dim)
            )
            kwargs[self._vision_encoder_namespace] = {
                VisionKwargs.sequence_first: sequence_first,
                VisionKwargs.sequence_k_dim: sequence_dim,
                VisionKwargs.sequence_q_dim: sequence_dim,
                VisionKwargs.hidden_dims: hidden_dims,
            }

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

        # If document cropping is enabled, extra tokens may belong to images and need to be removed.
        # TODO: Handle earlier.
        tokens_end = kwargs[AttentionKwargs.sequence_k_dim].size
        tokens_begin = tokens_end - kwargs[AttentionKwargs.sequence_q_dim].size
        cropped_image_patches = batch.image_patches.crop(tokens_begin, tokens_end)

        sequence_length = tokens.shape[:2].numel()
        sequence_dim = TensorDim("image_sequence", sequence_length)
        pad_size = sequence_length - cropped_image_patches.patches.size(0)

        patches = cropped_image_patches.patches.to(self._distributed.config.compute_dtype.torch)
        patches = torch.cat([patches, patches.new_zeros((pad_size,) + patches.shape[1:])])

        positions = torch.cat(
            [
                cropped_image_patches.positions,
                cropped_image_patches.positions.new_zeros((pad_size,) + cropped_image_patches.positions.shape[1:]),
            ]
        )

        hidden_dims = (
            (sequence_dim, scalar_dim, self.vision_encoder._hidden_dim)
            if (sequence_first := kwargs[LanguageModelKwargs.sequence_first])
            else (scalar_dim, sequence_dim, self.vision_encoder._hidden_dim)
        )
        kwargs[self._vision_encoder_namespace] = {
            VisionKwargs.sequence_first: sequence_first,
            VisionKwargs.patch_positions: positions,
            VisionKwargs.sequence_lengths: [cropped_image_patches.lengths],
            VisionKwargs.sequence_length: sequence_length,
            VisionKwargs.sequence_k_dim: sequence_dim,
            VisionKwargs.sequence_q_dim: sequence_dim,
            VisionKwargs.hidden_dims: hidden_dims,
            VisionKwargs.device: self._distributed.device,
        }

        kwargs[LanguageModelKwargs.embedding_map] = (
            (cropped_image_patches.token_map, cropped_image_patches.sample_map)
            if kwargs[LanguageModelKwargs.sequence_first]
            else (cropped_image_patches.sample_map, cropped_image_patches.token_map)
        )

        super().preprocess(kwargs)

        return [(patches, kwargs)]

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        # Hack to delay preprocessing in super().preprocess_batch (TODO: Improve)
        pass


class MultiModalModel[ConfigType: MultiModalModelConfig](GPTModel[ConfigType]):
    # TODO: Can we drop class?
    pass


class MultiModalInferenceRunner(InferenceRunner):
    model_class: typing.ClassVar[type[MultiModalModel]] = MultiModalModel
    batch_config_class: typing.ClassVar[type[MultiModalBatchConfig]] = MultiModalBatchConfig
