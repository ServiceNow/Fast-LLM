import logging
import typing

import torch

from fast_llm.data.data.gpt.data import GPTBatch
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.models.gpt.config import GPTBatchConfig
from fast_llm.models.gpt.model import GPTBaseModel, GPTModel
from fast_llm.models.multimodal.config import MultiModalBaseModelConfig, MultiModalBatchConfig, MultiModalModelConfig
from fast_llm.tensor import TensorMeta

logger = logging.getLogger(__name__)


class MultiModalBaseModel[ConfigType: MultiModalBaseModelConfig](GPTBaseModel[ConfigType]):
    """
    A transformer-based language model generalizing the GPT model architecture.
    """

    _config: ConfigType

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
    ):
        super().__init__(config, distributed_config)
        self.vision_encoder = self._config.vision_encoder.get_layer(
            distributed_config,
            self._hidden_dim,
            lr_scale=None,
            peft=self._config.peft,
        )

    def preprocess_meta(
        self, batch_meta: GPTBatchConfig | torch.Tensor, phase: PhaseType
    ) -> list[tuple[TensorMeta, dict]]:
        # TODO Remove (Move batch splitting elsewhere)
        # TODO: Use parallel/sequential dims, distinguish micro and full batch/sequence
        # TODO ====== Vision ======
        # if self._config.vision_encoder.enabled:
        #    try:
        #        max_image_size = batch_meta.max_image_size
        #    except AttributeError:
        #        max_image_size = 256
        #        logger.warning("Inference mode: max_image_size not provided, defaulting to 256")
        #    vision_kwargs = {
        #        VisionEncoderKwargs.patch_size: self._config.vision_encoder.patch_size,
        #        VisionEncoderKwargs.max_image_size: max_image_size,
        #        VisionEncoderKwargs.rope_theta: self._config.vision_encoder.transformer.rotary.theta,
        #        VisionEncoderKwargs.kv_channels: self._tensor_space[VisionTransformerDimNames.kv_channels].size,
        #        VisionEncoderKwargs.out_channels: self._tensor_space[VisionEncoderDimNames.out_channels].size,
        #    }
        #    vision_hidden_dim = self._tensor_space[VisionTransformerDimNames.hidden]
        #    vision_hidden_dims = (
        #        (hidden_sequence_q_dim, batch_dim, vision_hidden_dim)
        #        if sequence_first
        #        else (batch_dim, hidden_sequence_q_dim, vision_hidden_dim)
        #    )
        #    vision_kwargs.update(
        #        {
        #            VisionTransformerKwargs.hidden_dims: vision_hidden_dims,
        #        }
        #    )
        #    common_kwargs.update(vision_kwargs)

        # TODO ====== Vision ======
        # if self._config.vision_encoder.enabled:
        #     # patch_dimensions are (batch * sequence_length) x 3 x patch_size x patch_size
        #     preprocessed_meta.append((kwargs[VisionEncoderKwargs.image_patches_meta], kwargs))
        # else:
        #     preprocessed_meta.append((tokens, kwargs))
        pass

    def preprocess_batch(
        self,
        batch: GPTBatch,
        preprocessed_meta: list[tuple[TensorMeta, dict]] | None = None,
        *,
        phase: PhaseType,
        iteration: int,
        metrics: dict | None = None,
    ) -> list[tuple[torch.Tensor, dict]]:
        # TODO Move batch splitting elsewhere, align interface with LayerBase
        # TODO ====== Vision ======
        # if self._config.vision_encoder.enabled:
        #    if self._config.vision_encoder.image_break_token is not None:
        #        if not labels_cloned:
        #            labels = labels.clone()
        #            labels_cloned = True
        #        labels = torch.where(labels == self._config.vision_encoder.image_break_token, -100, labels)
        #    if self._config.vision_encoder.image_end_token is not None:
        #        if not labels_cloned:
        #            labels = labels.clone()
        #            labels_cloned = True
        #        labels = torch.where(labels == self._config.vision_encoder.image_end_token, -100, labels)
        # Loss-masking for distillation losses
        # TODO ====== Vision ======
        # if self._config.vision_encoder.enabled:
        #    batch_images = (
        #        batch.images if batch.images is not None else [[]] * kwargs[AttentionKwargs.micro_batch_size]
        #    )
        #    kwargs[VisionEncoderKwargs.images] = [
        #        [
        #            img.to(device=self._tensor_space.distributed.device, dtype=torch.uint8, non_blocking=True)
        #            for img in images
        #        ]
        #        for images in batch_images
        #    ]
        #    kwargs[VisionEncoderKwargs.image_positions] = (
        #        batch.image_positions
        #        if batch.image_positions is not None
        #        else [[]] * kwargs[AttentionKwargs.micro_batch_size]
        #    )
        #    kwargs[LanguageModelKwargs.tokens] = tokens
        # image_patches = kwargs.get(VisionEncoderKwargs.image_patches, None)
        # if image_patches is not None:
        #     preprocessed.append((image_patches, kwargs))
        # else:
        #     preprocessed.append((tokens, kwargs))
        pass


class MultiModalModel[ConfigType: MultiModalModelConfig](GPTModel[ConfigType]):
    # TODO: Can we drop class?
    pass


class MultiModalInferenceRunner(InferenceRunner):
    model_class: typing.ClassVar[type[MultiModalModel]] = MultiModalModel
    batch_config_class: typing.ClassVar[type[MultiModalBatchConfig]] = MultiModalBatchConfig
