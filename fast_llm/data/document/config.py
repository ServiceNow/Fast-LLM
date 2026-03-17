import functools
import logging
import typing

from fast_llm.config import Config, Field, config_class
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType

if typing.TYPE_CHECKING:
    import torch

    from fast_llm.data.document.language_model import LanguageModelBatch, LanguageModelInput
    from fast_llm.data.document.patch import PatchBatch

logger = logging.getLogger(__name__)


@config_class()
class BatchPreprocessingConfig(Config):
    pass


@config_class()
class LengthPreprocessingConfig(BatchPreprocessingConfig):
    causal: bool = Field(default=True)
    distributed: DistributedConfig = Field()
    return_cumulative_sequence_lengths: bool = Field(default=False)
    return_max_sequence_lengths: bool = Field(default=False)
    return_document_index: bool = Field(default=False)
    return_position_index: bool = Field(default=False)


@config_class()
class ImageNormalizationConfig(Config):
    scale: float = Field(default=255.0)
    # Default values from OpenAI Clip.
    mean: tuple[float, float, float] = Field(default=(0.48145466, 0.4578275, 0.40821073))
    std: tuple[float, float, float] = Field(default=(0.26862954, 0.26130258, 0.27577711))

    def normalize(self, image: "torch.Tensor") -> "torch.Tensor":
        import torchvision.transforms.v2 as torchvision_transforms

        return torchvision_transforms.functional.normalize(image / self.scale, list(self.mean), list(self.std))


@config_class()
class PatchPreprocessingConfig(LengthPreprocessingConfig):
    normalization: ImageNormalizationConfig | None = Field(default=None)
    shape: tuple[int, ...] = Field(default=(3, 16, 16))
    namespace: str = Field(default="vision")

    def get_batch_meta(self, size: int = 1) -> "PatchBatch":
        import torch

        from fast_llm.data.document.patch import PatchBatch

        return PatchBatch(
            patches=torch.empty(size, *self.shape, dtype=torch.uint8, device="meta"),
            token_map=torch.empty(size, *self.shape, dtype=torch.int32, device="meta"),
            positions=torch.empty(size, len(self.shape) - 1, dtype=torch.int32, device="meta"),
            lengths=[size],
        )


@config_class()
class LanguageModelBatchPreprocessingConfig(LengthPreprocessingConfig):
    _abstract = False
    phase: PhaseType = Field(default=PhaseType.training)
    micro_batch_splits: int = Field(default=1)
    predicted_tokens: int = Field(default=1)
    return_prediction_mask: bool = Field(default=False)
    vision_encoder: PatchPreprocessingConfig | None = Field(default=None)
    vocab_size: int | None = Field(default=None)
    use_loss_masking_spans: bool = Field(default=True)
    use_preference_spans: bool = Field(default=False)

    def _validate(self) -> None:
        super()._validate()
        # TODO: Implement?
        assert not self.use_preference_spans

    def get_input_meta(self, size: int = 1) -> "list[LanguageModelInput]":
        return self.get_batch_meta(size).get_model_inputs(self)

    def get_batch_meta(self, size: int = 1) -> "LanguageModelBatch":
        import torch

        from fast_llm.data.document.language_model import LanguageModelBatch

        total_size = size + self.num_labels

        batch = LanguageModelBatch(
            tokens=torch.empty(total_size, dtype=torch.int64, device="meta"), lengths=[total_size]
        )
        if self.vision_encoder is not None:
            batch.image_patches = self.vision_encoder.get_batch_meta(total_size)
        return batch

    @functools.cached_property
    def num_labels(self) -> int:
        return 0 if self.phase == PhaseType.inference else self.predicted_tokens

    @functools.cached_property
    def use_image_patches(self) -> bool:
        return self.vision_encoder is not None
