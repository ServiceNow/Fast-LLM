import dataclasses
import functools
import logging
import typing

from fast_llm.config import Field, config_class
from fast_llm.data.document.abstract import Document
from fast_llm.data.preprocessing.abstract import NullPreprocessingConfig, PreprocessingConfig
from fast_llm.data.preprocessing.image_patch import ImagePatchConfig
from fast_llm.data.preprocessing.language_model import LanguageModelPreprocessingConfig
from fast_llm.data.preprocessing.tokenizer import TokenizerConfig
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.models.gpt.config import GPTBatchConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@config_class()
class BatchPreprocessingConfig(PreprocessingConfig):
    pass


@config_class()
class LanguageModelBatchPreprocessingConfig(LanguageModelPreprocessingConfig):
    _abstract = False
    # TODO: Duplicate `use_loss_masking_spans`, `use_preference_spans`
    batch: GPTBatchConfig = Field()
    phase: PhaseType = Field(default=PhaseType.inference)
    predicted_tokens: int = Field(default=1)
    return_cumulative_sequence_lengths: bool = Field(default=False)
    return_max_sequence_lengths: bool = Field(default=False)
    return_document_index: bool = Field(default=False)
    return_position_index: bool = Field(default=False)
    return_prediction_mask: bool = Field(default=False)

    def _validate(self) -> None:
        super()._validate()
        Assert.custom(isinstance, self.image_patches, (ImagePatchConfig, NullPreprocessingConfig))
        Assert.custom(isinstance, self.tokenizer, (TokenizerConfig, NullPreprocessingConfig))

    @functools.cached_property
    def use_image_patches(self) -> bool:
        return isinstance(self.image_patches, ImagePatchConfig)

    def check_compatibility(self, preprocessing: typing.Self) -> None:
        Assert.custom(isinstance, preprocessing, LanguageModelPreprocessingConfig)
        # TODO: Check more tokenizer data, ex. bos/eos tokens? path if points to HF hub?
        if self.vocab_size is not None and preprocessing.vocab_size is not None:
            Assert.leq(self.vocab_size, preprocessing.vocab_size)
        if preprocessing.use_preference_spans:
            # Preference spans are strictly needed for DPO loss.
            assert self.use_preference_spans, "The dataset is missing required preference spans"
        if preprocessing.use_image_patches and self.use_image_patches:
            self.image_patches.check_compatibility(preprocessing.image_patches)


@dataclasses.dataclass
class MicroBatch:
    pass


@dataclasses.dataclass
class PreprocessedBatch:
    micro_batches: list[MicroBatch]


@config_class(registry=True)
class BatchPreprocessingConfig(PreprocessingConfig):
    batch: BatchConfig = Field()

    @classmethod
    def from_documents(
        cls,
        config: BatchPreprocessingConfig,
        distributed_config: DistributedConfig,
        documents: list[Document],
        device: "torch.device | None" = None,
    ) -> typing.Self:
        pass
