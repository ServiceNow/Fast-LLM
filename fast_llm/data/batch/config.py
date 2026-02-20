import abc
import dataclasses
import functools
import logging
import typing

from fast_llm.config import Configurable, Field, FieldUpdate, config_class
from fast_llm.data.document.abstract import Batch, Document
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
    batch: BatchConfig = Field()
    phase: PhaseType = Field(default=PhaseType.inference)

    def get_batch_meta(self) -> "PreprocessedBatch":
        raise NotImplementedError()


@config_class()
class LanguageModelBatchPreprocessingConfig(LanguageModelPreprocessingConfig, BatchPreprocessingConfig):
    _abstract = False
    # TODO: Duplicate `use_loss_masking_spans`, `use_preference_spans`
    batch: GPTBatchConfig = FieldUpdate()
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

    def get_batch_meta(self) -> "PreprocessedBatch":
        from fast_llm.data.batch.language_model import LanguageModelPreprocessedBatch
        from fast_llm.data.document.language_model import LanguageModelBatch, LanguageModelDocument
        from fast_llm.data.document.token import TokenDocument

        device = torch.device("meta")
        tokens = torch.empty(self.total_length, dtype=torch.int64, device=device)
        batch = LanguageModelBatch.from_documents([LanguageModelDocument(tokens=TokenDocument(tokens=tokens))])
        return LanguageModelPreprocessedBatch.from_batch(batch, config=self, device=device)

    @functools.cached_property
    def use_image_patches(self) -> bool:
        return isinstance(self.image_patches, ImagePatchConfig)

    @functools.cached_property
    def total_length(self) -> int:
        return self.batch.sequence_length + self.predicted_tokens

    @functools.cached_property
    def distributed(self) -> DistributedConfig:
        return self.batch.distributed

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


class PreprocessedBatch[ConfigType: BatchPreprocessingConfig, MicroBatchType: MicroBatch](Configurable[ConfigType]):
    def __init__(self, config: ConfigType, micro_batches: list[MicroBatchType]):
        super().__init__(config)
        self._micro_batches = micro_batches

    @property
    def micro_batches(self) -> list[MicroBatch]:
        return self._micro_batches

    def __len__(self) -> int:
        return len(self._micro_batches)

    def __getitem__(self, idx: int) -> MicroBatchType:
        return self._micro_batches[idx]

    @classmethod
    @abc.abstractmethod
    def from_documents(
        cls,
        documents: list[Document],
        config: BatchPreprocessingConfig,
        device: "torch.device | None" = None,
    ) -> typing.Self:
        pass

    @classmethod
    @abc.abstractmethod
    def from_batch(
        cls,
        batch: Batch,
        config: BatchPreprocessingConfig,
        device: "torch.device | None" = None,
    ) -> typing.Self:
        pass
