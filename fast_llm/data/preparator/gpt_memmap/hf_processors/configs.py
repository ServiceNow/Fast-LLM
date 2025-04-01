import abc
import datasets
import typing

from fast_llm.config import Config, Configurable, Field, FieldUpdate, config_class
from fast_llm.data.preparator.gpt_memmap.distributed_config import DatasetPreparatorDistributedConfig


# TODO: Add desc and hint to all fields.


@config_class
class HFProcessorConfig(Config):
    use_processor: bool = Field(default=True)
    human_readable_name: str = Field(default="")
    batch_size: int | None = Field(default=None)
    num_proc: int | None = Field(default=None)
    field: str | None = Field(default=None)


class HFProcessor[ConfigType: HFProcessorConfig](Configurable[ConfigType], abc.ABC):
    config_class: typing.ClassVar[type[HFProcessorConfig]] = HFProcessorConfig

    def __init__(self, config: ConfigType, distributed_config: DatasetPreparatorDistributedConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self._distributed_config = distributed_config

    @abc.abstractmethod
    def apply(self, dataset: datasets.Dataset) -> datasets.Dataset:
        raise NotImplementedError


@config_class
class DocLengthFilterProcessorConfig(HFProcessorConfig):
    human_readable_name: str | None = FieldUpdate(default="Document Length Filter")
    min_length_chars: int = Field(default=0)
    max_length_chars: int = Field(default=1_000_000)


class DocLengthFilterProcessor[ConfigType: DocLengthFilterProcessorConfig](HFProcessor[ConfigType]):
    config_class: typing.ClassVar[type[DocLengthFilterProcessorConfig]] = DocLengthFilterProcessorConfig

    def apply(self, dataset: datasets.Dataset) -> datasets.Dataset:
        from fast_llm.data.preparator.gpt_memmap.hf_processors.processors import apply_doc_length_filter_processor

        return apply_doc_length_filter_processor(self._config, dataset)


@config_class
class NGramRepetitionFilterProcessorConfig(HFProcessorConfig):
    human_readable_name: str | None = FieldUpdate(default="N-Gram Repetition Filter")
    n: int = Field(default=5)
    max_repetitions: int = Field(default=32)


class NGramRepetitionFilterProcessor[ConfigType: NGramRepetitionFilterProcessorConfig](HFProcessor[ConfigType]):
    config_class: typing.ClassVar[type[NGramRepetitionFilterProcessorConfig]] = NGramRepetitionFilterProcessorConfig

    def apply(self, dataset: datasets.Dataset) -> datasets.Dataset:
        from fast_llm.data.preparator.gpt_memmap.hf_processors.processors import (
            apply_ngram_repetition_filter_processor,
        )

        return apply_ngram_repetition_filter_processor(self._config, dataset)


@config_class
class FrequencyBasedFilterProcessorConfig(HFProcessorConfig):
    human_readable_name: str | None = FieldUpdate(default="Frequency-Based Filter")
    max_single_word_ratio: float = Field(default=0.3)
    max_top_two_word_ratio: float = Field(default=0.5)


class FrequencyBasedFilterProcessor[ConfigType: FrequencyBasedFilterProcessorConfig](HFProcessor[ConfigType]):
    config_class: typing.ClassVar[type[FrequencyBasedFilterProcessorConfig]] = FrequencyBasedFilterProcessorConfig

    def apply(self, dataset: datasets.Dataset) -> datasets.Dataset:
        from fast_llm.data.preparator.gpt_memmap.hf_processors.processors import apply_frequency_based_filter_processor

        return apply_frequency_based_filter_processor(self._config, dataset)


@config_class
class BinaryContentFilterProcessorConfig(HFProcessorConfig):
    human_readable_name: str | None = FieldUpdate(default="Binary Content Filter")
    max_bin_ratio: float = Field(default=0.5)


class BinaryContentFilterProcessor[ConfigType: BinaryContentFilterProcessorConfig](HFProcessor[ConfigType]):
    config_class: typing.ClassVar[type[BinaryContentFilterProcessorConfig]] = BinaryContentFilterProcessorConfig

    def apply(self, dataset: datasets.Dataset) -> datasets.Dataset:
        from fast_llm.data.preparator.gpt_memmap.hf_processors.processors import apply_binary_content_filter_processor

        return apply_binary_content_filter_processor(self._config, dataset)


@config_class
class NumericalContentFilterProcessorConfig(HFProcessorConfig):
    human_readable_name: str | None = FieldUpdate(default="Numerical Content Filter")
    max_numeric_token_ratio: float = Field(default=0.5)


class NumericalContentFilterProcessor[ConfigType: NumericalContentFilterProcessorConfig](HFProcessor[ConfigType]):
    config_class: typing.ClassVar[type[NumericalContentFilterProcessorConfig]] = NumericalContentFilterProcessorConfig

    def apply(self, dataset: datasets.Dataset) -> datasets.Dataset:
        from fast_llm.data.preparator.gpt_memmap.hf_processors.processors import (
            apply_numerical_content_filter_processor,
        )

        return apply_numerical_content_filter_processor(self._config, dataset)


@config_class
class PiiRedactionProcessorConfig(HFProcessorConfig):
    use_processor: bool = FieldUpdate(default=False)
    human_readable_name: str | None = FieldUpdate(default="PII Redaction Processor")
    # TODO: make enum
    redaction_method: str = Field(default="remove")  # Options: 'remove', 'mask'


class PiiRedactionProcessor[ConfigType: PiiRedactionProcessorConfig](HFProcessor[ConfigType]):
    config_class: typing.ClassVar[type[PiiRedactionProcessorConfig]] = PiiRedactionProcessorConfig

    def apply(self, dataset: datasets.Dataset) -> datasets.Dataset:
        from fast_llm.data.preparator.gpt_memmap.hf_processors.processors import apply_pii_redaction_processor

        return apply_pii_redaction_processor(self._config, self._distributed_config, dataset)


@config_class
class MalwareRemovalProcessorConfig(HFProcessorConfig):
    use_processor: bool = FieldUpdate(default=False)
    human_readable_name: str | None = FieldUpdate(default="Malware Removal Processor")


class MalwareRemovalProcessor[ConfigType: MalwareRemovalProcessorConfig](HFProcessor[ConfigType]):
    config_class: typing.ClassVar[type[MalwareRemovalProcessorConfig]] = MalwareRemovalProcessorConfig

    def apply(self, dataset: datasets.Dataset) -> datasets.Dataset:
        from fast_llm.data.preparator.gpt_memmap.hf_processors.processors import apply_malware_removal_processor

        return apply_malware_removal_processor(self._config, dataset)


@config_class
class ProcessorsConfig(Config):
    doc_length: DocLengthFilterProcessorConfig = Field(default=DocLengthFilterProcessorConfig)
    n_gramms: NGramRepetitionFilterProcessorConfig = Field(default=NGramRepetitionFilterProcessorConfig)
    frequency: FrequencyBasedFilterProcessorConfig = Field(default=FrequencyBasedFilterProcessorConfig)
    binary: BinaryContentFilterProcessorConfig = Field(default=BinaryContentFilterProcessorConfig)
    numerical: NumericalContentFilterProcessorConfig = Field(default=NumericalContentFilterProcessorConfig)
    pii: PiiRedactionProcessorConfig = Field(default=PiiRedactionProcessorConfig)
    malware: MalwareRemovalProcessorConfig = Field(default=MalwareRemovalProcessorConfig)

    # TODO: add validation so all steps are actual field names
    order: list[str] = Field(
        default_factory=lambda: ["doc_length", "n_gramms", "frequency", "binary", "numerical", "pii", "malware"]
    )

    def get_processor_types_map(self):
        return {
            "doc_length": DocLengthFilterProcessor,
            "n_gramms": NGramRepetitionFilterProcessor,
            "frequency": FrequencyBasedFilterProcessor,
            "binary": BinaryContentFilterProcessor,
            "numerical": NumericalContentFilterProcessor,
            "pii": PiiRedactionProcessor,
            "malware": MalwareRemovalProcessor,
        }
