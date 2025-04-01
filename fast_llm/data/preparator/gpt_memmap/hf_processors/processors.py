import collections
import datasets
import logging
import re


from fast_llm.data.preparator.gpt_memmap.distributed_config import DatasetPreparatorDistributedConfig
from fast_llm.data.preparator.gpt_memmap.hf_processors.configs import (
    DocLengthFilterProcessorConfig,
    NGramRepetitionFilterProcessorConfig,
    FrequencyBasedFilterProcessorConfig,
    BinaryContentFilterProcessorConfig,
    NumericalContentFilterProcessorConfig,
    PiiRedactionProcessorConfig,
    MalwareRemovalProcessorConfig,
)


logger = logging.getLogger(__name__)

WORD_PATTERN = r"\b\w+(?:'\w+)?\b"
NUMBER_PATTERN = r"\b\d+\b"


def apply_doc_length_filter_processor(
    config: DocLengthFilterProcessorConfig, dataset: datasets.Dataset
) -> datasets.Dataset:
    return dataset.filter(
        lambda batch: [
            config.min_length_chars <= len(text) <= config.max_length_chars for text in batch[config.field]
        ],
        num_proc=config.num_proc,
        batched=True,
        batch_size=config.batch_size,
    )


def apply_ngram_repetition_filter_processor(
    config: NGramRepetitionFilterProcessorConfig, dataset: datasets.Dataset
) -> datasets.Dataset:
    def has_repeated_ngrams(batch):
        results = []
        word_pattern = re.compile(WORD_PATTERN)
        for text in batch[config.field]:
            words = word_pattern.findall(text)
            ngrams = [tuple(words[i : i + config.n]) for i in range(len(words) - config.n + 1)]
            ngram_counts = collections.Counter(ngrams)
            results.append(max(ngram_counts.values(), default=0) <= config.max_repetitions)
        return results

    return dataset.filter(
        has_repeated_ngrams,
        num_proc=config.num_proc,
        batched=True,
        batch_size=config.batch_size,
    )


def apply_frequency_based_filter_processor(
    config: FrequencyBasedFilterProcessorConfig, dataset: datasets.Dataset
) -> datasets.Dataset:
    def exceeds_word_frequency_threshold(batch):
        results = []
        word_pattern = re.compile(WORD_PATTERN)
        for text in batch[config.field]:
            words = word_pattern.findall(text)
            total_words = len(words)
            word_counts = collections.Counter(words)
            most_common = word_counts.most_common(2)

            if most_common and (most_common[0][1] / total_words) > config.max_single_word_ratio:
                results.append(False)
            elif (
                len(most_common) > 1
                and ((most_common[0][1] + most_common[1][1]) / total_words) > config.max_top_two_word_ratio
            ):
                results.append(False)
            else:
                results.append(True)
        return results

    return dataset.filter(
        exceeds_word_frequency_threshold,
        num_proc=config.num_proc,
        batched=True,
        batch_size=config.batch_size,
    )


def apply_binary_content_filter_processor(
    config: BinaryContentFilterProcessorConfig, dataset: datasets.Dataset
) -> datasets.Dataset:
    def is_binary(batch):
        return [
            not sum(1 for char in text if char.isprintable()) / len(text) < config.max_bin_ratio
            for text in batch[config.field]
        ]

    return dataset.filter(is_binary, num_proc=config.num_proc, batched=True, batch_size=config.batch_size)


def apply_numerical_content_filter_processor(
    config: NumericalContentFilterProcessorConfig, dataset: datasets.Dataset
) -> datasets.Dataset:
    def exceeds_numeric_threshold(batch):
        results = []
        number_pattern = re.compile(NUMBER_PATTERN)
        for text in batch[config.field]:
            tokens = number_pattern.findall(text)
            results.append((len(tokens) / max(1, len(text.split()))) <= config.max_numeric_token_ratio)
        return results

    return dataset.filter(
        exceeds_numeric_threshold, num_proc=config.num_proc, batched=True, batch_size=config.batch_size
    )


def apply_pii_redaction_processor(
    config: PiiRedactionProcessorConfig,
    distributed_condig: DatasetPreparatorDistributedConfig,
    dataset: datasets.Dataset,
) -> datasets.Dataset:
    # TODO: check if multiprocessing is possible
    # TODO: manage explicit model download and loading as now it
    # internally install a python package which is not transferable to workrs

    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    def redact_pii(batch):
        results = []
        for text in batch[config.field]:
            entities = analyzer.analyze(
                text=text, entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"], language="en"
            )
            if config.redaction_method == "remove":
                for result in reversed(entities):
                    text = text[: result.start] + "" + text[result.end :]
            elif config.redaction_method == "mask":
                text = anonymizer.anonymize(text, entities).text
            else:
                raise ValueError(f"Unkown redaction method: {config.redaction_method}")
            results.append(text)
        return {config.field: results}

    return dataset.map(redact_pii, num_proc=None, batched=True, batch_size=config.batch_size)


def apply_malware_removal_processor(
    config: MalwareRemovalProcessorConfig, dataset: datasets.Dataset
) -> datasets.Dataset:
    # TODO: this is not working, scan_bytes does not exist.
    #  Rewrite either with downloading virus definitions file,
    #  loading dataset and running a file or use clamav directly
    import clamav

    def is_malicious(batch):
        return [not clamav.scan_bytes(text.encode()) for text in batch[config.field]]

    return dataset.filter(is_malicious, num_proc=config.num_proc, batched=True, batch_size=config.batch_size)
