import datasets

from fast_llm.data.preparator.gpt_memmap.distributed_config import DatasetPreparatorDistributedConfig
from fast_llm.data.preparator.gpt_memmap.hf_processors.configs import (
    DocLengthFilterProcessorConfig,
    NGramRepetitionFilterProcessorConfig,
    FrequencyBasedFilterProcessorConfig,
    BinaryContentFilterProcessorConfig,
    NumericalContentFilterProcessorConfig,
    PiiRedactionProcessorConfig,
    MalwareRemovalProcessorConfig,
    DocLengthFilterProcessor,
    NGramRepetitionFilterProcessor,
    FrequencyBasedFilterProcessor,
    BinaryContentFilterProcessor,
    NumericalContentFilterProcessor,
    PiiRedactionProcessor,
    MalwareRemovalProcessor,
)


def create_test_dataset(data):
    return datasets.Dataset.from_dict({"text": data})


def test_doc_length_filter_processor():
    dataset = create_test_dataset(["short", "this is a medium length sentence", "this is a very long text" * 100])
    config = DocLengthFilterProcessorConfig(min_length_chars=10, max_length_chars=50, field="text")
    processor = DocLengthFilterProcessor(config, DatasetPreparatorDistributedConfig())
    filtered_dataset = processor.apply(dataset)
    assert len(filtered_dataset) == 1  # Only one entry should match the criteria


def test_ngram_repetition_filter_processor():
    dataset = create_test_dataset(
        ["word word word", "word word word word", "unique words here", "repeat repeat repeat repeat repeat"]
    )
    config = NGramRepetitionFilterProcessorConfig(n=2, max_repetitions=2, field="text")
    processor = NGramRepetitionFilterProcessor(config, DatasetPreparatorDistributedConfig())
    filtered_dataset = processor.apply(dataset)
    assert len(filtered_dataset) == 2  # Only "word word word" and "unique words here" should remain


def test_frequency_based_filter_processor():
    dataset = create_test_dataset(["hello hello hello world", "this is fine just because", "spam spam spam spam spam"])
    config = FrequencyBasedFilterProcessorConfig(max_single_word_ratio=0.4, max_top_two_word_ratio=0.6, field="text")
    processor = FrequencyBasedFilterProcessor(config, DatasetPreparatorDistributedConfig())
    filtered_dataset = processor.apply(dataset)
    assert len(filtered_dataset) == 1  # Only "this is fine" should remain


def test_binary_content_filter_processor():
    dataset = create_test_dataset(["hello world", b"\x00\x00\x01\x02bin".decode("utf8"), "normal text"])
    config = BinaryContentFilterProcessorConfig(max_bin_ratio=0.5, field="text")
    processor = BinaryContentFilterProcessor(config, DatasetPreparatorDistributedConfig())
    filtered_dataset = processor.apply(dataset)
    assert len(filtered_dataset) == 2  # Binary data should be removed


def test_numerical_content_filter_processor():
    dataset = create_test_dataset(
        ["123 456 789", "some words and 123", "almost all numbers 123 456 789 101112 131415"]
    )
    config = NumericalContentFilterProcessorConfig(max_numeric_token_ratio=0.5, field="text")
    processor = NumericalContentFilterProcessor(config, DatasetPreparatorDistributedConfig())
    filtered_dataset = processor.apply(dataset)
    assert len(filtered_dataset) == 1  # Only "some words and 123" should remain


# TODO: Make optional conditioned on library installed
def test_pii_redaction_processor():
    dataset = create_test_dataset(["My name is John Doe", "Contact me at john@example.com", "This is safe text"])
    config = PiiRedactionProcessorConfig(redaction_method="remove", field="text")
    processor = PiiRedactionProcessor(config, DatasetPreparatorDistributedConfig())
    processed_dataset = processor.apply(dataset)
    assert "John Doe" not in processed_dataset["text"]
    assert "john@example.com" not in processed_dataset["text"]


# TODO: Make optional conditioned on library installed
# def test_malware_removal_processor():
#     dataset = create_test_dataset(["malicious_code();", "safe text", "virus_payload();"])
#     config = MalwareRemovalProcessorConfig(field="text")
#     processor = MalwareRemovalProcessor(config, DatasetPreparatorDistributedConfig())
#     filtered_dataset = processor.apply(dataset)
#     assert len(filtered_dataset) == 1  # Only "safe text" should remain
