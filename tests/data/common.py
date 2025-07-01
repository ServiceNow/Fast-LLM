import pathlib
import typing

import numpy as np
import torch

from fast_llm.config import Field, FieldHint, NoAutoValidate, config_class
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.gpt.config import (
    GPTIndexedDatasetConfig,
    GPTSampledDatasetConfig,
    GPTSamplingConfig,
    GPTSamplingData,
    GPTSamplingParameters,
    ShufflingType,
)
from fast_llm.data.dataset.gpt.indexed import GPTIndexedDataset
from fast_llm.data.dataset.gpt.sampled import GPTSampledIndexedDataset
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.models.gpt.config import GPTBatchConfig
from fast_llm.utils import Assert, div
from tests.utils.dataset import TEST_VOCAB_SIZE


def get_sampling_data(
    num_samples: int,
    *,
    seed: int = 54983,
    cache_directory: pathlib.Path | None = None,
    phase=PhaseType.training,
    sequence_length: int = 512,
    vocab_size=TEST_VOCAB_SIZE,
    tokenizer: Tokenizer | None = None,
    gpu: bool = False,
    shuffle: ShufflingType = ShufflingType.epoch,
    truncate_documents=True,
) -> GPTSamplingData:
    # Config with convenient defaults.
    distributed = Distributed(DistributedConfig(), use_cpu=True)
    return GPTSamplingData(
        config=GPTSamplingConfig(
            seed=seed,
            gpu=gpu,
            shuffle=shuffle,
        ),
        parameters=GPTSamplingParameters(
            num_samples=num_samples,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            truncate_documents=truncate_documents,
        ),
        cache_directory=cache_directory,
        distributed=distributed,
        dataset_name=phase.value,
        tokenizer=tokenizer,
    )


def get_dataset_config[T: GPTSampledDatasetConfig](config: dict[str, typing.Any], cls: type[T]) -> T:
    dataset_config = GPTSampledDatasetConfig.from_dict(config)
    Assert.custom(isinstance, dataset_config, cls)
    return typing.cast(cls, dataset_config)


def get_test_data_and_compare_samples(
    config: dict,
    samples_per_dataset: dict[str, int] | int,
    *,
    seed: int = 54983,
    gpu: bool = False,
    shuffle: ShufflingType = ShufflingType.epoch,
    cache_directory: pathlib.Path | None = None,
    sequence_length: int = 512,
    vocab_size=TEST_VOCAB_SIZE,
    expected_samples: dict[str, list[list[int]]] | list[list[int]],
    legacy: bool = False,
) -> GPTData:
    distributed_config = DistributedConfig(seed=seed if legacy else 87522)
    distributed = Distributed(distributed_config, use_cpu=True)
    if isinstance(samples_per_dataset, int):
        samples_per_dataset = {PhaseType.training.value.lower(): samples_per_dataset}

    sampling_parameters = {
        dataset_name: GPTSamplingParameters(
            num_samples=num_samples,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
        )
        for dataset_name, num_samples in samples_per_dataset.items()
    }

    if isinstance(expected_samples, list):
        expected_samples = {PhaseType.training.value.lower(): expected_samples}

    assert "sampling" not in config
    config["sampling"] = GPTSamplingConfig(
        seed=87522 if legacy else seed,
        gpu=gpu,
        shuffle=shuffle,
    )
    data = GPTData(GPTDataConfig.from_dict(config), distributed_config)
    data.setup(distributed, sampling_parameters, cache_directory)
    with NoAutoValidate():
        batch_config = GPTBatchConfig(batch_size=1, sequence_length=sequence_length)
    batch_config.setup(distributed_config)
    batch_config.validate()
    tokens = {
        phase: torch.stack(
            [batch.token_ids[0] for batch in data.get_iterator(batch_config, phase, consumed_samples=0, num_workers=0)]
        )
        for phase, samples in samples_per_dataset.items()
    }
    for phase, expected_samples_ in expected_samples.items():
        Assert.all_equal(tokens[phase], expected_samples_)
    return data


def compare_indexed_dataset(
    dataset: GPTIndexedDataset,
    length: int,
    num_tokens: int,
    expected_samples: dict[int, list[int]],
    loss_masking_spans: dict[int, list[int]] | None = None,
) -> None:
    Assert.eq(len(dataset), length)
    sizes = dataset.get_document_sizes()
    # Assert.eq(sizes.sum(), num_tokens)
    Assert.all_equal(
        [len(dataset.get(i).token_ids) for i in range(min(len(dataset), 100))], sizes[: min(len(dataset), 100)]
    )
    for i, expected_sample in expected_samples.items():
        Assert.all_equal(dataset.get(i).token_ids, np.array(expected_sample, dtype=np.uint16))
    if loss_masking_spans:
        for i, loss_masking_span in loss_masking_spans.items():
            Assert.all_equal(
                dataset.get(i, use_loss_masking_spans=True).loss_masking_spans,
                np.array(loss_masking_spans[i], dtype=np.int32).reshape(-1, 2),
            )


def compare_sampled_dataset(sampled: SampledDataset, expected_samples: list[list[int] | np.ndarray]) -> None:
    Assert.eq(len(sampled), len(expected_samples))
    Assert.all_equal([sampled[i].token_ids for i in range(len(expected_samples))], expected_samples)


def validate_indexed_dataset_sampling(
    sampled: GPTSampledIndexedDataset, expected_samples: list[list[int]] | None = None
):
    """
    Compare `GPTSampledIndexedDataset` sampling against a more basic approach
    """
    num_tokens = sampled._parameters.num_samples * sampled._parameters.sequence_length + 1
    all_tokens = np.full(sampled._parameters.num_samples * sampled._parameters.sequence_length + 1, -1, dtype=np.int64)
    unshuffled_epochs = div(sampled._unshuffled_documents, sampled._documents_per_epoch)

    document_sampling = np.tile(
        np.arange(sampled._documents_per_epoch, dtype=np.int64),
        unshuffled_epochs,
    )
    if sampled._document_shuffling.exists():
        document_sampling = np.concatenate(
            (
                document_sampling,
                sampled._document_shuffling.array,
            )
        )
    seen_tokens = 0
    for document_index in document_sampling:
        document = sampled._indexed_dataset.get(document_index).token_ids

        all_tokens[seen_tokens : seen_tokens + len(document)] = document[: num_tokens - seen_tokens]
        seen_tokens += len(document)
        if seen_tokens >= num_tokens:
            break

    validate_samples = [
        all_tokens[index * sampled._parameters.sequence_length : (index + 1) * sampled._parameters.sequence_length + 1]
        for index in range(sampled._parameters.num_samples)
    ]
    token_ids = [sampled[i].token_ids for i in range(len(sampled))]

    Assert.all_equal(token_ids, validate_samples)
    if expected_samples is not None:
        Assert.all_equal(token_ids, expected_samples)
    return token_ids


@config_class(dynamic_type={GPTSampledDatasetConfig: "mock_memmap"})
class MockGPTMemmapDatasetConfig(GPTIndexedDatasetConfig):
    _abstract: typing.ClassVar[bool] = False
    num_documents: int | None = Field(
        default=None,
        desc="Expected number of documents in the dataset.",
        hint=FieldHint.core,
    )
    num_tokens_per_document: int | None = Field(
        default=None,
        desc="Expected number of tokens in the dataset.",
        hint=FieldHint.optional,
    )
    path: pathlib.Path = Field(default=".")

    def build(self) -> "GPTIndexedDataset":
        return MockGPTMemmapDataset(self)

    @property
    def num_tokens(self) -> int:
        return self.num_documents * self.num_tokens_per_document


class MockGPTMemmapDataset(GPTIndexedDataset):
    def __init__(self, config: MockGPTMemmapDatasetConfig):
        self._config = config

    @property
    def name(self) -> str:
        return "mock_memmap"

    def __len__(self) -> int:
        return self._config.num_documents

    def get_document_sizes(self) -> np.ndarray:
        return np.full(self._config.num_documents, self._config.num_tokens_per_document, dtype=np.int64)

    def get_document_size(self, index: int) -> int:
        return self._config.num_tokens_per_document

    def get(self, index: int, *args, **kwargs) -> typing.Any:
        raise NotImplementedError()
