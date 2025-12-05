import pathlib
import typing

import numpy as np
import torch

from fast_llm.config import NoAutoValidate
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.config import SampledDatasetConfig, SamplingConfig, SamplingParameters, ShufflingType
from fast_llm.data.dataset.gpt.config import GPTSamplingData
from fast_llm.data.dataset.indexed import IndexedDataset
from fast_llm.data.dataset.sampled import SampledIndexedDataset
from fast_llm.data.preprocessing.language_model import LanguageModelPreprocessingConfig
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.models.gpt.config import GPTBatchConfig
from fast_llm.utils import Assert, div


def get_sampling_data(
    num_samples: int,
    *,
    seed: int = 54983,
    cache_directory: pathlib.Path | None = None,
    phase=PhaseType.training,
    sequence_length: int = 512,
    gpu: bool = False,
    shuffle: ShufflingType = ShufflingType.epoch,
    truncate_documents=True,
    preprocessing: LanguageModelPreprocessingConfig,
) -> GPTSamplingData:
    # Config with convenient defaults.
    distributed = Distributed(DistributedConfig(), use_cpu=True)
    return GPTSamplingData(
        config=SamplingConfig(
            seed=seed,
            gpu=gpu,
            shuffle=shuffle,
        ),
        parameters=SamplingParameters(
            num_samples=num_samples,
            sequence_length=sequence_length,
            truncate_documents=truncate_documents,
        ),
        preprocessing=preprocessing,
        cache_directory=cache_directory,
        distributed=distributed,
        dataset_name=phase.value,
    )


def get_dataset_config[T: SampledDatasetConfig](config: dict[str, typing.Any], cls: type[T]) -> T:
    dataset_config = SampledDatasetConfig.from_dict(config)
    Assert.custom(isinstance, dataset_config, getattr(cls, "__origin__", cls))
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
    expected_samples: dict[str, list[list[int]]] | list[list[int]],
    preprocessing: LanguageModelPreprocessingConfig,
) -> GPTData:
    distributed_config = DistributedConfig(seed=87522)
    distributed = Distributed(distributed_config, use_cpu=True)
    if isinstance(samples_per_dataset, int):
        samples_per_dataset = {PhaseType.training.value.lower(): samples_per_dataset}

    sampling_parameters = {
        dataset_name: SamplingParameters(num_samples=num_samples, sequence_length=sequence_length)
        for dataset_name, num_samples in samples_per_dataset.items()
    }

    if isinstance(expected_samples, list):
        expected_samples = {PhaseType.training.value.lower(): expected_samples}

    assert "sampling" not in config
    config["sampling"] = SamplingConfig(seed=seed, gpu=gpu, shuffle=shuffle)
    data = GPTData(GPTDataConfig.from_dict(config), distributed_config)
    data.setup(distributed, sampling_parameters, preprocessing, cache_directory)
    with NoAutoValidate():
        batch_config = GPTBatchConfig(batch_size=1, sequence_length=sequence_length)
    batch_config.setup(distributed_config)
    batch_config.validate()
    tokens = {
        phase: torch.stack(
            [
                batch.tokens.tokens[0]
                for batch in data.get_iterator(batch_config, phase, consumed_samples=0, num_workers=0)
            ]
        )
        for phase, samples in samples_per_dataset.items()
    }
    for phase, expected_samples_ in expected_samples.items():
        Assert.all_equal(tokens[phase], expected_samples_)
    return data


def compare_indexed_dataset_tokens(
    dataset: IndexedDataset,
    length: int,
    num_tokens: int,
    expected_samples: dict[int, list[int]],
) -> None:
    Assert.eq(len(dataset), length)
    sizes = dataset.get_document_sizes()
    Assert.eq(sizes.sum(), num_tokens, dataset.num_tokens)
    Assert.all_equal(
        [len(dataset.get_document(i).tokens.tokens) for i in range(min(len(dataset), 100))],
        sizes[: min(len(dataset), 100)],
    )
    for i, expected_sample in expected_samples.items():
        Assert.all_equal(dataset.get_document(i).tokens.tokens, np.array(expected_sample))


def compare_sampled_dataset(sampled: SampledDataset, expected_samples: list[list[int] | np.ndarray]) -> None:
    Assert.eq(len(sampled), len(expected_samples))
    Assert.all_equal(torch.stack([sampled[i].tokens.tokens for i in range(len(expected_samples))]), expected_samples)


def validate_indexed_dataset_sampling(sampled: SampledIndexedDataset, expected_samples: list[list[int]] | None = None):
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
        document = sampled._indexed_dataset.get_document(document_index).tokens.tokens

        all_tokens[seen_tokens : seen_tokens + len(document)] = document[: num_tokens - seen_tokens]
        seen_tokens += len(document)
        if seen_tokens >= num_tokens:
            break

    validate_samples = [
        all_tokens[index * sampled._parameters.sequence_length : (index + 1) * sampled._parameters.sequence_length + 1]
        for index in range(sampled._parameters.num_samples)
    ]
    token_ids = torch.stack([sampled[i].tokens.tokens for i in range(len(sampled))]).to(torch.int64)
    Assert.all_equal(token_ids, validate_samples)
    if expected_samples is not None:
        Assert.all_equal(token_ids, expected_samples)
    return token_ids
