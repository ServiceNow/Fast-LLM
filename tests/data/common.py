import pathlib
import typing

import numpy as np
import torch

from fast_llm.config import NoAutoValidate
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.gpt.config import GPTSampledDatasetConfig, GPTSamplingConfig
from fast_llm.data.dataset.gpt.indexed import GPTIndexedDataset
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.utils import Assert
from tests.common import TEST_VOCAB_SIZE


def get_sampling_config(
    num_samples: int,
    *,
    seed: int = 54983,
    cache_directory: pathlib.Path | None = None,
    distributed: Distributed = Distributed(DistributedConfig(), use_cpu=True),
    phase=PhaseType.training,
    sequence_length: int = 512,
    vocab_size=TEST_VOCAB_SIZE,
    tokenizer: Tokenizer | None = None,
) -> GPTSamplingConfig:
    # Config with convenient defaults.
    return GPTSamplingConfig(
        num_samples=num_samples,
        seed=seed,
        cache_directory=cache_directory,
        distributed=distributed,
        phase=phase,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
    )


def get_dataset_config[T: GPTSampledDatasetConfig](config: dict[str, typing.Any], cls: type[T]) -> T:
    dataset_config = GPTSampledDatasetConfig.from_dict(config)
    Assert.custom(isinstance, dataset_config, cls)
    return typing.cast(cls, dataset_config)


def get_test_data_and_compare_samples(
    config: dict,
    samples_per_phase: dict[PhaseType, int],
    *,
    seed: int = 54983,
    cache_directory: pathlib.Path | None = None,
    sequence_length: int = 512,
    vocab_size=TEST_VOCAB_SIZE,
    expected_samples: dict[PhaseType, list[list[int]]],
) -> GPTData:
    distributed_config = DistributedConfig(seed=seed)
    distributed = Distributed(distributed_config, use_cpu=True)
    data = GPTData(GPTDataConfig.from_dict(config), distributed_config, vocab_size, sequence_length)
    data.setup(distributed, samples_per_phase, cache_directory)
    with NoAutoValidate():
        batch_config = BatchConfig(batch_size=1, sequence_length=sequence_length)
    batch_config.setup(distributed_config)
    batch_config.validate()
    samples = {
        phase: torch.stack(
            [batch.token_ids[0] for batch in data.get_iterator(batch_config, phase, consumed_samples=0, num_workers=0)]
        )
        for phase, samples in samples_per_phase.items()
    }
    for phase, expected_samples_ in expected_samples.items():
        Assert.all_equal(samples[phase], expected_samples_)
    return data


def compare_indexed_dataset(
    dataset: GPTIndexedDataset,
    length: int,
    num_tokens: int,
    samples: dict[int, list[int]],
    loss_masking_spans: dict[int, list[int]] | None = None,
) -> None:
    Assert.eq(len(dataset), length)
    sizes = dataset.get_document_sizes()
    Assert.eq(sizes.sum(), num_tokens)
    Assert.all_equal(
        [len(dataset.get(i).token_ids) for i in range(min(len(dataset), 100))], sizes[: min(len(dataset), 100)]
    )
    for i, sample in samples.items():
        dataset_sample = dataset.get(i, use_loss_masking_spans=loss_masking_spans is not None)
        Assert.all_equal(dataset_sample.token_ids, np.array(sample, dtype=np.uint16))
        if loss_masking_spans:
            Assert.all_equal(
                dataset_sample.loss_masking_spans, np.array(loss_masking_spans[i], dtype=np.int32).reshape(-1, 2)
            )


def compare_sampled_dataset(sampled: SampledDataset, expected_samples: list[list[int]]) -> None:
    Assert.eq(len(sampled), len(expected_samples))
    Assert.all_equal([sampled[i].token_ids for i in range(len(expected_samples))], expected_samples)
