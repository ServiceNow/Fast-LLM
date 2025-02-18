import pathlib
import typing

import numpy as np
import torch

from fast_llm.config import NoAutoValidate
from fast_llm.data.data.gpt.config import GPTDataConfig, GPTSamplingDefaultConfig
from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.gpt.config import GPTSampledDatasetConfig, GPTSamplingData, ShufflingType
from fast_llm.data.dataset.gpt.indexed import GPTIndexedDataset
from fast_llm.data.dataset.gpt.sampled import GPTSampledIndexedDataset
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.utils import Assert, div
from tests.common import TEST_VOCAB_SIZE


def get_sampling_data(
    num_samples: int,
    *,
    seed: int = 54983,
    cache_directory: pathlib.Path | None = None,
    distributed: Distributed = Distributed(DistributedConfig(), use_cpu=True),
    phase=PhaseType.training,
    sequence_length: int = 512,
    vocab_size=TEST_VOCAB_SIZE,
    tokenizer: Tokenizer | None = None,
    gpu: bool = False,
    shuffle: ShufflingType = ShufflingType.epoch,
) -> GPTSamplingData:
    # Config with convenient defaults.
    return GPTSamplingData(
        config=GPTSamplingDefaultConfig(
            seed=seed,
            gpu=gpu,
            shuffle=shuffle,
        ),
        num_samples=num_samples,
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
    gpu: bool = False,
    shuffle: ShufflingType = ShufflingType.epoch,
    cache_directory: pathlib.Path | None = None,
    sequence_length: int = 512,
    vocab_size=TEST_VOCAB_SIZE,
    expected_samples: dict[PhaseType, list[list[int]]],
    legacy: bool = False,
) -> GPTData:
    distributed_config = DistributedConfig(seed=seed if legacy else 87522)
    distributed = Distributed(distributed_config, use_cpu=True)
    assert "sampling" not in config
    config["sampling"] = GPTSamplingDefaultConfig(
        seed=87522 if legacy else seed,
        gpu=gpu,
        shuffle=shuffle,
    )
    data = GPTData(GPTDataConfig.from_dict(config), distributed_config, vocab_size, sequence_length)
    data.setup(distributed, samples_per_phase, cache_directory)
    with NoAutoValidate():
        batch_config = BatchConfig(batch_size=1, sequence_length=sequence_length)
    batch_config.setup(distributed_config)
    batch_config.validate()
    tokens = {
        phase: torch.stack(
            [batch.token_ids[0] for batch in data.get_iterator(batch_config, phase, consumed_samples=0, num_workers=0)]
        )
        for phase, samples in samples_per_phase.items()
    }
    for phase, expected_samples_ in expected_samples.items():
        print("jerbn", phase, tokens[phase].tolist())
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
    Assert.eq(sizes.sum(), num_tokens)
    Assert.all_equal(
        [len(dataset.get(i).token_ids) for i in range(min(len(dataset), 100))], sizes[: min(len(dataset), 100)]
    )
    for i, expected_sample in expected_samples.items():
        Assert.all_equal(dataset.get(i).token_ids, np.array(expected_sample, dtype=np.uint16))
    if loss_masking_spans:
        for i, loss_masking_span in loss_masking_spans.items():
            print("AAAAAA", dataset.get(i, use_loss_masking_spans=True).loss_masking_spans, loss_masking_spans[i])
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
    num_tokens = sampled._num_samples * sampled._sequence_length + 1
    all_tokens = np.full(sampled._num_samples * sampled._sequence_length + 1, -1, dtype=np.int64)
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
        all_tokens[index * sampled._sequence_length : (index + 1) * sampled._sequence_length + 1]
        for index in range(sampled._num_samples)
    ]
    token_ids = [sampled[i].token_ids for i in range(len(sampled))]

    Assert.all_equal(token_ids, validate_samples)
    if expected_samples is not None:
        Assert.all_equal(token_ids, expected_samples)
    return token_ids
