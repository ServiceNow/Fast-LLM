import pathlib
import typing

import numpy as np

from fast_llm.config import NoAutoValidate
from fast_llm.data.data.gpt.config import GPTDataConfig, GPTSamplingDefaultConfig
from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.gpt.config import GPTSampledDatasetConfig, GPTSamplingData, ShufflingType
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


def get_test_data_and_samples(
    config: dict,
    samples_per_phase: dict[PhaseType, int],
    seed: int = 54983,
    cache_directory: pathlib.Path | None = None,
    sequence_length: int = 512,
    vocab_size=TEST_VOCAB_SIZE,
    expected_samples=None,
):
    distributed_config = DistributedConfig(seed=seed)
    distributed = Distributed(distributed_config, use_cpu=True)
    data = GPTData(GPTDataConfig.from_dict(config), distributed_config, vocab_size, sequence_length)
    data.setup(distributed, samples_per_phase, cache_directory)
    with NoAutoValidate():
        batch_config = BatchConfig(batch_size=1, sequence_length=sequence_length)
    batch_config.setup(distributed_config)
    batch_config.validate()
    samples = {
        phase: [batch[0] for batch in data.get_iterator(batch_config, phase, consumed_samples=0, num_workers=0)]
        for phase, samples in samples_per_phase.items()
    }
    if expected_samples:
        for phase, expected_samples_ in expected_samples.items():
            Assert.all_equal(
                np.stack(samples[phase]),
                np.array(expected_samples_),
            )
    return data, samples


def _check_sampling(self: GPTSampledIndexedDataset, expected_samples):
    """
    Compare `GPTSampledIndexedDataset` sampling against a more basic approach
    """
    num_tokens = self._num_samples * self._sequence_length + 1
    all_tokens = np.full(self._num_samples * self._sequence_length + 1, -1, dtype=np.int64)
    unshuffled_epochs = div(self._unshuffled_documents, self._documents_per_epoch)
    document_sampling = np.concatenate(
        (
            np.tile(np.arange(self._documents_per_epoch, dtype=self._document_shuffling), unshuffled_epochs),
            self._document_shuffling,
        )
    )
    seen_tokens = 0
    for document_index in document_sampling:
        document = self._indexed_dataset.get(document_index)
        all_tokens[seen_tokens : seen_tokens + len(document)] = document[: num_tokens - seen_tokens - len(document)]
        seen_tokens += len(document)
        if seen_tokens >= num_tokens:
            break

    samples = [self[index] for index in range(len(self))]
    Assert.all_equal(
        [
            all_tokens[index * self._sequence_length : (index + 1) * self._sequence_length + 1]
            for index in range(self._num_samples)
        ],
        samples,
    )

    if expected_samples:
        Assert.eq(len(self), len(expected_samples))
        Assert.all_equal(samples, expected_samples)
