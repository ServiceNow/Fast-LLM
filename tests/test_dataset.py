import pathlib
import typing

import numpy as np

from fast_llm.data.dataset.gpt.config import (
    GPTDummyDatasetConfig,
    GPTMemmapDatasetConfig,
    GPTSampledDatasetConfig,
    GPTSamplingConfig,
)
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.utils import Assert
from tests.common import DATASET_PREFIX, TEST_RESULTS_PATH, TEST_VOCAB_SIZE

DATASET_CACHE = TEST_RESULTS_PATH / "dataset" / "cache"


def get_sampling_config(
    num_samples: int,
    *,
    seed: int = 54983,
    cache_directory: pathlib.Path | None = None,
    verbose=True,
    distributed: Distributed = Distributed(DistributedConfig(), use_cpu=True),
    phase: PhaseType = PhaseType.training,
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


def _get_dataset_config[T: GPTSampledDatasetConfig](config: dict[str, typing.Any], cls: type[T]) -> T:
    dataset_config = GPTSampledDatasetConfig.from_dict(config)
    Assert.custom(isinstance, dataset_config, cls)
    return typing.cast(cls, dataset_config)


def test_dummy_dataset():
    # TODO: Update
    config = _get_dataset_config({"type": "dummy"}, GPTDummyDatasetConfig)
    sampled = config.build_and_sample(get_sampling_config(4))
    Assert.eq(len(sampled), 4)
    assert all(np.all(sampled[i] == sampled._dataset._dummy_sample) for i in range(4))


def test_memmap_dataset():
    # TODO: Update
    config = _get_dataset_config({"type": "memmap", "path": DATASET_PREFIX}, GPTMemmapDatasetConfig)
    dataset = config.build()
    sampled = dataset.sample(get_sampling_config(8, sequence_length=5))
    print(np.stack([dataset.get(i) for i in range(5)]))
    print(np.stack([sampled[i] for i in range(5)]))
    raise AssertionError()
