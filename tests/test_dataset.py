import pathlib

import numpy as np

from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.abstract import PhaseSplits
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.utils import Assert
from tests.common import DATASET_PREFIX, TEST_RESULTS_PATH, TEST_VOCAB_SIZE, get_test_dataset


def get_test_data(
    config: dict,
    samples_per_phase: dict[PhaseType, int],
    cache_directory: pathlib.Path | None = None,
    sequence_length: int = 512,
):
    distributed_config = DistributedConfig()
    distributed = Distributed(distributed_config, use_cpu=True)
    data = GPTData(GPTDataConfig.from_dict(config), distributed_config, TEST_VOCAB_SIZE, sequence_length)
    data.setup(distributed, PhaseSplits[int](samples_per_phase), cache_directory)
    return data


DATASET_CACHE = TEST_RESULTS_PATH / "dataset" / "cache"


def get_test_datasets(
    config: dict,
    samples_per_phase: dict[PhaseType, int],
    cache_directory: pathlib.Path | None = None,
    sequence_length: int = 512,
):
    return get_test_data({"dataset": config}, samples_per_phase, cache_directory, sequence_length)._datasets


def test_dummy_dataset():
    datasets = get_test_datasets(
        {"type": "dummy"},
        {PhaseType.training: 7, PhaseType.test: 4},
    )
    Assert.eq(datasets.keys(), {PhaseType.training, PhaseType.test})
    train = datasets[PhaseType.training]
    Assert.eq(len(train), 7)
    assert all(np.all(train[i] == train._dataset._dummy_sample) for i in range(7))
    test = datasets[PhaseType.test]
    Assert.eq(len(test), 4)
    assert all(np.all(test[i] == test._dataset._dummy_sample) for i in range(4))


def test_memmap_dataset():
    get_test_dataset()
    dataset = get_test_datasets(
        {"type": "memmap", "path": DATASET_PREFIX},
        {PhaseType.training: 1},
        sequence_length=5,
    )[PhaseType.training]
    Assert.eq(len(dataset), 5)
    raise AssertionError()
