import numpy as np

from fast_llm.data.dataset.gpt.config import GPTRandomDatasetConfig
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert
from tests.data.common import get_dataset_config, get_sampling_data, get_test_data_and_samples

RANDOM_DATASET_EXPECTED_SAMPLES = [
    [3954, 4105, 6766, 859, 5494, 1675, 1303, 6913],
    [1654, 5701, 32, 1662, 7053, 3487, 1861, 1502],
    [5409, 6240, 5504, 7458, 7667, 3955, 3151, 3912],
    [5640, 6131, 7750, 2699, 1349, 2585, 7113, 6981],
]


def test_gpt_random_dataset():
    # Make sure the random dataset works and check for unintended changes in behavior.
    sampled = get_dataset_config({"type": "random"}, GPTRandomDatasetConfig).build_and_sample(
        get_sampling_data(4, sequence_length=7)
    )
    Assert.eq(len(sampled), 4)
    Assert.all_equal(
        np.stack([sampled[i] for i in range(4)]),
        np.array(RANDOM_DATASET_EXPECTED_SAMPLES),
    )


def test_gpt_random_data():
    _, samples = get_test_data_and_samples(
        {
            "datasets": {
                "Training": {
                    "type": "random",
                }
            }
        },
        {PhaseType.training: 4},
        sequence_length=7,
        expected_samples=RANDOM_DATASET_EXPECTED_SAMPLES,
    )


def test_gpt_random_data_legacy():
    _, samples = get_test_data_and_samples(
        {"format": "random"},
        {PhaseType.training: 4},
        sequence_length=7,
        expected_samples=RANDOM_DATASET_EXPECTED_SAMPLES,
    )
