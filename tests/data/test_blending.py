import math

import numpy as np
import pytest

from fast_llm.data.dataset.gpt.config import GPTBlendedDatasetConfig
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert, normalize_probabilities
from tests.common import DATASET_CACHE, DATASET_PREFIX, get_test_dataset
from tests.data.common import (
    compare_sampled_dataset,
    get_dataset_config,
    get_sampling_data,
    get_test_data_and_compare_samples,
)

_DATASET_PREFIX_MIX_1 = DATASET_CACHE / "blended_mix_1" / "dataset"


def _get_test_dataset_mix_1():
    return get_test_dataset(prefix=_DATASET_PREFIX_MIX_1, seed=2345)


def _get_blending_alt(probs: list[float], num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    probs = np.array(probs)
    dataset_index = np.zeros(num_samples)
    sample_index = np.zeros(num_samples)
    sampled = np.zeros(len(probs))
    for idx in range(num_samples):
        error = probs * (idx + 1) - sampled
        dataset_index_ = np.argmax(error)
        dataset_index[idx] = dataset_index_
        sample_index[idx] = sampled[dataset_index_]
        sampled[dataset_index_] += 1
    return dataset_index, sample_index


GPT_BLENDED_SAMPLES = [
    [4709, 819, 79, 207, 277, 1790],
    [1790, 80, 6506, 1735, 542, 88],
    [4628, 7392, 920, 79, 1322, 387],
    [88, 4302, 269, 2794, 119, 80],
    [80, 207, 567, 498, 89, 207],
    [207, 4700, 549, 79, 417, 3036],
    [387, 4224, 87, 2713, 423, 324],
    [3036, 253, 207, 2968, 4536, 1178],
]

GPT_BLENDED_LEGACY_SAMPLES = [
    [1725, 74, 207, 1635, 4440, 2774],
    [359, 489, 4266, 2052, 5351, 80],
    [328, 80, 263, 890, 1797, 88],
    [374, 7534, 87, 1073, 79, 480],
    [8008, 498, 71, 727, 80, 315],
    [2210, 8179, 73, 2582, 897, 1178],
    [1852, 71, 776, 7878, 7390, 80],
    [409, 5091, 328, 1378, 5483, 88],
]

GPT_BLENDED_MIXED_SAMPLES = [
    [4709, 819, 79, 207, 277, 1790],
    [916, 6683, 7685, 1277, 5106, 378],
    [1790, 80, 6506, 1735, 542, 88],
    [3359, 6803, 780, 4561, 669, 7878],
    [88, 4302, 269, 2794, 119, 80],
    [80, 207, 567, 498, 89, 207],
    [6920, 2218, 2921, 3963, 7606, 6904],
    [207, 4700, 549, 79, 417, 3036],
]


@pytest.mark.parametrize(
    "probs",
    [
        # Two datasets
        [0.5, 0.5],
        [0.6, 0.4],
        [0.75, 0.25],
        [0.2, 0.8],
        # Irrational, not normalized.
        [math.pi, 2],
        # More datasets
        [0.3, 0.4, 0.3],
        [0.75, 0.05, 0.20],
        [0.3, 0.2, 0.4, 0.1],
        # Lots of datasets, not normalized.
        (np.arange(200) % 7 + 0.2).tolist(),
        # Useless but should still work.
        [1],
        [1, 0],
        [0, 1],
    ],
)
def test_blending(probs):
    num_samples = 100
    from fast_llm.data.dataset.blended import BlendedDataset

    dataset = BlendedDataset(
        "dataset",
        # Use a list of integers as a mock dataset, encoding both indexes in the sample.
        [list(range(i * num_samples, (i + 1) * num_samples)) for i, _ in enumerate(probs)],  # noqa
        probs,
        get_sampling_data(num_samples),
    )
    probs = normalize_probabilities(probs)
    samples = np.array([dataset[i] for i in range(num_samples)])
    dataset_index = samples // 100
    sample_index = samples % 100
    # Consistency checks, just in case the alt implementation is also wrong.
    for i, p in enumerate(probs):
        s = sample_index[dataset_index == i]
        # Samples for each dataset should be a sequence of natural numbers.
        Assert.all_equal(sorted(s), np.arange(len(s)))
        # And close enough to the target.
        Assert.leq(abs(len(s) - p * num_samples), 1)

    # Compare to the alternate implementation.
    dataset_index_alt, sample_index_alt = _get_blending_alt(probs, num_samples)
    samples_alt = sample_index_alt + dataset_index_alt * num_samples
    Assert.all_equal(samples, samples_alt)


def test_gpt_blended():
    # Make sure dataset blending works and check for unintended changes in behavior.
    get_test_dataset()
    _get_test_dataset_mix_1()
    sampled = get_dataset_config(
        {
            "type": "blended",
            "datasets": [
                {"type": "memmap", "path": DATASET_PREFIX},
                {"type": "memmap", "path": _DATASET_PREFIX_MIX_1},
            ],
            "weights": [0.75, 0.25],
        },
        GPTBlendedDatasetConfig,
    ).build_and_sample(get_sampling_data(8, sequence_length=5))
    compare_sampled_dataset(sampled, GPT_BLENDED_SAMPLES)


def test_gpt_blended_data():
    get_test_dataset()
    _get_test_dataset_mix_1()
    get_test_data_and_compare_samples(
        {
            "datasets": {
                "Training": {
                    "type": "blended",
                    "datasets": [
                        {"type": "memmap", "path": DATASET_PREFIX},
                        {"type": "memmap", "path": _DATASET_PREFIX_MIX_1},
                    ],
                    "weights": [0.75, 0.25],
                }
            }
        },
        {PhaseType.training: 8},
        sequence_length=5,
        expected_samples={PhaseType.training: GPT_BLENDED_SAMPLES},
    )


def test_gpt_blended_data_legacy():
    get_test_dataset()
    _get_test_dataset_mix_1()
    get_test_data_and_compare_samples(
        {
            "format": "list",
            "path": ["0.75", str(DATASET_PREFIX), "0.25", str(_DATASET_PREFIX_MIX_1)],
            "split": [1, 0, 0],
        },
        {PhaseType.training: 8},
        sequence_length=5,
        expected_samples={PhaseType.training: GPT_BLENDED_LEGACY_SAMPLES},
        legacy=True,
    )


def test_gpt_blended_mixed():
    # Make sure dataset blending works and check for unintended changes in behavior.
    get_test_dataset()
    sampled = get_dataset_config(
        {
            "type": "blended",
            "datasets": [
                {"type": "memmap", "path": DATASET_PREFIX},
                {"type": "random"},
            ],
            "weights": [0.6, 0.4],
        },
        GPTBlendedDatasetConfig,
    ).build_and_sample(get_sampling_data(8, sequence_length=5))
    compare_sampled_dataset(sampled, GPT_BLENDED_MIXED_SAMPLES)


def test_gpt_blended_mixed_data():
    get_test_data_and_compare_samples(
        {
            "datasets": {
                "Training": {
                    "type": "blended",
                    "datasets": [{"type": "memmap", "path": DATASET_PREFIX}, {"type": "random"}],
                    "weights": [0.6, 0.4],
                }
            }
        },
        {PhaseType.training: 8},
        sequence_length=5,
        expected_samples={PhaseType.training: GPT_BLENDED_MIXED_SAMPLES},
    )
