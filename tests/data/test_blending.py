import math

import numpy as np
import pytest

from fast_llm.data.dataset.config import BlendedDatasetConfig
from fast_llm.data.sample.language_model import LanguageModelSample
from fast_llm.utils import Assert, normalize_probabilities
from tests.data.common import (
    compare_sampled_dataset,
    get_dataset_config,
    get_sampling_data,
    get_test_data_and_compare_samples,
)
from tests.utils.dataset import get_alt_test_dataset, get_common_test_dataset


def _get_blending_alt(probs: list[float], num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    # Alternate implementation for blending.
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
    [49152, 46, 10, 819, 19, 45],
    [45, 69, 17, 86, 38826, 15],
    [49152, 83, 80, 20452, 45, 93],
    [15, 25, 51, 31, 32348, 64],
    [64, 17, 93, 78, 40, 1793],
    [1793, 1, 1746, 38, 27, 58],
    [93, 90, 39, 6, 75, 9],
    [58, 22885, 93, 37, 92, 76],
]

GPT_BLENDED_MIXED_SAMPLES = [
    [49152, 46, 10, 819, 19, 45],
    [25492, 15877, 37874, 8570, 31649, 15521],
    [45, 69, 17, 86, 38826, 15],
    [3359, 20945, 33437, 32454, 42084, 45942],
    [15, 25, 51, 31, 32348, 64],
    [64, 17, 93, 78, 40, 1793],
    [15112, 36731, 47864, 35586, 33356, 37537],
    [1793, 1, 1746, 38, 27, 58],
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
    _, config, _, preprocessing = get_common_test_dataset()
    _, alt_config, _, _ = get_alt_test_dataset()
    sampled = get_dataset_config(
        dataset_config := {
            "type": "blended",
            "datasets": [config, alt_config],
            "weights": [0.75, 0.25],
        },
        BlendedDatasetConfig[LanguageModelSample],
    ).build_and_sample(get_sampling_data(8, sequence_length=5, preprocessing=preprocessing))
    compare_sampled_dataset(sampled, GPT_BLENDED_SAMPLES)

    # Test in data.
    get_test_data_and_compare_samples(
        {"datasets": {"training": dataset_config}},
        8,
        sequence_length=5,
        expected_samples=GPT_BLENDED_SAMPLES,
        preprocessing=preprocessing,
    )


def test_gpt_blended_mixed():
    # Make sure dataset blending works and check for unintended changes in behavior.
    _, config, _, preprocessing = get_common_test_dataset()
    # Random dataset needs an explicit vocab size.
    preprocessing = preprocessing.from_dict(preprocessing, {"vocab_size": 50000})
    sampled = get_dataset_config(
        dataset_config := {
            "type": "blended",
            "datasets": [
                config,
                {"type": "random"},
            ],
            "weights": [0.6, 0.4],
        },
        BlendedDatasetConfig[LanguageModelSample],
    ).build_and_sample(get_sampling_data(8, sequence_length=5, preprocessing=preprocessing))
    compare_sampled_dataset(sampled, GPT_BLENDED_MIXED_SAMPLES)

    # Test in data.
    get_test_data_and_compare_samples(
        {"datasets": {"training": dataset_config}},
        8,
        sequence_length=5,
        expected_samples=GPT_BLENDED_MIXED_SAMPLES,
        preprocessing=preprocessing,
    )
