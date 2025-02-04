import math
import pathlib
import typing

import numpy as np
import pytest

from fast_llm.config import NoAutoValidate
from fast_llm.data.config import TokenizerConfig
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.gpt.config import (
    GPTBlendedDatasetConfig,
    GPTConcatenatedDatasetConfig,
    GPTConcatenatedMemmapConfig,
    GPTDatasetSliceConfig,
    GPTFimSampledDatasetConfig,
    GPTMemmapDatasetConfig,
    GPTRandomDatasetConfig,
    GPTSampledDatasetConfig,
    GPTSamplingConfig,
)
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.utils import Assert, normalize_probabilities
from tests.common import (
    DATASET_CACHE,
    DATASET_PREFIX,
    DATASET_SAMPLING_CACHE,
    TEST_VOCAB_SIZE,
    TOKENIZER_PATH,
    get_test_concatenated_memmap_dataset,
    get_test_dataset,
)


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


def _get_dataset_config[T: GPTSampledDatasetConfig](config: dict[str, typing.Any], cls: type[T]) -> T:
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
    return data, samples


_DATASET_PREFIX_MIX_1 = DATASET_PREFIX.with_name("blended_mix_1")
_DATASET_PREFIX_MIX_CONCATENATED_MEMMAP = DATASET_CACHE / "concatenated_memmap"


def _get_test_dataset_mix_1():
    return get_test_dataset(prefix=_DATASET_PREFIX_MIX_1, seed=2345)


def _get_test_dataset_concatenated_memmap():
    return get_test_concatenated_memmap_dataset(_DATASET_PREFIX_MIX_CONCATENATED_MEMMAP, 4)


RANDOM_DATASET_EXPECTED_SAMPLES = [
    [3954, 4105, 6766, 859, 5494, 1675, 1303, 6913],
    [1654, 5701, 32, 1662, 7053, 3487, 1861, 1502],
    [5409, 6240, 5504, 7458, 7667, 3955, 3151, 3912],
    [5640, 6131, 7750, 2699, 1349, 2585, 7113, 6981],
]


def test_gpt_random_dataset():
    # Make sure the random dataset works and check for unintended changes in behavior.
    sampled = _get_dataset_config({"type": "random"}, GPTRandomDatasetConfig).build_and_sample(
        get_sampling_config(4, sequence_length=7)
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
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.training]),
        np.array(RANDOM_DATASET_EXPECTED_SAMPLES),
    )


def test_gpt_random_data_legacy():
    _, samples = get_test_data_and_samples({"format": "random"}, {PhaseType.training: 4}, sequence_length=7)
    Assert.all_equal(
        np.stack(samples[PhaseType.training]),
        np.array(RANDOM_DATASET_EXPECTED_SAMPLES),
    )


# Most documents are too long to write here, we test a few known short ones.
MEMMAP_DATASET_EXPECTED_LENGTH = 6153
MEMMAP_DATASET_EXPECTED_TOKENS = 508327
MEMMAP_DATASET_EXPECTED_SAMPLES = {
    9: [],
    10: [80, 85, 4295, 4182, 489, 727, 84, 698, 1197, 583],
    13: [78, 727, 74, 317, 1358, 89],
    15: [78],
}


@pytest.mark.parametrize("cache_directory", (None, pathlib.Path(DATASET_SAMPLING_CACHE) / "test_memmap"))
def test_gpt_memmap(cache_directory):
    # Make sure the memmap dataset works and check for unintended changes in behavior.
    get_test_dataset()
    dataset = _get_dataset_config({"type": "memmap", "path": DATASET_PREFIX}, GPTMemmapDatasetConfig).build()
    Assert.eq(len(dataset), MEMMAP_DATASET_EXPECTED_LENGTH)
    sizes = dataset.get_document_sizes()
    Assert.eq(sizes.sum(), MEMMAP_DATASET_EXPECTED_TOKENS)
    Assert.all_equal([len(dataset.get(i)) for i in range(100)], sizes[:100])
    for i, sample in MEMMAP_DATASET_EXPECTED_SAMPLES.items():
        Assert.all_equal(dataset.get(i), np.array(sample, dtype=np.uint16))


GPT_SAMPLED_EXPECTED_SAMPLES = [
    [1725, 74, 207, 1635, 4440, 2774],
    [359, 489, 4266, 2052, 5351, 80],
    [374, 7534, 87, 1073, 79, 480],
    [8008, 498, 71, 727, 80, 315],
    [2210, 8179, 73, 2582, 897, 1178],
    [409, 5091, 328, 1378, 5483, 88],
    [83, 4457, 3316, 333, 489, 317],
    [330, 155, 2449, 1136, 1106, 5370],
]


def test_gpt_sampled():
    # Make sure the memmap dataset works and check for unintended changes in behavior.
    get_test_dataset()
    sampled = _get_dataset_config({"type": "memmap", "path": DATASET_PREFIX}, GPTMemmapDatasetConfig).build_and_sample(
        get_sampling_config(8, sequence_length=5)
    )
    Assert.eq(len(sampled), 8)
    Assert.all_equal(
        np.stack([sampled[i] for i in range(8)]),
        np.array(GPT_SAMPLED_EXPECTED_SAMPLES),
    )


def test_gpt_sampled_data():
    get_test_dataset()
    _, samples = get_test_data_and_samples(
        {
            "datasets": {
                "Training": {
                    "type": "memmap",
                    "path": DATASET_PREFIX,
                }
            }
        },
        {PhaseType.training: 8},
        sequence_length=5,
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.training]),
        np.array(GPT_SAMPLED_EXPECTED_SAMPLES),
    )


def test_gpt_sampled_data_legacy():
    _, samples = get_test_data_and_samples(
        {"format": "list", "path": [str(DATASET_PREFIX)], "split": [1, 0, 0]},
        {PhaseType.training: 8},
        sequence_length=5,
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.training]),
        np.array(GPT_SAMPLED_EXPECTED_SAMPLES),
    )


GPT_CONCATENATED_EXPECTED_SAMPLES = [
    [243, 498, 7172, 777, 306, 74],
    [821, 6042, 89, 977, 4797, 499],
    [387, 74, 330, 328, 1858, 484],
    [7722, 3069, 819, 4266, 304, 80],
    [80, 634, 4913, 373, 207, 1046],
    [72, 65, 5570, 73, 2210, 5514],
    [7983, 977, 4147, 4739, 890, 386],
    [5375, 275, 69, 771, 593, 8171],
]


def test_gpt_concatenate():
    # Make sure the dataset concatenation works and check for unintended changes in behavior.
    get_test_dataset()
    dataset = _get_dataset_config(
        {"type": "concatenated", "datasets": [{"type": "memmap", "path": DATASET_PREFIX} for _ in range(3)]},
        GPTConcatenatedDatasetConfig,
    ).build()
    Assert.eq(len(dataset), 3 * MEMMAP_DATASET_EXPECTED_LENGTH)
    sizes = dataset.get_document_sizes()
    Assert.eq(sizes.sum(), 3 * MEMMAP_DATASET_EXPECTED_TOKENS)
    for i in range(3):
        begin = i * MEMMAP_DATASET_EXPECTED_LENGTH
        Assert.all_equal([len(dataset.get(begin + i)) for i in range(100)], sizes[begin : begin + 100])
        for i, sample in MEMMAP_DATASET_EXPECTED_SAMPLES.items():
            Assert.all_equal(dataset.get(begin + i), np.array(sample, dtype=np.uint16))
    sampled = dataset.sample(get_sampling_config(8, sequence_length=5))
    Assert.eq(len(sampled), 8)
    Assert.all_equal(
        np.stack([sampled[i] for i in range(8)]),
        np.array(GPT_CONCATENATED_EXPECTED_SAMPLES),
    )


def test_gpt_concatenate_data():
    _, samples = get_test_data_and_samples(
        {
            "datasets": {
                "Training": {
                    "type": "concatenated",
                    "datasets": [{"type": "memmap", "path": DATASET_PREFIX} for _ in range(3)],
                }
            }
        },
        {PhaseType.training: 8},
        sequence_length=5,
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.training]),
        np.array(GPT_CONCATENATED_EXPECTED_SAMPLES),
    )


GPT_SLICE_EXPECTED_TRAINING_SAMPLES = [
    [2625, 76, 2625, 2639, 74, 243],
    [207, 481, 5546, 74, 414, 498],
    [74, 333, 1963, 310, 5337, 3628],
    [79, 2361, 80, 2012, 84, 480],
]

GPT_SLICE_EXPECTED_VALIDATION_SAMPLES = [
    [2352, 3687, 2311, 4900, 542, 3732],
    [2551, 5283, 900, 3140, 328, 68],
    [7979, 2283, 329, 727, 2740, 2818],
    [4117, 8056, 79, 1798, 243, 498],
    [243, 542, 387, 6476, 6686, 785],
    [95, 6641, 207, 279, 2304, 602],
    [89, 4446, 947, 293, 947, 1544],
    [243, 3712, 86, 476, 80, 2547],
]


def test_gpt_slice():
    # Make sure dataset splitting works and check for unintended changes in behavior.
    get_test_dataset()
    # samples[9:18]
    dataset = _get_dataset_config(
        {"type": "slice", "dataset": {"type": "memmap", "path": DATASET_PREFIX}, "begin": 0.0015, "end": 0.003},
        GPTDatasetSliceConfig,
    ).build()
    Assert.eq(len(dataset), 9)
    sizes = dataset.get_document_sizes()
    Assert.all_equal([len(dataset.get(i)) for i in range(9)], sizes[:9])
    for i, sample in MEMMAP_DATASET_EXPECTED_SAMPLES.items():
        Assert.all_equal(dataset.get(i - 9), np.array(sample, dtype=np.uint16))
    sampled = dataset.sample(get_sampling_config(8, sequence_length=5))
    Assert.eq(len(sampled), 8)
    Assert.all_equal(
        np.stack([sampled[i] for i in range(8)]),
        np.array(GPT_SLICE_EXPECTED_VALIDATION_SAMPLES),
    )


def test_gpt_slice_data():
    _, samples = get_test_data_and_samples(
        {
            "datasets": {
                "Training": {
                    "type": "slice",
                    "dataset": {"type": "memmap", "path": DATASET_PREFIX},
                    "begin": 0,
                    "end": 0.0015,
                },
                "Validation": {
                    "type": "slice",
                    "dataset": {"type": "memmap", "path": DATASET_PREFIX},
                    "begin": 0.0015,
                    "end": 0.003,
                },
                "Test": {
                    "type": "slice",
                    "dataset": {"type": "memmap", "path": DATASET_PREFIX},
                    "begin": 0.003,
                    "end": 1,
                },
            }
        },
        {PhaseType.training: 4, PhaseType.validation: 8, PhaseType.test: 5},
        sequence_length=5,
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.validation]),
        np.array(GPT_SLICE_EXPECTED_VALIDATION_SAMPLES),
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.training]),
        np.array(GPT_SLICE_EXPECTED_TRAINING_SAMPLES),
    )


def test_gpt_slice_data_legacy():
    get_test_dataset()
    _, samples = get_test_data_and_samples(
        {"format": "list", "path": [str(DATASET_PREFIX)], "split": [0.0015, 0.0015, 0.997]},
        {PhaseType.training: 4, PhaseType.validation: 8, PhaseType.test: 5},
        sequence_length=5,
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.validation]),
        np.array(GPT_SLICE_EXPECTED_VALIDATION_SAMPLES),
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.training]),
        np.array(GPT_SLICE_EXPECTED_TRAINING_SAMPLES),
    )


COMPOSED_DATASET_EXPECTED_LENGTH = 24806
COMPOSED_DATASET_EXPECTED_TOKENS = 2033639

COMPOSED_DATASET_EXPECTED_SAMPLES = {
    **MEMMAP_DATASET_EXPECTED_SAMPLES,
    6930: [65, 2327],
    11962: [7078, 2713, 1431],
    15958: [207],
    19362: [69],
    24098: [555, 668, 70],
}


GPT_COMPOSED_EXPECTED_SAMPLES = [
    [1411, 819, 6791, 7022, 285, 249],
    [329, 328, 512, 1985, 3069, 7838],
    [5158, 1023, 8171, 798, 1431, 313],
    [1073, 3917, 275, 480, 74, 1752],
    [207, 317, 269, 6662, 4357, 498],
    [74, 310, 277, 7091, 668, 367],
    [7828, 480, 89, 116, 4604, 69],
    [79, 6042, 577, 225, 207, 207],
]


def test_gpt_concatenated_memmap():
    # Make sure dataset splitting works and check for unintended changes in behavior.
    _get_test_dataset_concatenated_memmap()
    # samples[9:18]
    dataset = _get_dataset_config(
        {"type": "concatenated_memmap", "path": _DATASET_PREFIX_MIX_CONCATENATED_MEMMAP},
        GPTConcatenatedMemmapConfig,
    ).build()
    Assert.eq(len(dataset), COMPOSED_DATASET_EXPECTED_LENGTH)
    sizes = dataset.get_document_sizes()
    Assert.eq(sizes.sum(), COMPOSED_DATASET_EXPECTED_TOKENS)
    Assert.all_equal([len(dataset.get(i)) for i in range(0, len(dataset), 20)], sizes[::20])
    for i, sample in COMPOSED_DATASET_EXPECTED_SAMPLES.items():
        Assert.all_equal(dataset.get(i), np.array(sample, dtype=np.uint16))
    sampled = dataset.sample(get_sampling_config(8, sequence_length=5))
    Assert.eq(len(sampled), 8)
    print(np.stack([sampled[i] for i in range(8)]).tolist())
    Assert.all_equal(
        np.stack([sampled[i] for i in range(8)]),
        np.array(GPT_COMPOSED_EXPECTED_SAMPLES),
    )


def test_gpt_concatenated_memmap_data():
    _get_test_dataset_concatenated_memmap()
    _, samples = get_test_data_and_samples(
        {
            "datasets": {
                "Training": {
                    "type": "concatenated_memmap",
                    "path": _DATASET_PREFIX_MIX_CONCATENATED_MEMMAP,
                }
            }
        },
        {PhaseType.training: 8},
        sequence_length=5,
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.training]),
        np.array(GPT_COMPOSED_EXPECTED_SAMPLES),
    )


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
        get_sampling_config(num_samples),
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


GPT_BLENDED_EXPECTED_SAMPLES = [
    [1725, 74, 207, 1635, 4440, 2774],
    [359, 489, 4266, 2052, 5351, 80],
    [2066, 207, 6436, 2360, 2210, 6633],
    [374, 7534, 87, 1073, 79, 480],
    [8008, 498, 71, 727, 80, 315],
    [2210, 8179, 73, 2582, 897, 1178],
    [555, 3042, 83, 207, 498, 3373],
    [409, 5091, 328, 1378, 5483, 88],
]


def test_gpt_blended():
    # Make sure dataset blending works and check for unintended changes in behavior.
    get_test_dataset()
    _get_test_dataset_mix_1()
    sampled = _get_dataset_config(
        {
            "type": "blended",
            "datasets": [
                {"type": "memmap", "path": DATASET_PREFIX},
                {"type": "memmap", "path": _DATASET_PREFIX_MIX_1},
            ],
            "weights": [0.75, 0.25],
        },
        GPTBlendedDatasetConfig,
    ).build_and_sample(get_sampling_config(8, sequence_length=5))
    Assert.eq(len(sampled), 8)
    Assert.all_equal(
        np.stack([sampled[i] for i in range(8)]),
        np.array(GPT_BLENDED_EXPECTED_SAMPLES),
    )


def test_gpt_blended_data():
    get_test_dataset()
    _get_test_dataset_mix_1()
    _, samples = get_test_data_and_samples(
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
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.training]),
        np.array(GPT_BLENDED_EXPECTED_SAMPLES),
    )


GPT_BLENDED_LEGACY_EXPECTED_SAMPLES = [
    [1725, 74, 207, 1635, 4440, 2774],
    [359, 489, 4266, 2052, 5351, 80],
    [328, 80, 263, 890, 1797, 88],
    [374, 7534, 87, 1073, 79, 480],
    [8008, 498, 71, 727, 80, 315],
    [2210, 8179, 73, 2582, 897, 1178],
    [1852, 71, 776, 7878, 7390, 80],
    [409, 5091, 328, 1378, 5483, 88],
]


def test_gpt_blended_data_legacy():
    get_test_dataset()
    _get_test_dataset_mix_1()
    _, samples = get_test_data_and_samples(
        {
            "format": "list",
            "path": ["0.75", str(DATASET_PREFIX), "0.25", str(_DATASET_PREFIX_MIX_1)],
            "split": [1, 0, 0],
        },
        {PhaseType.training: 8},
        sequence_length=5,
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.training]),
        np.array(GPT_BLENDED_LEGACY_EXPECTED_SAMPLES),
    )


GPT_BLENDED_MIXED_EXPECTED_SAMPLES = [
    [1725, 74, 207, 1635, 4440, 2774],
    [916, 6683, 7685, 1277, 5106, 378],
    [359, 489, 4266, 2052, 5351, 80],
    [3359, 6803, 780, 4561, 669, 7878],
    [374, 7534, 87, 1073, 79, 480],
    [8008, 498, 71, 727, 80, 315],
    [6920, 2218, 2921, 3963, 7606, 6904],
    [2210, 8179, 73, 2582, 897, 1178],
]


def test_gpt_blended_mixed():
    # Make sure dataset blending works and check for unintended changes in behavior.
    get_test_dataset()
    sampled = _get_dataset_config(
        {
            "type": "blended",
            "datasets": [
                {"type": "memmap", "path": DATASET_PREFIX},
                {"type": "random"},
            ],
            "weights": [0.6, 0.4],
        },
        GPTBlendedDatasetConfig,
    ).build_and_sample(get_sampling_config(8, sequence_length=5))
    Assert.eq(len(sampled), 8)
    Assert.all_equal(
        np.stack([sampled[i] for i in range(8)]),
        np.array(GPT_BLENDED_MIXED_EXPECTED_SAMPLES),
    )


def test_gpt_blended_mixed_data():
    _, samples = get_test_data_and_samples(
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
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.training]),
        np.array(GPT_BLENDED_MIXED_EXPECTED_SAMPLES),
    )


GPT_FIM_EXPECTED_SAMPLES = [
    [1725, 74, 207, 1635, 4440, 2774],
    [359, 489, 4266, 2052, 5351, 80],
    [86, 89, 22255, 1073, 79, 480],
    [8008, 498, 71, 727, 80, 315],
    [2210, 8179, 73, 2582, 897, 1178],
    [86, 89, 88, 87, 409, 70],
    [86, 83, 744, 89, 64, 333],
    [86, 89, 1461, 87, 330, 7876],
]


def test_gpt_fim():
    # Make sure the FIM wrapper works in a simple case and check for unintended changes in behavior.
    get_test_dataset()
    # The test tokenizer doesn't have fim tokens, so we work around it.
    sampling_config = get_sampling_config(
        8, sequence_length=5, tokenizer=Tokenizer(TokenizerConfig.from_dict({"path": TOKENIZER_PATH}))
    )
    sampled = _get_dataset_config(
        {
            "type": "fim",
            "dataset": {"type": "memmap", "path": DATASET_PREFIX},
            "rate": 0.5,
            "prefix_token": "w",
            "middle_token": "x",
            "pad_token": "y",
            "suffix_token": "z",
        },
        GPTFimSampledDatasetConfig,
    ).build_and_sample(sampling_config)
    Assert.eq(len(sampled), 8)
    # TODO: Does this output make sense?
    Assert.all_equal(
        np.stack([sampled[i] for i in range(8)]),
        np.array(GPT_FIM_EXPECTED_SAMPLES),
    )


def test_gpt_fim_data():
    _, samples = get_test_data_and_samples(
        {
            "datasets": {
                "Training": {
                    "type": "fim",
                    "dataset": {"type": "memmap", "path": DATASET_PREFIX},
                    "rate": 0.5,
                    "prefix_token": "w",
                    "middle_token": "x",
                    "pad_token": "y",
                    "suffix_token": "z",
                }
            },
            "tokenizer": {"path": TOKENIZER_PATH},
        },
        {PhaseType.training: 8},
        sequence_length=5,
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.training]),
        np.array(GPT_FIM_EXPECTED_SAMPLES),
    )


def test_gpt_fim_data_legacy():
    _, samples = get_test_data_and_samples(
        {
            "format": "list",
            "path": [str(DATASET_PREFIX)],
            "fim": {"rate": 0.5, "prefix_token": "w", "middle_token": "x", "pad_token": "y", "suffix_token": "z"},
            "tokenizer": {"path": TOKENIZER_PATH},
            "split": [1, 0, 0],
        },
        {PhaseType.training: 8},
        sequence_length=5,
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.training]),
        np.array(GPT_FIM_EXPECTED_SAMPLES),
    )
