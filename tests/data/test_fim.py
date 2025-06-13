from fast_llm.data.config import TokenizerConfig
from fast_llm.data.dataset.gpt.config import GPTFimSampledDatasetConfig
from fast_llm.data.tokenizer import Tokenizer
from tests.data.common import (
    compare_sampled_dataset,
    get_dataset_config,
    get_sampling_data,
    get_test_data_and_compare_samples,
)
from tests.utils.dataset import DATASET_PREFIX, TOKENIZER_PATH, get_test_dataset

GPT_FIM_SAMPLES = [
    [4709, 819, 79, 207, 277, 1790],
    [1790, 80, 6506, 1735, 542, 88],
    [86, 89, 7876, 80, 49152, 87],
    [80, 207, 567, 498, 89, 207],
    [207, 4700, 549, 79, 417, 3036],
    [86, 89, 1178, 49152, 87, 49152],
    [86, 49152, 1178, 64, 89, 900],
    [86, 49152, 89, 542, 395, 89],
]

GPT_FIM_SAMPLES_LEGACY = [
    [1725, 74, 207, 1635, 4440, 2774],
    [359, 489, 4266, 2052, 5351, 80],
    [86, 49152, 89, 22255, 1073, 79],
    [8008, 498, 71, 727, 80, 315],
    [2210, 8179, 73, 2582, 897, 1178],
    [86, 89, 88, 49152, 87, 49152],
    [86, 49152, 83, 744, 89, 64],
    [86, 89, 1461, 49152, 87, 49152],
]


def test_gpt_fim():
    # Make sure the FIM wrapper works in a simple case and check for unintended changes in behavior.
    get_test_dataset()
    # The test tokenizer doesn't have fim tokens, so we work around it.
    sampling_config = get_sampling_data(
        8,
        sequence_length=5,
        tokenizer=Tokenizer(TokenizerConfig.from_dict({"path": TOKENIZER_PATH})),
        vocab_size=49157,
    )
    sampled = get_dataset_config(
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
    compare_sampled_dataset(sampled, GPT_FIM_SAMPLES)


def test_gpt_fim_data():
    get_test_data_and_compare_samples(
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
        8,
        sequence_length=5,
        expected_samples=GPT_FIM_SAMPLES,
        vocab_size=49157,
    )


def test_gpt_fim_data_legacy():
    get_test_data_and_compare_samples(
        {
            "format": "list",
            "path": [str(DATASET_PREFIX)],
            "fim": {"rate": 0.5, "prefix_token": "w", "middle_token": "x", "pad_token": "y", "suffix_token": "z"},
            "tokenizer": {"path": TOKENIZER_PATH},
            "split": [1, 0, 0],
        },
        8,
        sequence_length=5,
        expected_samples=GPT_FIM_SAMPLES_LEGACY,
        legacy=True,
        vocab_size=49157,
    )
