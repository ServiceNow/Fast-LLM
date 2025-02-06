from fast_llm.data.config import TokenizerConfig
from fast_llm.data.dataset.gpt.config import GPTFimSampledDatasetConfig
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.distributed.config import PhaseType
from tests.common import DATASET_PREFIX, TOKENIZER_PATH, get_test_dataset
from tests.data.common import (
    compare_sampled_dataset,
    get_dataset_config,
    get_sampling_config,
    get_test_data_and_compare_samples,
)

GPT_FIM_SAMPLES = [
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
    sampling_config = get_sampling_config(
        8, sequence_length=5, tokenizer=Tokenizer(TokenizerConfig.from_dict({"path": TOKENIZER_PATH}))
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
        {PhaseType.training: 8},
        sequence_length=5,
        expected_samples={PhaseType.training: GPT_FIM_SAMPLES},
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
        {PhaseType.training: 8},
        sequence_length=5,
        expected_samples={PhaseType.training: GPT_FIM_SAMPLES},
    )
