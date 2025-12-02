from fast_llm.data.dataset.gpt.config import GPTFimSampledDatasetConfig
from tests.data.common import (
    compare_sampled_dataset,
    get_dataset_config,
    get_sampling_data,
    get_test_data_and_compare_samples,
)
from tests.utils.dataset import get_common_test_dataset
from tests.utils.global_variables import TOKENIZER_PATH

GPT_FIM_SAMPLES = [
    [46, 10, 819, 19, 45, 88],
    [45, 69, 17, 86, 38826, 15],
    [86, 89, 32348, 64, 49152, 87],
    [64, 17, 93, 78, 40, 1793],
    [1793, 1, 1746, 38, 27, 58],
    [86, 89, 37, 92, 76, 49152],
    [86, 49152, 76, 29, 19, 89],
    [86, 49152, 46, 83, 17211, 1],
]


def test_gpt_fim():
    # Make sure the FIM wrapper works in a simple case and check for unintended changes in behavior.
    _, config, _ = get_common_test_dataset()
    # The test tokenizer doesn't have fim tokens, so we work around it.
    sampling_config = get_sampling_data(8, sequence_length=5)
    sampled = get_dataset_config(
        dataset_config := {
            "type": "fim",
            "dataset": config,
            "tokenizer": {"path": TOKENIZER_PATH},
            "rate": 0.5,
            "prefix_token": "w",
            "middle_token": "x",
            "pad_token": "y",
            "suffix_token": "z",
        },
        GPTFimSampledDatasetConfig,
    ).build_and_sample(sampling_config)
    compare_sampled_dataset(sampled, GPT_FIM_SAMPLES)

    get_test_data_and_compare_samples(
        {"datasets": {"training": dataset_config}},
        8,
        sequence_length=5,
        expected_samples=GPT_FIM_SAMPLES,
    )
