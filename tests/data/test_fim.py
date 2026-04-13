from fast_llm.data.dataset.gpt.config import GPTFimSampledDatasetConfig
from tests.data.common import (
    compare_sampled_dataset,
    get_dataset_config,
    get_sampling_config,
    get_test_data_and_compare_samples,
)
from tests.utils.dataset import get_common_test_dataset
from tests.utils.global_variables import TOKENIZER_PATH

GPT_FIM_SAMPLES = [
    [46, 10, 721, 19, 45, 88],
    [45, 69, 17, 86, 92, 0],
    [86, 89, 31, 27, 50256, 87],
    [27, 0, 64, 17, 93, 78],
    [78, 3955, 43, 1, 1395, 38],
    [86, 89, 55, 93, 37, 50256],
    [86, 50256, 37, 92, 76, 89],
    [86, 89, 1, 50256, 87, 50256],
]


def test_gpt_fim(data_result_path):
    # Make sure the FIM wrapper works in a simple case and check for unintended changes in behavior.
    _, config, _, preprocessing = get_common_test_dataset()
    # The test tokenizer doesn't have fim tokens, so we work around it.
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
    ).build_and_sample(*get_sampling_config(8, sequence_length=5, preprocessing=preprocessing))
    compare_sampled_dataset(sampled, GPT_FIM_SAMPLES)

    get_test_data_and_compare_samples(
        {"datasets": {"training": dataset_config}},
        8,
        sequence_length=5,
        expected_samples=GPT_FIM_SAMPLES,
        preprocessing=preprocessing,
        cache_directory=data_result_path / "fim",
    )
