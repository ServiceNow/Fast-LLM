import numpy as np

from fast_llm.data.config import TokenizerConfig
from fast_llm.data.dataset.gpt.config import GPTFimSampledDatasetConfig
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert
from tests.common import DATASET_PREFIX, TOKENIZER_PATH, get_test_dataset
from tests.data.common import get_dataset_config, get_sampling_data, get_test_data_and_samples

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
    sampling_config = get_sampling_data(
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
