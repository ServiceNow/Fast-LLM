import json

import datasets
import pytest

from fast_llm.data.dataset.config import BlendedDatasetConfig, MemmapDatasetConfig
from fast_llm.data.dataset.gpt.config import GPTDatasetFromFileConfig, GPTSamplingParameters
from fast_llm.data.dataset.memmap import MemmapDataset
from fast_llm.data.preparator.gpt_memmap.config import GPTMemmapDatasetPreparatorConfig
from fast_llm.data.preprocessing.tokenizer import TokenizerConfig
from fast_llm.utils import Assert
from tests.data.common import get_dataset_config
from tests.utils.dataset import (
    get_common_test_dataset,
    get_sharded_test_dataset,
    get_split_sharded_test_dataset,
    get_split_test_dataset,
)
from tests.utils.global_variables import DATASET_CACHE, TOKENIZER_NAME

COMMON_DATASET_LENGTH = 1000
COMMON_DATASET_TOKENS = 44883
COMMON_DATASET_TEXT = {
    27: "`s,uh",
    30: "@vo<CO_L",
    31: "]Xq?v73",
    77: "jqvU7O&yo",
    87: "Uzl,h",
}
COMMON_DATASET_SAMPLES = {
    27: [49152, 63, 82, 11, 27799, 49152],
    30: [49152, 31, 2327, 27, 1448, 62, 43, 49152],
    31: [49152, 60, 55, 80, 30, 85, 22, 18, 49152],
    77: [49152, 13736, 85, 52, 22, 46, 5, 11807, 49152],
    87: [49152, 52, 42536, 11, 71, 49152],
}


@pytest.mark.slow
def test_common_prepared_dataset():
    """
    We already test the dataset preparator indirectly through the test dataset (`get_test_dataset`).
    Here we verify the correctness of the prepared dataset directly and check for regressions.
    """
    path, config, hf_path = get_common_test_dataset()
    dataset = get_dataset_config(config, GPTDatasetFromFileConfig).build()
    dataset_from_shard = get_dataset_config(
        {"type": "memmap", "path": path / "shard_0_0.fast_llm_dataset"}, MemmapDatasetConfig
    ).build()

    hf_dataset = datasets.load_from_disk(hf_path)["train"]
    tokenizer = TokenizerConfig(path=TOKENIZER_NAME).get_tokenizer()

    # Check global stats.
    Assert.eq(len(dataset_from_shard), len(dataset), len(hf_dataset), COMMON_DATASET_LENGTH)
    Assert.eq(dataset_from_shard.num_tokens, dataset.num_tokens, COMMON_DATASET_TOKENS)

    for index in range(0, 200, 8):
        # Compare tokens for some samples.
        Assert.all_equal(
            dataset_from_shard.get_document(index).tokens.tokens,
            dataset.get_document(index).tokens.tokens,
            tokenizer.tokenize(hf_dataset[index]["text"]),
        )
        # Compare text.
        Assert.eq(
            tokenizer.detokenize(dataset.get_document(index).tokens.tokens, True, True),
            hf_dataset[index]["text"],
        )

    # Check some numerical values.
    for index in COMMON_DATASET_SAMPLES:
        Assert.eq(hf_dataset[index]["text"], COMMON_DATASET_TEXT[index])
        document = dataset.get_document(index, parameters=GPTSamplingParameters(num_samples=0, sequence_length=0))
        Assert.eq(document.tokens.tokens.tolist(), COMMON_DATASET_SAMPLES[index])


@pytest.mark.slow
def test_preparator_sharded():
    path, config, hf_path = get_sharded_test_dataset()

    dataset_config = get_dataset_config(config, GPTDatasetFromFileConfig)._load_config()
    Assert.custom(isinstance, dataset_config, BlendedDatasetConfig)
    Assert.eq(dataset_config.weights, [0.33003587104248827, 0.3455874161709333, 0.3243767127865784])
    datasets_ = [dataset_config_.build() for dataset_config_ in dataset_config.datasets]
    Assert.eq([len(dataset) for dataset in datasets_], lengths := [334, 333, 333])
    Assert.eq([dataset.num_tokens for dataset in datasets_], [14813, 15511, 14559])

    hf_dataset = datasets.load_from_disk(hf_path)["train"]
    tokenizer = TokenizerConfig(path=TOKENIZER_NAME).get_tokenizer()

    for index in range(0, 50, 8):
        Assert.all_equal(datasets_[0].get_document(index).tokens.tokens, tokenizer.tokenize(hf_dataset[index]["text"]))
        Assert.all_equal(
            datasets_[1].get_document(index).tokens.tokens, tokenizer.tokenize(hf_dataset[index + 334]["text"])
        )
        Assert.all_equal(
            datasets_[2].get_document(index).tokens.tokens, tokenizer.tokenize(hf_dataset[index + 667]["text"])
        )


@pytest.mark.slow
def test_preparator_split():
    path, config, hf_path = get_split_test_dataset()
    dataset_config = {
        split: get_dataset_config(split_config, GPTDatasetFromFileConfig)._load_config().to_dict()
        for split, split_config in config.items()
    }
    expected_config = {
        "training": {
            "type": "slice",
            "dataset": {"type": "memmap", "path": str(path / "shard_0_0.fast_llm_dataset")},
            "begin": 0,
            "end": 0.501,
        },
        "validation": {
            "type": "slice",
            "dataset": {"type": "memmap", "path": str(path / "shard_0_0.fast_llm_dataset")},
            "begin": 0.501,
            "end": 1,
        },
    }
    Assert.eq(dataset_config, expected_config)


@pytest.mark.slow
def test_preparator_split_sharded():
    path, config, hf_path = get_split_sharded_test_dataset()
    dataset_config = {
        split: get_dataset_config(split_config, GPTDatasetFromFileConfig)._load_config().to_dict()
        for split, split_config in config.items()
    }
    expected_config = {
        "training": {
            "type": "blended",
            "datasets": [
                {"type": "memmap", "path": str(path / "shard_0_0.fast_llm_dataset")},
                {
                    "type": "slice",
                    "dataset": {"type": "memmap", "path": str(path / "shard_0_1.fast_llm_dataset")},
                    "begin": 0,
                    "end": 0.5015015015015015,
                },
            ],
            "weights": [0.6602629819478494, 0.3397370180521507],
        },
        "validation": {
            "type": "blended",
            "datasets": [
                {
                    "type": "slice",
                    "dataset": {"type": "memmap", "path": str(path / "shard_0_1.fast_llm_dataset")},
                    "begin": 0.5015015015015015,
                    "end": 1,
                },
                {"type": "memmap", "path": str(path / "shard_0_2.fast_llm_dataset")},
            ],
            "weights": [0.3514344262295082, 0.6485655737704918],
        },
    }
    Assert.eq(dataset_config, expected_config)


@pytest.mark.slow
def test_dataset_preparator_from_hub():
    # TODO: Find or make a smaller dataset to speed things up.
    output_path = DATASET_CACHE / "preparator_from_hub"
    preparator_config = GPTMemmapDatasetPreparatorConfig.from_dict(
        {
            "dataset": {
                "path": "openai/gsm8k",
                "config_name": "main",
                "split": "test",
                "source_schema": {"text": "answer"},
            },
            "tokenizer": {"path": TOKENIZER_NAME},
            "output_path": output_path,
        }
    )
    preparator_config.run()

    assert (croissant_path := output_path / "croissant.json").is_file()
    Assert.eq(json.load(croissant_path.open("r"))["url"], "https://huggingface.co/datasets/openai/gsm8k")

    dataset = GPTDatasetFromFileConfig(path=output_path / "fast_llm_config.yaml").build()
    Assert.custom(isinstance, dataset, MemmapDataset)

    hf_dataset = datasets.load_dataset("openai/gsm8k", "main", split="test")
    tokenizer = preparator_config.tokenizer.get_tokenizer()

    Assert.eq(len(dataset), len(hf_dataset), 1319)
    Assert.eq(dataset.num_tokens, 179248)
    for index in range(0, 200, 8):
        Assert.eq(
            tokenizer.detokenize(dataset.get_document(index).tokens.tokens),
            f"<|endoftext|>{hf_dataset[index]["answer"]}<|endoftext|>",
        )
