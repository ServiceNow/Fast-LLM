import json
import pathlib
import tempfile

import numpy as np
import pytest
import torch

from fast_llm.data.dataset.config import IndexedDatasetConfig
from fast_llm.data.dataset.gpt.config import GPTSamplingParameters
from fast_llm.data.dataset.memmap import MemmapDataset
from fast_llm.data.preparator.gpt_memmap.config import MEMMAP_DTYPES, GPTMemmapDatasetPreparatorConfig
from fast_llm.data.preparator.gpt_memmap.prepare import GPTMemmapDatasetPreparator
from fast_llm.data.sample.language_model import LanguageModelSample
from fast_llm.utils import Assert
from tests.data.common import MockGPTMemmapDatasetConfig  # Noqa


def get_preparator(output_path: str, dataset_path_name: str) -> GPTMemmapDatasetPreparator:
    config = GPTMemmapDatasetPreparatorConfig.from_dict(
        {
            "output_path": output_path,
            "dataset": {"path": dataset_path_name},
            "tokenizer": {"path": "no_tokenizer"},
        },
        {},
    )
    return config.get_dataset_preparator_class()(config=config)


@pytest.mark.parametrize("dtype", MEMMAP_DTYPES.values())
def test_write_memmap_dataset(dtype):
    documents = [
        (torch.from_numpy(np.random.randint(1000, size=np.random.randint(1, 100)).astype(dtype)), None, None, None)
        for _ in range(100)
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        prefix = pathlib.Path(temp_dir)
        MemmapDataset.write_dataset(prefix=prefix, documents=documents)
        dataset = MemmapDataset(name="foo", prefix=prefix)
        for i, (tokens, _, _, _) in enumerate(documents):
            Assert.all_equal(dataset.get_document(i).tokens.tokens, tokens.to(torch.int64))


def _generate_valid_span(max_seq_length):
    return np.sort(np.random.choice(np.arange(0, max_seq_length - 1), size=2, replace=False)).tolist()


@pytest.mark.parametrize("dtype", MEMMAP_DTYPES.values())
def test_write_memmap_preference_dataset(dtype):
    documents = [
        (
            torch.from_numpy(np.random.randint(1000, size=100).astype(dtype)),
            None,
            _generate_valid_span(100),
            _generate_valid_span(100),
        )
        for _ in range(50)
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        prefix = pathlib.Path(temp_dir)
        MemmapDataset.write_dataset(prefix=prefix, documents=documents)
        dataset = MemmapDataset(name="foo", prefix=prefix)
        parameters = GPTSamplingParameters(
            num_samples=0, sequence_length=0, vocab_size=0, use_preference_loss_spans=True
        )
        for i, (token_ids, _, (chosen_begin, chosen_end), (rejected_begin, rejected_end)) in enumerate(documents):
            document = dataset.get_document(i, parameters=parameters)
            Assert.all_equal(document.tokens.tokens, token_ids.to(torch.int64))
            Assert.eq(document.chosen_spans.ranges, [(chosen_begin, chosen_end + 1)])
            Assert.eq(document.rejected_spans.ranges, [(rejected_begin, rejected_end + 1)])


def test_load_metadata_from_hub():
    with tempfile.TemporaryDirectory(suffix="test") as local_folder:
        get_preparator(local_folder, "lhoestq/demo1")._save_croissant_metadata()
        croissant_path = pathlib.Path(local_folder) / "croissant.json"
        assert croissant_path.is_file()
        metadata = json.load(croissant_path.open("r"))
        assert metadata["url"] == "https://huggingface.co/datasets/lhoestq/demo1"


def test_absent_metadata_from_hub():
    with tempfile.TemporaryDirectory(suffix="test") as local_folder:
        get_preparator(local_folder, "allenai/dolma")._save_croissant_metadata()
        assert not (pathlib.Path(local_folder) / "croissant.json").is_file()


def test_load_metadata_local():
    with (
        tempfile.TemporaryDirectory(suffix="dataset") as dataset_folder,
        tempfile.TemporaryDirectory(suffix="test") as local_folder,
    ):
        metadata = {"name": "test"}
        json.dump(metadata, (pathlib.Path(dataset_folder) / "croissant.json").open("w"))
        get_preparator(local_folder, dataset_folder)._save_croissant_metadata()
        croissant_path = pathlib.Path(local_folder) / "croissant.json"
        assert croissant_path.is_file()
        assert json.loads(croissant_path.open("r").read()) == metadata


def test_absent_metadata_local():
    with (
        tempfile.TemporaryDirectory(suffix="dataset") as dataset_folder,
        tempfile.TemporaryDirectory(suffix="test") as local_folder,
    ):
        get_preparator(local_folder, dataset_folder)._save_croissant_metadata()
        assert not (pathlib.Path(local_folder) / "croissant.json").is_file()


DATASET_DICT_0 = {
    "type": "mock_memmap",
    "num_documents": 500,
    "num_tokens_per_document": 300,
}
DATASET_DICT_1 = {
    "type": "mock_memmap",
    "num_documents": 1500,
    "num_tokens_per_document": 100,
}


def test_split_dataset():
    dataset_config_0 = IndexedDatasetConfig[LanguageModelSample].from_dict(DATASET_DICT_0.copy())
    config = GPTMemmapDatasetPreparator._split_and_blend_dataset_configs(
        [dataset_config_0],
        {"training": 3, "validation": 1},
        pathlib.Path("."),
    )
    config = {key: value.to_dict() for key, value in config.items()}

    Assert.eq(
        config,
        {
            "training": {
                "type": "slice",
                "dataset": dataset_config_0.to_dict(),
                "begin": 0,
                "end": 0.75,
            },
            "validation": {
                "type": "slice",
                "dataset": dataset_config_0.to_dict(),
                "begin": 0.75,
                "end": 1,
            },
        },
    )


def test_split_datasets_0():
    dataset_config_0 = IndexedDatasetConfig[LanguageModelSample].from_dict(DATASET_DICT_0.copy())
    dataset_config_1 = IndexedDatasetConfig[LanguageModelSample].from_dict(DATASET_DICT_1.copy())
    config = GPTMemmapDatasetPreparator._split_and_blend_dataset_configs(
        [dataset_config_0, dataset_config_1],
        {"training": 1, "validation": 1},
        pathlib.Path("."),
    )
    config = {key: value.to_dict() for key, value in config.items()}

    Assert.eq(
        config,
        {
            "training": dataset_config_0.to_dict(),
            "validation": dataset_config_1.to_dict(),
        },
    )


def test_split_datasets_1():
    dataset_config_0 = IndexedDatasetConfig[LanguageModelSample].from_dict(DATASET_DICT_0.copy())
    dataset_config_1 = IndexedDatasetConfig[LanguageModelSample].from_dict(DATASET_DICT_1.copy())
    config = GPTMemmapDatasetPreparator._split_and_blend_dataset_configs(
        [dataset_config_0, dataset_config_1], {"training": 3, "validation": 1}, pathlib.Path(".")
    )
    config = {key: value.to_dict() for key, value in config.items()}

    Assert.eq(
        config,
        {
            "training": {
                "type": "blended",
                "datasets": [
                    dataset_config_0.to_dict(),
                    {
                        "type": "slice",
                        "dataset": dataset_config_1.to_dict(),
                        "begin": 0,
                        "end": 0.5,
                    },
                ],
                "weights": [2 / 3, 1 / 3],
            },
            "validation": {
                "type": "slice",
                "dataset": dataset_config_1.to_dict(),
                "begin": 0.5,
                "end": 1,
            },
        },
    )
