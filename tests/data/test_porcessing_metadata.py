import tempfile
import pathlib

from fast_llm.data.preparator.gpt_memmap.config import GPTMemmapDatasetPreparatorConfig
from fast_llm.data.preparator.gpt_memmap.prepare import GPTMemmapDatasetPreparator


def get_prep(output_path: str, dataset_path_name: str) -> GPTMemmapDatasetPreparator:
    config = GPTMemmapDatasetPreparatorConfig.from_dict(
        {
            "output_path": output_path,
            "dataset": {"path": dataset_path_name},
            "tokenizer": {"path": "no_tokenizer"},
        },
        {},
    )
    return config.get_dataset_preparator_class()(config=config)


def test_existing_metadata_hf_hub_dataset():
    with tempfile.TemporaryDirectory(suffix="test") as local_folder:
        prep = get_prep(local_folder, "lhoestq/demo1")
        prep._save_croissant_metadata()
        assert (pathlib.Path(local_folder) / "croissant.json").is_file()


def test_absent_metadata_hf_hub_dataset():
    with tempfile.TemporaryDirectory(suffix="test") as local_folder:
        prep = get_prep(local_folder, "allenai/dolma")
        prep._save_croissant_metadata()
        assert not (pathlib.Path(local_folder) / "croissant.json").is_file()


def test_existing_metadata_local_dataset():
    with tempfile.TemporaryDirectory(suffix="dataset") as dataset_folder:
        (pathlib.Path(dataset_folder) / "croissant.json").touch()
        with tempfile.TemporaryDirectory(suffix="test") as local_folder:
            prep = get_prep(local_folder, dataset_folder)
            prep._save_croissant_metadata()
            assert (pathlib.Path(local_folder) / "croissant.json").is_file()


def test_absent_metadata_local_dataset():
    with tempfile.TemporaryDirectory(suffix="dataset") as dataset_folder:
        with tempfile.TemporaryDirectory(suffix="test") as local_folder:
            prep = get_prep(local_folder, dataset_folder)
            prep._save_croissant_metadata()
            assert not (pathlib.Path(local_folder) / "croissant.json").is_file()
