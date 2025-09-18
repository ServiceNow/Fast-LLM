from fast_llm.data.dataset.gpt.config import GPTDatasetFromFileConfig
from tests.data.common import compare_indexed_dataset, get_dataset_config
from tests.data.test_memmap import MEMMAP_DATASET_LENGTH, MEMMAP_DATASET_SAMPLES, MEMMAP_DATASET_TOKENS
from tests.utils.dataset import get_test_dataset
from tests.utils.global_variables import DATASET_PREFIX


def test_dataset_from_file():
    get_test_dataset()
    dataset_config = {"type": "file", "path": str(DATASET_PREFIX.parent.joinpath("fast_llm_config.yaml"))}
    dataset = get_dataset_config(dataset_config, GPTDatasetFromFileConfig).build()
    compare_indexed_dataset(dataset, MEMMAP_DATASET_LENGTH, MEMMAP_DATASET_TOKENS, MEMMAP_DATASET_SAMPLES)
