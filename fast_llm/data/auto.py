from fast_llm.data.preparator.gpt_memmap.config import GPTMemmapDatasetPreparatorConfig
from fast_llm.utils import Registry

dataset_preparator_registry = Registry(
    "DatasetPreparator",
    {
        dataset_preparator.preparator_name: dataset_preparator
        for dataset_preparator in [
            GPTMemmapDatasetPreparatorConfig,
        ]
    },
)
