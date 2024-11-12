from fast_llm.data.gpt.prepare import GPTDatasetPreparatorConfig
from fast_llm.utils import Registry

dataset_preparator_registry = Registry(
    "DatasetPreparator",
    {
        dataset_preparator.model_name: dataset_preparator
        for dataset_preparator in [
            GPTDatasetPreparatorConfig,
        ]
    },
)
