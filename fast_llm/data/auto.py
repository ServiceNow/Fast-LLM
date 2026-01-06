"""
Import these submodules to ensure classes are added to the dynamic class registry.
"""

from fast_llm.data.dataset.config import (  # isort: skip
    BlendedDatasetConfig,
    ConcatenatedDatasetConfig,
    DatasetSliceConfig,
    MemmapDatasetConfig,
    SampledDatasetUpdateConfig,
)
from fast_llm.data.dataset.gpt.config import (  # isort: skip
    GPTDatasetFromFileConfig,
    GPTFimSampledDatasetConfig,
    GPTRandomDatasetConfig,
)
from fast_llm.data.preparator.dataset_discovery.config import DatasetDiscoveryConfig  # isort: skip
from fast_llm.data.preparator.gpt_memmap.config import GPTMemmapDatasetPreparatorConfig  # isort: skip
