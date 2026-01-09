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
from fast_llm.data.sample.abstract import NullReaderConfig  # isort: skip
from fast_llm.data.sample.language_model import LanguageModelReaderConfig  # isort: skip
from fast_llm.data.sample.patch import PatchReaderConfig  # isort: skip
from fast_llm.data.sample.range import RangeReaderConfig  # isort: skip
from fast_llm.data.sample.token import TokenReaderConfig  # isort: skip
