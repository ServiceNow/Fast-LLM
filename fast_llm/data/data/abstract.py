import abc
import pathlib
import typing

from fast_llm.config import Configurable
from fast_llm.data.data.config import DataConfig
from fast_llm.data.document.abstract import Batch, ModelInput
from fast_llm.data.document.config import BatchPreprocessingConfig
from fast_llm.engine.distributed.config import DistributedConfig

if typing.TYPE_CHECKING:
    from fast_llm.engine.distributed.distributed import Distributed


class Data[ConfigType: DataConfig](Configurable[ConfigType], abc.ABC):
    _distributed: "Distributed"
    # _sampling_parameters: dict[str, SamplingParameters]
    # _preprocessing: dict[str, PreprocessingConfig]
    _cache_directory: pathlib.Path | None
    _is_setup: bool = False

    def __init__(self, config: DataConfig, distributed_config: DistributedConfig) -> None:
        super().__init__(config)
        self._distributed_config = distributed_config

    # TODO: Improve interface
    def setup(self, cache_directory: pathlib.Path) -> None:
        self._cache_directory = cache_directory
        self._is_setup = True

    @abc.abstractmethod
    def sample_dataset(
        self,
        dataset_name: str,
        config: BatchPreprocessingConfig,
        num_samples: int,
    ) -> list[ModelInput]:
        pass

    def get_iterator(
        self,
        dataset_name: str,
        *,
        consumed_samples: int,
        num_workers: int,
        prefetch_factor: int | None = None,
        timeout: float = 60,
        preprocess: bool = True,
    ) -> typing.Iterator[list[ModelInput] | Batch]:
        pass
