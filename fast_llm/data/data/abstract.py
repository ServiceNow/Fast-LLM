import abc
import pathlib
import typing

from fast_llm.config import Configurable
from fast_llm.data.data.config import DataConfig
from fast_llm.data.dataset.config import SamplingParameters
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.schedule.config import BatchConfig

if typing.TYPE_CHECKING:
    from fast_llm.engine.distributed.distributed import Distributed


class Data[ConfigType: DataConfig](Configurable[ConfigType], abc.ABC):
    _distributed: "Distributed"
    _sampling_parameters: dict[str, SamplingParameters]
    _cache_directory: pathlib.Path | None

    def __init__(self, config: DataConfig, distributed_config: DistributedConfig) -> None:
        super().__init__(config)
        self._distributed_config = distributed_config

    # TODO: Improve interface
    def setup(
        self,
        distributed: "Distributed",
        sampling_parameters: dict[str, SamplingParameters],
        cache_directory: pathlib.Path,
        timeout: float | None = None,
    ) -> None:
        self._distributed = distributed
        self._sampling_parameters = sampling_parameters
        self._cache_directory = cache_directory

    @property
    def distributed(self):
        return self._distributed

    @abc.abstractmethod
    def get_iterator(
        self,
        batch_config: BatchConfig,
        dataset_name: str,
        *,
        consumed_samples: int,
        num_workers: int,
        prefetch_factor: int | None = None,
        timeout: float = 60,
    ) -> typing.Iterator[typing.Any]:
        pass
