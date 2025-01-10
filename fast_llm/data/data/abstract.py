import abc
import pathlib
import typing

from fast_llm.data.data.config import DataConfig
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.schedule.config import BatchConfig

if typing.TYPE_CHECKING:
    from fast_llm.engine.distributed.distributed import Distributed


class Data(abc.ABC):
    _distributed: "Distributed"
    _samples_per_phase: dict[PhaseType, int]
    _cache_directory: pathlib.Path | None

    def __init__(self, config: DataConfig, distributed_config: DistributedConfig) -> None:
        self._config = config
        self._distributed_config = distributed_config

    # TODO: Improve interface
    def setup(
        self,
        distributed: "Distributed",
        samples_per_phase: dict[PhaseType, int],
        cache_directory: pathlib.Path,
    ):
        self._distributed = distributed
        self._samples_per_phase = samples_per_phase
        self._cache_directory = cache_directory

    @property
    def config(self):
        return self._config

    @property
    def distributed(self):
        return self._distributed

    @abc.abstractmethod
    def get_iterator(
        self,
        batch_config: BatchConfig,
        phase: PhaseType,
        *,
        consumed_samples: int,
        num_workers: int,
        prefetch_factor: int | None = None,
    ):
        pass
