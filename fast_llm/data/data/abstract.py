import abc

from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.schedule.config import BatchConfig


class Data(abc.ABC):
    # TODO: Improve interface
    @abc.abstractmethod
    def setup(self, distributed: Distributed, samples_per_phase: dict[PhaseType, int]):
        pass

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
