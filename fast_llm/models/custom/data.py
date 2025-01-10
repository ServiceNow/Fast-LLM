import pathlib

from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.abstract import PhaseSplits
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.models.custom.config import CustomDataConfig


class CustomData(GPTData):
    # TODO: If needed, inherit from AbstractData instead and re-implement everything.
    def __init__(
        self,
        config: CustomDataConfig,
        distributed_config: DistributedConfig,
        vocab_size: int,
        max_sequence_length: int,
    ):
        # TODO: Adjust or reimplement.
        super().__init__(config, distributed_config, vocab_size, max_sequence_length)

    def setup(
        self,
        distributed: Distributed,
        samples_per_phase: PhaseSplits[int],
        cache_directory: pathlib.Path,
    ):
        # TODO: Adjust or reimplement.
        return super().setup(distributed, samples_per_phase, cache_directory)

    def get_iterator(
        self,
        batch_config,
        phase,
        *,
        consumed_samples,
        num_workers,
        prefetch_factor=None,
    ):
        # TODO: Adjust or reimplement.
        return super().get_iterator(
            batch_config,
            phase,
            consumed_samples=consumed_samples,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
