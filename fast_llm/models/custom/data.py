from fast_llm.data.data import Data
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.models.custom.config import CustomDataConfig


class CustomData(Data):
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

    def setup(self, distributed, samples_per_phase):
        # TODO: Adjust or reimplement.
        return super().setup(distributed, samples_per_phase)

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
