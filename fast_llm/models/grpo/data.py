import json
import torch
from fast_llm.data.data import Data, DatasetSource
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.models.grpo.config import GRPODataConfig
from fast_llm.data.dataset import BlendedDataset, SampledDataset
from fast_llm.utils import Assert


class GRPODataset(SampledDataset):
    """Dataset wrapper that adds GRPO-specific fields (rewards, advantages, etc)"""
    def __init__(self, base_dataset: SampledDataset, data_path: str):
        self.base_dataset = base_dataset
        self.data_path = data_path
        
        # Load the JSONL data
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        data_item = self.data[idx]
        
        # Extract fields from the JSONL data
        batch = {
            "input_ids": item,  # Original input tokens
            "rewards": torch.tensor(data_item["reward"]),
            "old_logprobs": torch.tensor(data_item["logprobs"]),  # These are the logprobs from previous iteration
            "ref_logprobs": torch.tensor(data_item["ref_logprobs"]),
        }
        
        # Compute advantages if not provided in data
        # Here we're using rewards as advantages, but you might want to implement
        # proper advantage estimation
        batch["advantages"] = batch["rewards"].clone()
        
        return batch


class GRPOData(Data):
    def __init__(
        self,
        config: GRPODataConfig,
        distributed_config: DistributedConfig,
        vocab_size: int,
        max_sequence_length: int,
    ):
        super().__init__(config, distributed_config, vocab_size, max_sequence_length)
        
    def setup(self, distributed, samples_per_phase):
        # setup the base data infrastructure
        super().setup(distributed, samples_per_phase)
        
        # wrap each dataset with GRPO-specific functionality
        for phase in self._blended_datasets:
            if isinstance(self._blended_datasets[phase], BlendedDataset):
                # if it's a blended dataset, wrap each underlying dataset
                for i, dataset in enumerate(self._blended_datasets[phase].datasets):
                    dataset = GRPODataset(
                        dataset,
                        data_path=self._dataset_prefixes[f"dataset_{i}"]
                    )
            else:
                # single dataset case
                self._blended_datasets[phase] = GRPODataset(
                    self._blended_datasets[phase],
                    data_path=next(iter(self._dataset_prefixes.values()))
                )

    def get_iterator(
        self,
        batch_config,
        phase,
        *,
        consumed_samples,
        num_workers,
        prefetch_factor=None,
    ):
        return super().get_iterator(
            batch_config,
            phase,
            consumed_samples=consumed_samples,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
