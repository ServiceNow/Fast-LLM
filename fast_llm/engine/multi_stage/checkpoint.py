import json
import logging

import safetensors.torch
import torch
import yaml

from fast_llm.core.distributed import safe_barrier
from fast_llm.engine.config_utils.checkpoint import CheckpointSaveConfig
from fast_llm.engine.distributed.distributed import Distributed

logger = logging.getLogger(__name__)


def _export_safetensors_metadata(metadata):
    """
    Safetensor only accepts string entries, so we convert to string explicitly.
    We use yaml rather than json because json requires explicit quotation marks on strings, which breaks things.
    (ex. "format": "pt" becomes '"pt"' which breaks huggingface models.)
    We avoid using safe_dump for scalars because it adds junk ("\n...\n") at the end of the string
    (decoding is unaffected.)
    """
    return {
        key: str(value) if isinstance(value, (str, int, float, bool)) else yaml.safe_dump(value)
        for key, value in metadata.items()
    }


class StateDictSaver:
    def __init__(
        self,
        config: CheckpointSaveConfig,
        *,
        distributed: Distributed,
        metadata,
        base_file_name: str,
    ):
        self._config = config
        self._metadata = metadata
        self.distributed = distributed
        self._distributed_config = distributed.config
        self.base_file_name = (
            base_file_name
            if self._distributed_config.pipeline_parallel == 1
            else f"{base_file_name}_{self._distributed_config.pipeline_rank}"
        )
        # All ranks reconstruct the pipeline-parallel state (for simplicity), but only one saves it.
        self._do_save = self._distributed_config.data_rank == self._distributed_config.tensor_rank == 0

    def add_tensor(self, name: str, tensor: torch.Tensor):
        assert name not in self.tensors
        self.tensors[name] = tensor
        self.param_count += tensor.numel()
        if self.param_count >= self._config.parameters_per_file:
            self._save_next_file()

    def __enter__(self):
        self.file_count = 0
        self.param_count = 0
        self.tensors = {}
        self.index = {}
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tensors:
            # Save the last file.
            self._save_next_file()

        if self._do_save and self._distributed_config.pipeline_parallel != 1:
            # Combine the indexes from all pipeline ranks.
            logger.info(f"Merging pipeline-parallel indexes.")
            json.dump(
                self.index, (self._config.path / f"index_{self._distributed_config.pipeline_rank}.json").open("w")
            )
            safe_barrier(self.distributed.pipeline_group, "save state dict")
            if self._distributed_config.pipeline_rank == 0:
                self.index = {}
                for rank in range(self._distributed_config.pipeline_parallel):
                    file_name = self._config.path / f"index_{rank}.json"
                    local_index = json.load(file_name.open("r"))
                    for key, value in local_index.items():
                        assert key not in self.index, key
                        self.index[key] = value
                    file_name.unlink()

        if self._distributed_config.rank == 0:
            path = self._config.path / f"{self.base_file_name}.safetensors.index.json"
            logger.info(f"Saving index to {path}")
            # Save the index.
            json.dump(
                {"metadata": self._metadata, "weight_map": self.index},
                path.open("w"),
                indent=4,
            )

    def _save_next_file(self):
        file_name = f"{self.base_file_name}_{self.file_count}.safetensors"
        if self._do_save:
            logger.info(f"Saving tensors to {self._config.path / file_name}")
            safetensors.torch.save_file(
                tensors=self.tensors,
                filename=self._config.path / file_name,
                metadata=_export_safetensors_metadata(self._metadata),
            )
        for name_ in self.tensors:
            assert name_ not in self.index
            self.index[name_] = file_name
        self.file_count += 1
        self.param_count = 0
        self.tensors = {}
