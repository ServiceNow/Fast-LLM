import abc
import logging
import typing

import safetensors
import safetensors.torch
import torch
import yaml

from fast_llm.core.distributed import safe_barrier
from fast_llm.engine.checkpoint.config import (
    CheckpointFormat,
    CheckpointHandler,
    CheckpointLoadConfig,
    CheckpointLoadMetadataConfig,
    CheckpointSaveConfig,
    CheckpointSaveMetadataConfig,
    FastLLMCheckpointFormat,
    export_safetensors_metadata,
)
from fast_llm.engine.checkpoint.safe_load import SafeLoad
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import CheckpointMetadata
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class StateDictCheckpointHandler(CheckpointHandler):
    base_file_name: typing.ClassVar[str] = "model"

    def save(self, config: CheckpointSaveConfig, metadata: CheckpointMetadata):
        serialized_metadata = self._serialize_metadata(config, metadata)
        saver = StateDictSaver(
            config,
            distributed=self._model.distributed,
            serialized_metadata=serialized_metadata,
            base_file_name=self.base_file_name,
        )
        # The tensor mapping may not be one-to-one. `convert_state_dict` pops all tensors from
        #   `state_dict` that are ready for conversion,
        #   and return a dict containing the converted tensors(s).
        #   If converting a tensor requires another one that is not yet available (e.g. for concatenation),
        #   it will remain in `state_dict` until that tensor is available.
        state_dict = {}
        for parameter_name, shard_name, tensor in self._model.get_state_tensor_iterator(
            self.get_shard_names(config), config.data_type
        ):
            if shard_name not in state_dict:
                state_dict[shard_name] = {}
            shard_state_dict = state_dict[shard_name]
            assert parameter_name not in shard_state_dict
            shard_state_dict[parameter_name] = tensor
            for exported_name, exported_tensor in self._convert_state_dict(shard_state_dict, True).items():
                saver.add_tensor(self._get_key(exported_name, shard_name), exported_tensor)

        for shard_name, shard_state_dict in state_dict.items():
            assert not shard_state_dict, (shard_name, list(state_dict))

        index = saver.finalize()
        if self._model.distributed_config.rank == 0:
            self._save_serialized_metadata(config, serialized_metadata, index)

    @abc.abstractmethod
    def _save_serialized_metadata(self, config: CheckpointSaveMetadataConfig, metadata: dict, index: dict):
        pass

    def _serialize_metadata(self, config: CheckpointSaveMetadataConfig, metadata: CheckpointMetadata) -> dict:
        return metadata.to_serialized()

    def load(self, config: CheckpointLoadConfig, metadata: CheckpointMetadata):
        with SafeLoad(self._model, num_shards=self.get_num_shards(config)) as context:
            # The tensor mapping may not be one-to-one. `convert_state_dict` pops all tensors from
            #   `state_dict` that are ready for conversion,
            #   and return a dict containing the converted tensors(s).
            #   If converting a tensor requires another one that is not yet available (e.g. for concatenation),
            #   it will remain in `state_dict` until that tensor is available.
            state_dict = {}
            for parameter_name, shard_name, tensor in self._load_weights(config, self._model.distributed.device):
                if shard_name not in state_dict:
                    state_dict[shard_name] = {}
                shard_state_dict = state_dict[shard_name]
                assert parameter_name not in shard_state_dict
                shard_state_dict[parameter_name] = tensor
                for parameter_name, fast_llm_tensor in self._convert_state_dict(shard_state_dict, False).items():
                    loaded = self._model.import_state_tensor(parameter_name, shard_name, fast_llm_tensor)
                    context.mark_as_loaded(loaded, (parameter_name, shard_name))

            for shard_name, shard_state_dict in state_dict.items():
                assert not shard_state_dict, (shard_name, list(state_dict))

    @classmethod
    @abc.abstractmethod
    def _get_key(cls, parameter_name: str, shard_name: str) -> str:
        pass

    @abc.abstractmethod
    def _convert_state_dict(
        self, state_dict: dict[str, torch.Tensor | SafeTensorSlice], export: bool
    ) -> dict[str, torch.Tensor | SafeTensorSlice]:
        pass

    @abc.abstractmethod
    def _load_weights(
        self, config: CheckpointLoadConfig, device
    ) -> typing.Iterator[tuple[str, str, torch.Tensor | SafeTensorSlice]]:
        pass


class FastLLMCheckpointHandler(StateDictCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = FastLLMCheckpointFormat

    @classmethod
    def load_metadata(cls, config: CheckpointLoadMetadataConfig):
        path = config.path / f"metadata.yaml"
        logger.warning(f"Loading metadata from {path}")
        return CheckpointMetadata.from_dict(yaml.safe_load(path.open("r")))

    def _save_serialized_metadata(self, config: CheckpointSaveMetadataConfig, serialized_metadata: dict, index: dict):
        path = config.path / f"metadata.yaml"
        logger.info(f"Saving metadata to {path}")
        if "metadata" not in serialized_metadata:
            serialized_metadata["metadata"] = {}
        serialized_metadata["metadata"]["state_index"] = index
        yaml.safe_dump(serialized_metadata, path.open("w"))

    @classmethod
    def _get_key(cls, parameter_name: str, shard_name: str) -> str:
        return f"{parameter_name}/{shard_name}"

    def _convert_state_dict(
        self, state_dict: dict[str, torch.Tensor | SafeTensorSlice], export: bool
    ) -> dict[str, torch.Tensor | SafeTensorSlice]:
        out_state_dict = state_dict.copy()
        state_dict.clear()
        return out_state_dict

    def _load_weights(
        self, config: CheckpointLoadConfig, device
    ) -> typing.Iterator[tuple[str, str, torch.Tensor | SafeTensorSlice]]:
        metadata = self.load_metadata(config)
        shard_names = self.get_shard_names(config)
        Assert.eq(metadata.shards[: self.get_num_shards(config)], list(shard_names))
        for file_name in set(metadata.metadata["state_index"].values()):
            logger.info(f"Loading from {config.path / file_name}")
            with safetensors.safe_open(
                config.path / file_name,
                framework="pt",
                device=str(device),
            ) as f:
                for key in f.keys():
                    parameter_name, shard_name = key.split("/", 1)
                    if shard_name in shard_names:
                        yield parameter_name, shard_name, f.get_slice(key)


class StateDictSaver:
    def __init__(
        self,
        config: CheckpointSaveConfig,
        *,
        distributed: Distributed,
        serialized_metadata: dict,
        base_file_name: str,
    ):
        self._config = config
        self._safetensors_metadata = export_safetensors_metadata(serialized_metadata)
        self._distributed = distributed
        self._distributed_config = distributed.config
        self.base_file_name = (
            base_file_name
            if self._distributed_config.pipeline_parallel == 1
            else f"{base_file_name}_{self._distributed_config.pipeline_rank}"
        )
        # All ranks reconstruct the pipeline-parallel state (for simplicity), but only one saves it.
        self._do_save = self._distributed_config.data_rank == self._distributed_config.tensor_rank == 0
        self._file_count = 0
        self._parameter_count = 0
        self._tensors = {}
        self._index = {}

    def add_tensor(self, name: str, tensor: torch.Tensor):
        assert name not in self._tensors
        self._tensors[name] = tensor
        self._parameter_count += tensor.numel()
        if self._parameter_count >= self._config.parameters_per_file:
            self._save_next_file()

    def finalize(self):
        if self._tensors:
            # Save the last file.
            self._save_next_file()
        # Merge pipeline-parallel indexes.
        self._merge_index()
        return self._index

    def _save_next_file(self):
        file_name = f"{self.base_file_name}_{self._file_count}.safetensors"
        if self._do_save:
            logger.info(f"Saving tensors to {self._config.path / file_name}")
            safetensors.torch.save_file(
                tensors=self._tensors,
                filename=self._config.path / file_name,
                metadata=self._safetensors_metadata,
            )
        for name_ in self._tensors:
            assert name_ not in self._index
            self._index[name_] = file_name
        self._file_count += 1
        self._parameter_count = 0
        self._tensors = {}

    def _merge_index(self):
        if self._do_save and self._distributed_config.pipeline_parallel != 1:
            # Combine the indexes from all pipeline ranks.
            logger.info(f"Merging pipeline-parallel indexes.")
            yaml.dump(
                self._index, (self._config.path / f"index_{self._distributed_config.pipeline_rank}.yaml").open("w")
            )
            safe_barrier(self._distributed.pipeline_group, "save state dict")
            self._index = {}
            if self._distributed_config.pipeline_rank == 0:
                for rank in range(self._distributed_config.pipeline_parallel):
                    file_name = self._config.path / f"index_{rank}.yaml"
                    local_index = yaml.safe_load(file_name.open("r"))
                    for key, value in local_index.items():
                        assert key not in self._index, key
                        self._index[key] = value
                    file_name.unlink()
