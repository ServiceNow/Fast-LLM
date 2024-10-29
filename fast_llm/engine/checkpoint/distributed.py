import logging
import typing

import safetensors.torch
import torch
import yaml

from fast_llm.engine.checkpoint.config import (
    CheckpointFormat,
    CheckpointHandler,
    CheckpointLoader,
    CheckpointLoadMetadataConfig,
    CheckpointSaver,
    DistributedCheckpointFormat,
    ModelConfigType,
    export_safetensors_metadata,
)
from fast_llm.engine.checkpoint.safe_load import SafeLoad
from fast_llm.engine.multi_stage.config import CheckpointMetadata
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class DistributedCheckpointHandler(CheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = DistributedCheckpointFormat


class DistributedCheckpointSaver(DistributedCheckpointHandler, CheckpointSaver):

    def save(self, metadata: CheckpointMetadata):
        serialized_metadata = metadata.to_serialized()
        if self._model.distributed_config.rank == 0:
            yaml.safe_dump(metadata.to_serialized(), (self._config.path / "metadata.yaml").open("w"))
        safetensors.torch.save_file(
            tensors={"state_shard": self._model.state_shard[: self._num_shards]},
            filename=self._config.path / f"rank_{self._model.distributed_config.rank}.safetensors",
            metadata=export_safetensors_metadata(serialized_metadata),
        )


class DistributedCheckpointLoader(DistributedCheckpointHandler, CheckpointLoader):
    @classmethod
    def load_metadata(cls, config: CheckpointLoadMetadataConfig):
        return CheckpointMetadata.from_dict(yaml.safe_load((config.path / "metadata.yaml").open("r")))

    def load(self, metadata: CheckpointMetadata):
        # TODO: More safety checks
        loaded_config_dict = self._config.to_copy({"load_config": ModelConfigType.fast_llm})
        loaded_config = self._model.config_class.from_metadata(loaded_config_dict, metadata)
        Assert.eq(metadata.shards[: self._num_shards], list(self._shard_names))

        if (
            loaded_config.to_serialized(verbose=None) == self._model.fast_llm_config.to_serialized(verbose=None)
            and self._config.optimizer_state
        ):
            logger.info("Checkpoint format matches, using fast load")
            # TODO: Add version without optimizer state?
            with safetensors.safe_open(
                self._config.path / f"rank_{self._model.distributed_config.rank}.safetensors",
                framework="pt",
                device=str(self._model.distributed.device),
            ) as f:
                # TODO: Does this copy twice?
                self._model.state_shard[: self._num_shards].copy_(f.get_slice("state_shard")[: self._num_shards])
        else:
            logger.info("Checkpoint format doesn't match, using safe load")
            self._model.base_model_config.compare_architecture(loaded_config.base_model, self._config.compare_log_fn)
            with SafeLoad(self._model, num_shards=self._num_shards) as context:
                for rank in range(loaded_config.distributed.world_size):
                    loaded_model = self._model.__class__(
                        loaded_config.to_copy({("distributed", "rank"): rank}),
                        optimizer_state_names=self._shard_names[1:],
                        verbose=False,
                    )
                    path = self._config.path / f"rank_{rank}.safetensors"
                    logger.info(f"Loading from {path}")
                    # TODO: skip shards without overlap.
                    with safetensors.safe_open(path, framework="pt", device=str(self._model.distributed.device)) as f:
                        # TODO: Use self_shard
                        loaded_shard = f.get_slice("state_shard")[: self._num_shards]
                        loaded_model.state_shard_meta.validate(loaded_shard)

                        # TODO: Improve num shard selection.
                        self_shard_split = self._model.state_shard[: loaded_shard.size(0)].split(
                            self._model.stage_shard_sizes, 1
                        )
                        loaded_shard_split = loaded_shard.split(loaded_model.stage_shard_sizes, 1)

                        counter = torch.zeros(1, dtype=torch.int64, device=self._model.distributed.device)
                        for loaded_shard_index, loaded_stage in enumerate(loaded_model.stages_on_device.values()):
                            loaded_shards = (
                                loaded_shard_split[loaded_shard_index].to(self._model.distributed.device).unbind(0)
                            )
                            for self_shard_index, self_stage in enumerate(self._model.stages_on_device.values()):
                                self_stage._copy_shard_overlaps(  # noqa
                                    loaded_stage,
                                    self_shard_split[self_shard_index].unbind(0),
                                    loaded_shards,
                                    counter,
                                )
                        context.mark_as_loaded(counter.item())
