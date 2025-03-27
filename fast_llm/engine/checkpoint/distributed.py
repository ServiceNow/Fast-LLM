import logging
import typing

import safetensors.torch
import torch
import yaml

from fast_llm.core.distributed import broadcast_scalar
from fast_llm.engine.checkpoint.config import (
    CheckpointFormat,
    CheckpointHandler,
    CheckpointLoadConfig,
    CheckpointLoadMetadataConfig,
    CheckpointSaveConfig,
    DistributedCheckpointFormat,
    ModelConfigType,
    export_safetensors_metadata,
)
from fast_llm.engine.checkpoint.safe_load import SafeLoad
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.engine.multi_stage.config import CheckpointMetadata
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class DistributedCheckpointHandler(CheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = DistributedCheckpointFormat

    @classmethod
    def load_metadata(cls, config: CheckpointLoadMetadataConfig) -> CheckpointMetadata:
        return CheckpointMetadata.from_dict(yaml.safe_load((config.path / "metadata.yaml").open("r")))

    def save(self, config: CheckpointSaveConfig, metadata: CheckpointMetadata) -> None:
        serialized_metadata = metadata.to_serialized()
        if self._model.config.distributed.rank == 0:
            yaml.safe_dump(serialized_metadata, (config.path / "metadata.yaml").open("w"))
        safetensors.torch.save_file(
            tensors={f"{shard_name}_shard": self._model.get_shard(shard_name) for shard_name in metadata.shards},
            filename=config.path / f"rank_{self._model.config.distributed.rank}.safetensors",
            metadata=export_safetensors_metadata(serialized_metadata),
        )

    def load(self, config: CheckpointLoadConfig, metadata: CheckpointMetadata) -> None:
        # TODO: More safety checks
        loaded_config_dict = config.to_copy({"load_config": ModelConfigType.fast_llm})
        loaded_config = self._model.config_class.from_metadata(loaded_config_dict, metadata)
        shard_names = self.get_shard_names(config)
        # Make sure all shards to load are in the checkpoint.
        Assert.leq(set(self.get_shard_names(config)), set(metadata.shards))
        Assert.eq(metadata.shards[: len(shard_names)], list(shard_names))

        same_format = (
            loaded_config.to_serialized(verbose=None) == self._model.config.to_serialized(verbose=None)
            and config.optimizer_state
        )
        # Make sure all nodes agree on which loading scheme to use.
        # Note: they may not agree before the broadcast because of the rank comparison, but that's ok.
        same_format = broadcast_scalar(same_format, torch.uint8, self._model.distributed.world_group)

        if same_format:
            log_main_rank("Checkpoint format matches, using fast load", log_fn=logger.info)
            # TODO: Add version without optimizer state?
            with safetensors.safe_open(
                config.path / f"rank_{self._model.config.distributed.rank}.safetensors",
                framework="pt",
                device=str(self._model.distributed.device),
            ) as f:
                if "state_shard" in f.keys():
                    # Old format `state_shard` with shape `(num_shards, shard_size)
                    # TODO v0.3: Use checkpoint version? Drop support?
                    log_main_rank("Using legacy distributed checkpoint loader.", log_fn=logger.warning)
                    for shard_name in shard_names:
                        self._model.get_shard(shard_name).copy_(
                            f.get_slice("state_shard")[metadata.shards.index(shard_name)]
                        )
                else:
                    # TODO: Does this copy twice?
                    for shard_name in shard_names:
                        self._model.get_shard(shard_name).copy_(f.get_tensor(f"{shard_name}_shard"))

        else:
            log_main_rank("Checkpoint format doesn't match, using safe load", log_fn=logger.info)
            self._model.config.base_model.compare_architecture(loaded_config.base_model, config.compare_log_fn)
            with SafeLoad(self._model, shard_names=shard_names, timeout=config.timeout) as context:
                for rank in range(loaded_config.distributed.world_size):
                    loaded_model = self._model.__class__(
                        loaded_config.to_copy({("distributed", "rank"): rank}),
                        optimizer_state_names=shard_names[1:],
                        verbose=False,
                    )
                    path = config.path / f"rank_{rank}.safetensors"
                    log_main_rank(f"Loading from {path}", log_fn=logger.info)
                    # TODO: skip shards without overlap.
                    with safetensors.safe_open(path, framework="pt", device=str(self._model.distributed.device)) as f:
                        # TODO: Use self_shard
                        if "state_shard" in f.keys():
                            # Old format `state_shard` with shape `(num_shards, shard_size)
                            # TODO v0.3: Use checkpoint version? Drop support?
                            log_main_rank("Using legacy distributed checkpoint loader.", log_fn=logger.warning)
                            loaded_shards = {
                                shard_name: f.get_slice("state_shard")[metadata.shards.index(shard_name)]
                                for shard_name in shard_names
                            }
                        else:
                            loaded_shards = {
                                shard_name: f.get_tensor(f"{shard_name}_shard") for shard_name in shard_names
                            }

                        for shard_name, loaded_shard in loaded_shards.items():
                            loaded_model.get_shard_meta(shard_name).validate(loaded_shard)

                        self_shards = {shard_name: self._model.get_shard(shard_name) for shard_name in shard_names}

                        counter = torch.zeros(1, dtype=torch.int64, device=self._model.distributed.device)
                        for _, loaded_fsdp, loaded_fsdp_shards in loaded_model.split_shards_by_fsdp(loaded_shards):
                            for _, self_fsdp, self_fsdp_shards in self._model.split_shards_by_fsdp(self_shards):
                                self_fsdp.copy_shard_overlaps(
                                    loaded_fsdp,
                                    self_fsdp_shards,
                                    loaded_fsdp_shards,
                                    counter,
                                    self._model.distributed.device,
                                )

                        context.mark_as_loaded(counter.item())
