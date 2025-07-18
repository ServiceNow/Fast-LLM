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
    CheckpointSaveMetadataConfig,
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
    def save_metadata(cls, config: CheckpointSaveMetadataConfig, metadata: CheckpointMetadata):
        serialized_metadata = metadata.to_dict()
        config.path.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(serialized_metadata, (config.path / "metadata.yaml").open("w"))

    @classmethod
    def _load_metadata(cls, config: CheckpointLoadMetadataConfig) -> CheckpointMetadata:
        return CheckpointMetadata.from_dict(yaml.safe_load((config.path / "metadata.yaml").open("r")))

    def save(self, config: CheckpointSaveConfig, metadata: CheckpointMetadata) -> None:
        serialized_metadata = metadata.to_dict()
        config.path.mkdir(parents=True, exist_ok=True)
        if self._model.config.distributed.rank == 0:
            yaml.safe_dump(serialized_metadata, (config.path / "metadata.yaml").open("w"))
        safetensors.torch.save_file(
            tensors={f"{shard_name}_shard": self._model.get_shard(shard_name) for shard_name in metadata.shards},
            filename=config.path / f"rank_{self._model.config.distributed.rank}.safetensors",
            metadata=export_safetensors_metadata(serialized_metadata),
        )

    def load(self, config: CheckpointLoadConfig) -> dict[str, typing.Any] | None:
        # TODO: More safety checks
        loaded_metadata = self._model.config.load_metadata(config.to_copy({"load_config": ModelConfigType.fast_llm}))
        shard_names = self.get_shard_names(config)
        # Make sure all shards to load are in the checkpoint.
        Assert.leq(set(shard_names), set(loaded_metadata.shards))
        Assert.eq(loaded_metadata.shards[: len(shard_names)], list(shard_names))

        # Using `log_fn=bool` sets the output to true if the error list is non-empty.
        same_format = config.optimizer_state and not loaded_metadata.config.compare(self._model.config, log_fn=bool)
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
                            f.get_slice("state_shard")[loaded_metadata.shards.index(shard_name)]
                        )
                else:
                    # TODO: Does this copy twice?
                    for shard_name in shard_names:
                        self._model.get_shard(shard_name).copy_(f.get_tensor(f"{shard_name}_shard"))

        else:
            log_main_rank("Checkpoint format doesn't match, using safe load", log_fn=logger.info)
            self._model.config.base_model.compare_architecture(loaded_metadata.config.base_model, logger.warning)
            with SafeLoad(self._model, shard_names=shard_names, timeout=config.timeout) as context:
                for rank in range(loaded_metadata.config.distributed.world_size):
                    loaded_model = self._model.__class__(
                        loaded_metadata.config.to_copy({("distributed", "rank"): rank}),
                        optimizer_state_names=shard_names[1:],
                        verbose=False,
                    )
                    path = config.path / f"rank_{rank}.safetensors"
                    log_main_rank(f"Loading from {path}", log_fn=logger.info)

                    # First do a dry run to check if there is any overlap.
                    if not self._has_shard_overlaps(loaded_model):
                        # No overlap found, skip this file.
                        continue

                    # TODO: Lazy loading?
                    with safetensors.safe_open(path, framework="pt", device=str(self._model.distributed.device)) as f:
                        # TODO: Use self_shard
                        if "state_shard" in f.keys():
                            # Old format `state_shard` with shape `(num_shards, shard_size)
                            # TODO v0.3: Use checkpoint version? Drop support?
                            log_main_rank("Using legacy distributed checkpoint loader.", log_fn=logger.warning)
                            loaded_shards = {
                                shard_name: f.get_slice("state_shard")[loaded_metadata.shards.index(shard_name)]
                                for shard_name in shard_names
                            }
                        else:
                            loaded_shards = {
                                shard_name: f.get_tensor(f"{shard_name}_shard") for shard_name in shard_names
                            }

                    self._copy_shard_overlaps(loaded_model, loaded_shards, context)

        return loaded_metadata.metadata

    def _has_shard_overlaps(self, loaded_model) -> bool:
        for _, loaded_fsdp, _ in loaded_model.split_shards_by_fsdp({}):
            for _, self_fsdp, _ in self._model.split_shards_by_fsdp({}):
                counter = self_fsdp.copy_shard_overlaps(
                    loaded_fsdp,
                    None,
                    None,
                )
                if counter:
                    return True
        return False

    def _copy_shard_overlaps(self, loaded_model, loaded_shards, context):
        for shard_name, loaded_shard in loaded_shards.items():
            loaded_model.get_shard_meta(shard_name).validate(loaded_shard)

        self_shards = {shard_name: self._model.get_shard(shard_name) for shard_name in loaded_shards}

        for _, loaded_fsdp, loaded_fsdp_shards in loaded_model.split_shards_by_fsdp(loaded_shards):
            for _, self_fsdp, self_fsdp_shards in self._model.split_shards_by_fsdp(self_shards):
                counter = self_fsdp.copy_shard_overlaps(
                    loaded_fsdp,
                    self_fsdp_shards,
                    loaded_fsdp_shards,
                )
                for parameter, count in counter.items():
                    context.mark_as_loaded(count, parameter, True)
