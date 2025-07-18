import logging
import math

import torch
from torch.distributed import all_reduce

from fast_llm.core.distributed import add_ephemeral_timeout
from fast_llm.engine.multi_stage.config import ShardName
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.functional.triton.pointwise import triton_fill

logger = logging.getLogger(__name__)


class SafeLoad:
    """
    A context with multiple safety checks to ensure the model state is loaded correctly:
    * Pre-filling the state with nans and verify that no such value remains at the end.
      This ensures that all values are set at least once.
    * Keep a counter for the number of tensor values set, and validate against the expected number.
      This ensures that all values are set at most once when the nan check succeeds.
    * Optionally keep track of all set parameters and shards, and ensure that each is set exactly once.

    In case of failure, it will attempt to find out as precisely as possible where the problem comes from.
    """

    def __init__(self, model: "FastLLMModel", *, shard_names: tuple[str, ...], timeout: float | None = None):
        self._model = model
        self._distributed = self._model.distributed
        # self._num_shards = num_shards
        self._self_shards = {shard_name: self._model.get_shard(shard_name) for shard_name in shard_names}
        self._timeout = timeout

    def __enter__(self) -> "SafeLoad":
        self._loaded = 0
        self._loaded_parameters = {}
        # Track the number of loaded entries.
        # Use nan to mark non-loaded entries.
        for self_shard in self._self_shards.values():
            triton_fill(self_shard, math.nan)
        # Reset and count shard pads
        for _, fsdp, fsdp_shards in self._model.split_shards_by_fsdp(self._self_shards):
            for shard_name, fsdp_shard in fsdp_shards.items():
                self._loaded += fsdp.reset_shard_pad(fsdp_shard, shard_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            self._validate()

    def mark_as_loaded(self, count: int, parameter: tuple[str, str] | None = None, partial: bool = False) -> None:
        self._loaded += count
        if parameter is not None:
            parameter_name, shard_name = parameter
            if shard_name not in self._loaded_parameters:
                self._loaded_parameters[shard_name] = {}
            if not partial and parameter_name in self._loaded_parameters[shard_name]:
                raise ValueError(f"Duplicate loaded parameter ({parameter_name}, {shard_name})")
            self._loaded_parameters[shard_name][parameter_name] = (
                self._loaded_parameters[shard_name].get(parameter_name, 0) + count
            )

    def _validate(self) -> None:
        errors = []
        self._check_counter(errors)
        self._check_missing(errors)
        if self._loaded_parameters:
            self._check_parameters(errors)
        if errors:
            for error in errors:
                logger.error(error)
            raise RuntimeError("Model loading validation failed. See logs for details.")
        logger.info(f"{self._loaded:,} state entries loaded successfully")

    def _check_counter(self, errors: list[str]) -> None:
        to_load = sum(self_shard.numel() for self_shard in self._self_shards.values())
        if self._loaded != to_load:
            # Ensure the right amount of weights is loaded.
            errors.append(f"Loaded a total of {self._loaded:,}, state entries, expected {to_load:,}")

    def _check_missing(self, errors: list[str]) -> None:
        # Ensure the loaded weights have a 1-1 mapping by looking for nans.
        missing = torch.zeros([], dtype=torch.int64, device=self._distributed.device)
        # Count nans in slices of 100M parameters to limit memory usage.
        # TODO: Find better solution (triton kernel?)
        for shard in self._self_shards.values():
            for shard_slice in shard.flatten().split(100000000):
                missing += shard_slice.isnan().sum()
        local_missing = missing.item()
        if self._distributed.world_group is not None:
            all_reduce(missing, group=self._distributed.world_group)
        global_missing = missing.item()
        if global_missing:
            errors.append(f"{global_missing:,} state entries failed to load or corrupted (local={local_missing:,}).")
            # Determine where the missing values are coming from.
            global_total, local_total = 0, 0
            for stage, fsdp, fsdp_shards in self._model.split_shards_by_fsdp(self._self_shards):
                for shard_name, fsdp_shard in fsdp_shards.items():
                    buffer = fsdp.reconstruct_from_shard(fsdp_shard)
                    for parameter_name, parameter in fsdp.split_buffer(buffer).items():
                        missing_for_param = parameter.isnan().sum().item()
                        if missing_for_param > 0:
                            global_total += missing_for_param
                            local_values = fsdp.split_shard(fsdp_shard)[parameter_name]
                            local_missing_for_param = local_values.isnan().sum().item()
                            local_total += local_missing_for_param
                            errors.append(
                                f"{missing_for_param:,} values missing out of {parameter.numel():,} for parameter {parameter_name} in stage {stage.index}, shard {shard_name}"
                                f" (locally {local_missing_for_param:,} out of {local_values.numel():,})"
                            )
                    missing_for_pad = buffer[-fsdp._global_pad :].isnan().sum().item() if fsdp._global_pad > 0 else 0
                    if missing_for_pad > 0:
                        global_total += missing_for_pad
                        local_missing_for_pad = (
                            fsdp_shard[-fsdp._shard_pad :].isnan().sum().item() if fsdp._shard_pad > 0 else 0
                        )
                        local_total += local_missing_for_pad
                        errors.append(
                            f"{missing_for_pad:,} values missing out of {fsdp._global_pad:,} for padding in stage {stage.index}, shard {shard_name}"
                            f" (locally {local_missing_for_pad:,} out of {fsdp._shard_pad:,})"
                        )

            if global_total != global_missing:
                errors.append(
                    f"Incorrect global breakdown of missing state entries (expected {global_missing:,}, got {global_total:,})"
                )
            if local_total != local_missing:
                errors.append(
                    f"Incorrect local breakdown of missing state entries (expected {local_missing:,}, got {local_total:,})"
                )

    def _check_parameters(self, errors: list[str]) -> None:
        if set(self._loaded_parameters) != set(self._self_shards):
            errors.append(f"Incorrect loaded shards: {tuple(self._loaded_parameters)}!={tuple(self._self_shards)}")

        counters = []
        # Compare local counts against expected values.
        for stage, fsdp, parameter_name, parameter_meta in self._model.stages_fsdp_parameters:
            for shard_name in self._self_shards if fsdp.requires_grad else [ShardName.weights]:
                counter = self._loaded_parameters[shard_name].pop(parameter_meta.tensor_name, 0)
                local_size = (
                    fsdp.get_parameter_size_in_shard(parameter_name, shard_name)
                    if self._model.is_parameter_on_device(parameter_name)
                    else 0
                )
                if counter != local_size:
                    errors.append(
                        f'Local counter mismatch for parameter "{parameter_name}"'
                        f' and shard "{shard_name}": loaded {counter}, expected {local_size}'
                    )

                counter_ = counter
                # Accumulate in a list for global counter check.
                if (
                    not parameter_meta.is_tensor_parallel and self._distributed.config.tensor_rank != 0
                ) or stage.is_tied_weight_copy:
                    # Ignore the counter from duplicate tensors.
                    counter = 0
                if parameter_name == "layers.1.norm_1.weight":
                    logger.info(
                        f"Parameter {parameter_name} local {counter_} keep {counter} (size {parameter_meta.numel()} / {parameter_meta.global_shape.numel()})"
                    )
                counters.append(counter)

        # Check for unexpected parameters.
        for shard_name, loaded in self._loaded_parameters.items():
            for parameter_name, count in loaded.items():
                errors.append(f'Loaded unknown parameter "{parameter_name}" for shard "{shard_name}" (count={count})')

        # All-reduce to get global counts.
        if self._distributed.world_group is not None:
            counter_tensor = torch.tensor(counters, dtype=torch.int64).to(self._distributed.device)
            # This may be the first distributed barrier after loading, so we need to wait for everyone to finish.
            add_ephemeral_timeout(self._distributed.world_group, self._timeout)
            all_reduce(counter_tensor, group=self._distributed.world_group)
            counters = counter_tensor.tolist()

        # Compare global counts against expected values.
        for stage, fsdp, parameter_name, parameter_meta in self._model.stages_fsdp_parameters:
            for shard_name in self._self_shards if fsdp.requires_grad else [ShardName.weights]:
                counter = counters.pop(0)
                if parameter_name == "layers.1.norm_1.weight":
                    logger.info(
                        f"Parameter {parameter_name} global {counter} (size {parameter_meta.numel()} / {parameter_meta.global_shape.numel()})"
                    )
                parameter_size = parameter_meta.global_shape.numel()
                if counter != parameter_size:
                    errors.append(
                        f'Global counter mismatch for parameter "{parameter_name}"'
                        f' and shard "{shard_name}": loaded {counter}, expected {parameter_size}'
                    )
        assert not counters
