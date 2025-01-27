import logging
import math

import torch
from torch.distributed import all_reduce

from fast_llm.core.distributed import add_ephemeral_timeout
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.functional.triton.pointwise import triton_fill
from fast_llm.utils import Assert

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

    def __init__(self, model: "FastLLMModel", *, num_shards: int, timeout: float | None = None):
        self._model = model
        self._distributed = self._model.distributed
        self._num_shards = num_shards
        self._self_shard = self._model.state_shard[: self._num_shards]
        self._timeout = timeout

    def __enter__(self) -> "SafeLoad":
        self._loaded = 0
        self._loaded_parameters = {}
        # Track the number of loaded entries.
        # Use nan to mark non-loaded entries.
        triton_fill(self._self_shard, math.nan)
        # Reset and count shard pads
        for shard in self._model.state_shard[: self._num_shards]:
            shard_split = shard.split(self._model.stage_shard_sizes, 0)
            for stage, stage_shard in zip(self._model.stages_on_device.values(), shard_split):
                self._loaded += stage.reset_shard_pad(stage_shard)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            self._validate()

    def mark_as_loaded(self, count: int, parameter: tuple[str, str] | None = None) -> None:
        self._loaded += count
        if parameter is not None:
            parameter_name, shard_name = parameter
            if shard_name not in self._loaded_parameters:
                self._loaded_parameters[shard_name] = {}
            Assert.not_incl(parameter_name, self._loaded_parameters[shard_name])
            self._loaded_parameters[shard_name][parameter_name] = count

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
        to_load = self._self_shard.numel()
        if self._loaded != to_load:
            # Ensure the right amount of weights is loaded.
            errors.append(f"Loaded a total of {self._loaded:,}, state entries, expected {to_load:,}")

    def _check_missing(self, errors: list[str]) -> None:
        # Ensure the loaded weights have a 1-1 mapping by looking for nans.
        missing = self._self_shard.new_zeros([], dtype=torch.int64)
        # Count nans in slices of 100M parameters to limit memory usage.
        # TODO: Find better solution (triton kernel?)
        for shard_slice in self._self_shard.flatten().split(100000000):
            missing += shard_slice.isnan().sum()
        local_missing = missing.item()
        if self._distributed.world_group is not None:
            all_reduce(missing, group=self._distributed.world_group)
        global_missing = missing.item()
        if global_missing:
            errors.append(f"{global_missing:,} state entries failed to load or corrupted (local={local_missing:,}).")
            # Determine where the missing values are coming from.
            global_total, local_total = 0, 0
            for shard_name, shard_ in zip(self._model.state_shard_names[: self._num_shards], self._self_shard):
                shard_split = shard_.split(self._model.stage_shard_sizes, 0)
                for stage, shard in zip(self._model.stages_on_device.values(), shard_split):
                    buffer = stage._reconstruct_from_shard(shard)
                    for i, parameter in enumerate(stage._split_buffer(buffer)):
                        missing_for_param = parameter.isnan().sum().item()
                        if missing_for_param > 0:
                            global_total += missing_for_param
                            local_values = stage._split_shard(shard)[i]
                            local_missing_for_param = local_values.isnan().sum().item()
                            local_total += local_missing_for_param
                            errors.append(
                                f"{missing_for_param:,} values missing out of {parameter.numel():,} for parameter {stage.parameter_names[i]} in stage {stage.index}, shard {shard_name}"
                                f" (locally {local_missing_for_param:,} out of {local_values.numel():,})"
                            )
                    missing_for_pad = buffer[-stage._global_pad :].isnan().sum().item()
                    if missing_for_pad > 0:
                        global_total += missing_for_pad
                        local_missing_for_pad = (
                            shard[-stage._shard_pad :].isnan().sum().item() if stage._shard_pad > 0 else 0
                        )
                        local_total += local_missing_for_pad
                        errors.append(
                            f"{missing_for_pad:,} values missing out of {stage._global_pad:,} for padding in stage {stage.index}, shard {shard_name}"
                            f" (locally {local_missing_for_pad:,} out of {stage._shard_pad:,})"
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
        loaded_shard_names = set(self._loaded_parameters)
        shard_names = set(self._model.state_shard_names[: self._num_shards])
        if loaded_shard_names != shard_names:
            errors.append(f"Incorrect loaded shards: {loaded_shard_names}!={shard_names}")
        for shard_name in shard_names & loaded_shard_names:
            counter_per_parameter = {
                parameter_name: self._loaded_parameters[shard_name].pop(parameter_name, None)
                for parameter_name in self._model.parameter_names
            }
            for parameter_name, count in self._loaded_parameters[shard_name].items():
                errors.append(f'Loaded unknown parameter "{parameter_name}" for shard "{shard_name}" (count={count})')
            for parameter_name, counter in counter_per_parameter.items():
                if self._model.is_parameter_on_device(parameter_name):
                    if counter is None:
                        errors.append(f'Missing parameter "{parameter_name}" for shard "{shard_name}"')
                elif counter is not None and counter > 0:
                    errors.append(f'Loaded off-device parameter : "{parameter_name}" for shard "{shard_name}"')
            if self._distributed.world_group is not None:
                counter_tensor = torch.tensor(
                    [counter or 0 for counter in counter_per_parameter.values()], dtype=torch.int64
                ).to(self._distributed.device)
                add_ephemeral_timeout(self._distributed.world_group, self._timeout)
                all_reduce(counter_tensor, group=self._distributed.world_group)
                counter_per_parameter = {
                    parameter_name: counter
                    for parameter_name, counter in zip(counter_per_parameter, counter_tensor.tolist())
                }
            for parameter_name, counter in counter_per_parameter.items():
                parameter_size = (
                    self._model.get_parameter_stage(parameter_name)
                    .get_parameter_meta(parameter_name)
                    .global_shape.numel()
                )
                if counter != parameter_size:
                    errors.append(
                        f'Global counter mismatch for parameter "{parameter_name}" and shard "{shard_name}": {counter} != {parameter_size}'
                    )
