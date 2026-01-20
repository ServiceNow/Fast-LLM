#!/usr/bin/env python3
"""
Debug script to simulate checkpoint loading and check byte counts.
"""
import sys
from pathlib import Path

import safetensors
import yaml

# Import to register all types
import fast_llm.models.auto  # noqa

sys.path.insert(0, "/home/toolkit/dev/Fast-LLM")


CHECKPOINT_PATH = Path(
    "/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_sndist_mix1_1_lr3e-05_lfmse0.5_rkl0.5_fkl0.5_sl16384_iters500000/checkpoint/22000"
)


def main():
    from fast_llm.engine.multi_stage.config import CheckpointMetadata

    # Load checkpoint metadata
    with open(CHECKPOINT_PATH / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)

    checkpoint_metadata = CheckpointMetadata.from_dict(metadata)

    print("=" * 80)
    print("Checkpoint loading simulation")
    print("=" * 80)

    print(f"\nCheckpoint world_size: {checkpoint_metadata.config.distributed.world_size}")
    print(f"Checkpoint tensor_parallel: {checkpoint_metadata.config.distributed.tensor_parallel}")
    print(f"Checkpoint data_parallel: {checkpoint_metadata.config.distributed.data_parallel}")

    # Check shard sizes from rank 0
    rank_file = CHECKPOINT_PATH / "rank_0.safetensors"
    with safetensors.safe_open(rank_file, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"  {key}: {tensor.shape}, {tensor.numel()} elements")

    # Now simulate what SafeLoad does
    print("\n" + "=" * 80)
    print("Simulating SafeLoad (loading rank 0 into a model with same config)")
    print("=" * 80)

    from fast_llm.models.multimodal.model import MultiModalModel

    # Create loaded model (checkpoint's config)
    loaded_config = checkpoint_metadata.config.to_copy({("distributed", "rank"): 0})
    loaded_model = MultiModalModel(
        loaded_config,
        optimizer_state_names=[],
        verbose=False,
    )

    # Create current model (same config for now)
    current_config = checkpoint_metadata.config.to_copy({("distributed", "rank"): 0})
    current_model = MultiModalModel(
        current_config,
        optimizer_state_names=[],
        verbose=False,
    )

    # Check overlap
    print("\nChecking shard overlaps:")
    total_overlap = 0
    overlap_details = {}

    for loaded_stage, loaded_fsdp, _ in loaded_model.split_shards_by_fsdp({}):
        for current_stage, current_fsdp, _ in current_model.split_shards_by_fsdp({}):
            counter = current_fsdp.copy_shard_overlaps(
                loaded_fsdp,
                None,
                None,
            )
            for (param_loaded, param_current), count in counter.items():
                total_overlap += count
                key = f"{param_loaded} -> {param_current}"
                if key not in overlap_details:
                    overlap_details[key] = 0
                overlap_details[key] += count

    print(f"\nTotal overlap: {total_overlap} elements")
    print(f"Number of parameters with overlap: {len(overlap_details)}")

    # Compare to expected
    expected_elements = 0
    stages = current_model._stages.values() if hasattr(current_model._stages, "values") else current_model._stages
    for stage in stages:
        for fsdp in stage._fsdps:
            expected_elements += fsdp._shard_size

    print(f"Expected shard size (rank 0): {expected_elements}")

    # Show sample overlaps
    print("\nSample overlaps (first 10):")
    for i, (key, count) in enumerate(sorted(overlap_details.items())[:10]):
        print(f"  {key}: {count}")

    # Check parameter name matching
    print("\n" + "=" * 80)
    print("Checking parameter name matching")
    print("=" * 80)

    loaded_params = set()
    current_params = set()

    for stage in loaded_model._stages:
        for fsdp in stage._fsdps:
            for name in fsdp.parameter_names:
                loaded_params.add(name)

    for stage in current_model._stages:
        for fsdp in stage._fsdps:
            for name in fsdp.parameter_names:
                current_params.add(name)

    only_in_loaded = loaded_params - current_params
    only_in_current = current_params - loaded_params

    print(f"\nLoaded model params: {len(loaded_params)}")
    print(f"Current model params: {len(current_params)}")

    if only_in_loaded:
        print(f"\nParams only in loaded checkpoint ({len(only_in_loaded)}):")
        for name in sorted(only_in_loaded)[:20]:
            print(f"  {name}")
        if len(only_in_loaded) > 20:
            print(f"  ... and {len(only_in_loaded) - 20} more")

    if only_in_current:
        print(f"\nParams only in current model ({len(only_in_current)}):")
        for name in sorted(only_in_current)[:20]:
            print(f"  {name}")
        if len(only_in_current) > 20:
            print(f"  ... and {len(only_in_current) - 20} more")

    if not only_in_loaded and not only_in_current:
        print("\n✓ All parameter names match!")


if __name__ == "__main__":
    main()
