#!/usr/bin/env python3
"""
Debug script to compare parameter names and shapes between
a distributed checkpoint and the current model configuration.
"""
import sys
from pathlib import Path

import safetensors.torch
import yaml

# Add Fast-LLM to path
sys.path.insert(0, "/home/toolkit/dev/Fast-LLM")


CHECKPOINT_PATH = Path(
    "/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_sndist_mix1_1_lr3e-05_lfmse0.5_rkl0.5_fkl0.5_sl16384_iters500000/checkpoint/22000"
)


def load_checkpoint_metadata():
    """Load checkpoint metadata."""
    with open(CHECKPOINT_PATH / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)
    return metadata


def get_checkpoint_param_names(rank: int = 0):
    """Get parameter names from a checkpoint rank file."""
    rank_file = CHECKPOINT_PATH / f"rank_{rank}.safetensors"
    if not rank_file.exists():
        print(f"Rank file {rank_file} not found")
        return None

    with safetensors.safe_open(rank_file, framework="pt") as f:
        keys = list(f.keys())
    return keys


def main():
    print("=" * 80)
    print("Checkpoint Parameter Analysis")
    print("=" * 80)

    # Load metadata
    metadata = load_checkpoint_metadata()
    print("\nCheckpoint metadata config keys:")
    for key in metadata.get("config", {}).keys():
        print(f"  - {key}")

    # Get parameter names from rank 0
    print("\n" + "=" * 80)
    print("Parameters in rank 0 checkpoint:")
    print("=" * 80)

    param_names = get_checkpoint_param_names(0)
    if param_names:
        # Group by prefix
        prefixes = {}
        for name in sorted(param_names):
            prefix = name.split(".")[0]
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(name)

        for prefix, names in sorted(prefixes.items()):
            print(f"\n{prefix} ({len(names)} parameters):")
            for name in names[:10]:  # Show first 10
                print(f"  {name}")
            if len(names) > 10:
                print(f"  ... and {len(names) - 10} more")

    # Check for any parameters with unusual names
    print("\n" + "=" * 80)
    print("Looking for mixer-related parameters:")
    print("=" * 80)
    if param_names:
        mixer_params = [n for n in param_names if "mixer" in n.lower()]
        for name in sorted(mixer_params)[:50]:
            print(f"  {name}")
        if len(mixer_params) > 50:
            print(f"  ... and {len(mixer_params) - 50} more")

    # Check shard naming
    print("\n" + "=" * 80)
    print("Shard naming convention:")
    print("=" * 80)
    if param_names:
        shards = [n for n in param_names if "_shard" in n]
        non_shards = [n for n in param_names if "_shard" not in n]
        print(f"  Parameters with '_shard' suffix: {len(shards)}")
        print(f"  Parameters without '_shard' suffix: {len(non_shards)}")
        if non_shards:
            print("  Non-shard parameters:")
            for n in non_shards[:10]:
                print(f"    {n}")

    # Compare parameter shapes from checkpoint vs expected
    print("\n" + "=" * 80)
    print("Sample parameter shapes from rank 0:")
    print("=" * 80)
    rank_file = CHECKPOINT_PATH / "rank_0.safetensors"
    with safetensors.safe_open(rank_file, framework="pt") as f:
        for key in sorted(f.keys())[:20]:
            tensor = f.get_tensor(key)
            print(f"  {key}: {tensor.shape}")


if __name__ == "__main__":
    main()
