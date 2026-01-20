#!/usr/bin/env python3
"""
Debug script to check FSDP organization and verify shard boundaries.
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

    from fast_llm.models.multimodal.model import MultiModalModel

    # Create model with checkpoint config at rank 0
    config = checkpoint_metadata.config.to_copy({("distributed", "rank"): 0})
    model = MultiModalModel(
        config,
        optimizer_state_names=[],
        verbose=False,
    )

    print("=" * 80)
    print("FSDP Organization Summary")
    print("=" * 80)

    total_shard_size = 0
    stage_shard_sizes = []

    for stage_idx, stage in enumerate(model._stages):
        stage_shard_size = sum(fsdp._shard_size for fsdp in stage._fsdps)
        stage_shard_sizes.append(stage_shard_size)
        total_shard_size += stage_shard_size

    print(f"\nTotal stages: {len(model._stages)}")
    print(f"Total shard size: {total_shard_size}")

    # Load actual shard and verify size
    rank_file = CHECKPOINT_PATH / "rank_0.safetensors"
    with safetensors.safe_open(rank_file, framework="pt") as f:
        weights_shard = f.get_tensor("weights_shard")

    print(f"Checkpoint shard size: {weights_shard.shape[0]}")

    if total_shard_size == weights_shard.shape[0]:
        print("✓ Shard sizes match!")
    else:
        print("✗ SHARD SIZE MISMATCH!")
        print(f"  Expected: {total_shard_size}")
        print(f"  Actual: {weights_shard.shape[0]}")

    # Show stage breakdown
    print("\n" + "=" * 80)
    print("Stage breakdown")
    print("=" * 80)

    cumulative = 0
    for stage_idx in range(0, min(10, len(model._stages))):
        stage = model._stages[stage_idx]
        stage_size = stage_shard_sizes[stage_idx]
        print(f"\nStage {stage_idx}: offset={cumulative}, size={stage_size}")
        for fsdp_idx, fsdp in enumerate(stage._fsdps):
            print(f"  FSDP {fsdp_idx}: shard_size={fsdp._shard_size}, param_count={fsdp._parameter_count}")
            # Show first parameter
            if fsdp._parameter_metas:
                first_param = list(fsdp._parameter_metas.keys())[0]
                print(f"    First param: {first_param}")
        cumulative += stage_size

    # Check if the shard organization matches what we expect
    print("\n" + "=" * 80)
    print("Verifying stage boundaries with actual data")
    print("=" * 80)

    # Check decoder.0.norm_1.weight which should be in stage 27 FSDP 1
    target_param = "decoder.0.norm_1.weight"
    cumulative = 0
    for stage_idx, stage in enumerate(model._stages):
        for fsdp_idx, fsdp in enumerate(stage._fsdps):
            if target_param in fsdp._parameter_metas:
                begin = fsdp._parameter_begins_in_buffer[target_param]
                end = fsdp._parameter_ends_in_buffer[target_param]
                shard_begin = fsdp._fsdp_dim.rank * fsdp._shard_size
                (fsdp._fsdp_dim.rank + 1) * fsdp._shard_size

                param_begin_in_shard = max(0, begin - shard_begin)
                param_end_in_shard = min(fsdp._shard_size, end - shard_begin)

                global_offset = cumulative + param_begin_in_shard

                print(f"\n{target_param}:")
                print(f"  Found in Stage {stage_idx}, FSDP {fsdp_idx}")
                print(f"  Buffer range: [{begin}, {end})")
                print(f"  Shard range: [{param_begin_in_shard}, {param_end_in_shard})")
                print(f"  Global offset: {global_offset}")
                print(f"  Cumulative before this stage: {cumulative}")

                # Read the value from the shard
                param_slice = weights_shard[global_offset : global_offset + (end - begin)]
                print(f"  Shard values: {param_slice[:5].tolist()}")
                break
            cumulative += fsdp._shard_size
        else:
            continue
        break


if __name__ == "__main__":
    main()
