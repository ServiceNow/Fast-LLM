#!/usr/bin/env python3
"""
Debug script to compare parameter names and offsets between checkpoint and model.
"""
import sys
from pathlib import Path

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
    print("Creating model with checkpoint config (rank 0)")
    print("=" * 80)

    # Create a model with the checkpoint's config (like SafeLoad does)
    loaded_config = checkpoint_metadata.config.to_copy({("distributed", "rank"): 0})

    # Create the model class
    from fast_llm.models.multimodal.model import MultiModalModel

    loaded_model = MultiModalModel(
        loaded_config,
        optimizer_state_names=[],
        verbose=False,
    )

    # Get parameter names from loaded model
    print("\nLoaded model parameter names (first 50):")
    loaded_param_names = []
    stages = loaded_model._stages.values() if hasattr(loaded_model._stages, "values") else loaded_model._stages
    for stage in stages:
        for fsdp in stage._fsdps:
            for name in fsdp.parameter_names:
                loaded_param_names.append(name)

    for i, name in enumerate(loaded_param_names[:50]):
        print(f"  {i:4d}: {name}")
    print(f"  ... total: {len(loaded_param_names)} parameters")

    # Now create a "new" model with the same config but potentially different instantiation
    print()
    print("=" * 80)
    print("Creating fresh model with same base_model config")
    print("=" * 80)

    # Simulate what happens when loading pretrained
    fresh_config = checkpoint_metadata.config.to_copy({("distributed", "rank"): 0})
    fresh_model = MultiModalModel(
        fresh_config,
        optimizer_state_names=[],
        verbose=False,
    )

    # Get parameter names from fresh model
    print("\nFresh model parameter names (first 50):")
    fresh_param_names = []
    stages = fresh_model._stages.values() if hasattr(fresh_model._stages, "values") else fresh_model._stages
    for stage in stages:
        for fsdp in stage._fsdps:
            for name in fsdp.parameter_names:
                fresh_param_names.append(name)

    for i, name in enumerate(fresh_param_names[:50]):
        print(f"  {i:4d}: {name}")
    print(f"  ... total: {len(fresh_param_names)} parameters")

    # Compare
    print()
    print("=" * 80)
    print("Comparison")
    print("=" * 80)

    if loaded_param_names == fresh_param_names:
        print("✓ Parameter names and order match exactly!")
    else:
        print("✗ PARAMETER NAMES OR ORDER DIFFER!")

        loaded_set = set(loaded_param_names)
        fresh_set = set(fresh_param_names)

        only_in_loaded = loaded_set - fresh_set
        only_in_fresh = fresh_set - loaded_set

        if only_in_loaded:
            print(f"\n  Only in loaded model: {len(only_in_loaded)}")
            for name in sorted(only_in_loaded)[:10]:
                print(f"    {name}")

        if only_in_fresh:
            print(f"\n  Only in fresh model: {len(only_in_fresh)}")
            for name in sorted(only_in_fresh)[:10]:
                print(f"    {name}")

        if not only_in_loaded and not only_in_fresh:
            print("\n  Same parameters but different ORDER!")
            print("  First mismatches:")
            for i, (l, f) in enumerate(zip(loaded_param_names, fresh_param_names)):
                if l != f:
                    print(f"    Position {i}: loaded='{l}' vs fresh='{f}'")
                    if i > 10:
                        print("    ...")
                        break


if __name__ == "__main__":
    main()
