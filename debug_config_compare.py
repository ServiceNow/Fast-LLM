#!/usr/bin/env python3
"""
Debug script to compare checkpoint config with model config.
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
    print("Config comparison (like same_format check)")
    print("=" * 80)

    # Create a model config (simulating "new experiment" scenario)
    model_config = checkpoint_metadata.config.to_copy({("distributed", "rank"): 0})

    # Compare configs
    print("\nComparing checkpoint config with itself (should match):")
    errors = []
    checkpoint_metadata.config.compare(model_config, log_fn=lambda msg: errors.append(msg))
    if errors:
        print("  Differences found:")
        for e in errors[:10]:
            print(f"    - {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")
    else:
        print("  No differences found")

    # Check which fields are compared
    print("\n" + "=" * 80)
    print("Checking base_model config architecture comparison")
    print("=" * 80)

    errors = []
    checkpoint_metadata.config.base_model.compare_architecture(
        model_config.base_model, log_fn=lambda msg: errors.append(msg)
    )
    if errors:
        print("  Architecture differences found:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  No architecture differences")

    # Show what the compare method actually compares
    print("\n" + "=" * 80)
    print("Config compare method details")
    print("=" * 80)

    # Let's look at what makes configs "same_format"
    # same_format = config.optimizer_state and not loaded_metadata.config.compare(self._model.config, log_fn=bool)

    print("\nFor same_format=True, we need:")
    print("  1. optimizer_state=True")
    print("  2. config.compare() returns no errors")
    print()
    print("The compare() method checks if configs are identical")


if __name__ == "__main__":
    main()
