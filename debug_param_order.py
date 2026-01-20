#!/usr/bin/env python3
"""
Debug script to compare parameter ordering between checkpoint and current model.
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
    # Load checkpoint config
    with open(CHECKPOINT_PATH / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)

    config = metadata["config"]

    # Now let's create a minimal model to see the parameter ordering
    # We need to set up the distributed config first

    # Create a minimal distributed config for single GPU
    distributed_config = {
        "world_size": 1,
        "rank": 0,
        "local_world_size": 1,
        "local_rank": 0,
        "tensor_parallel": 1,
        "pipeline_parallel": 1,
        "data_parallel": 1,
        "sequence_data_parallel": 1,
    }

    # Merge base_model config
    base_model_config = config["base_model"]

    # Print decoder block mixer config
    print("=" * 80)
    print("Checkpoint decoder.block.mixer config:")
    print("=" * 80)
    mixer_cfg = base_model_config["decoder"]["block"]["mixer"]
    print(f"type: {mixer_cfg.get('type')}")
    print(f"mixers keys (in order): {list(mixer_cfg.get('mixers', {}).keys())}")

    # Try to instantiate the config to see if order is preserved
    print()
    print("=" * 80)
    print("Creating model config from checkpoint config...")
    print("=" * 80)

    try:
        from fast_llm.layers.decoder.config import StochasticMixerConfig

        # Load just the mixer config to check ordering
        mixer_dict = mixer_cfg.copy()
        mixer_dict.pop("type", None)

        print(f"Before Config creation - mixers dict keys: {list(mixer_dict.get('mixers', {}).keys())}")

        # Create the config
        mixer_config = StochasticMixerConfig.from_dict(mixer_dict)

        print(f"After Config creation - mixers dict keys: {list(mixer_config.mixers.keys())}")

        # Check if they match
        original_order = list(mixer_cfg.get("mixers", {}).keys())
        config_order = list(mixer_config.mixers.keys())

        print()
        if original_order == config_order:
            print("✓ Mixer ordering is preserved!")
        else:
            print("✗ MIXER ORDERING CHANGED!")
            print(f"  Original: {original_order}")
            print(f"  Config:   {config_order}")

    except Exception as e:
        print(f"Error creating config: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
