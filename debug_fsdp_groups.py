#!/usr/bin/env python3
"""
Debug script to compare FSDP grouping between checkpoint and model.
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

    from fast_llm.models.multimodal.model import MultiModalModel

    # Create loaded model (checkpoint's config)
    loaded_config = checkpoint_metadata.config.to_copy({("distributed", "rank"): 0})
    loaded_model = MultiModalModel(
        loaded_config,
        optimizer_state_names=[],
        verbose=False,
    )

    # Check parameter offsets in FSDP
    print("=" * 80)
    print("Parameter offsets in FSDP (loaded model)")
    print("=" * 80)

    for stage_idx, stage in enumerate(loaded_model._stages):
        print(f"\nStage {stage_idx}:")
        for fsdp_idx, fsdp in enumerate(stage._fsdps):
            print(f"  FSDP {fsdp_idx}:")
            print(f"    Parameter count: {fsdp._parameter_count}")
            print(f"    Shard size: {fsdp._shard_size}")
            print(f"    Global pad: {fsdp._global_pad}")

            # Show first 5 parameters with their offsets
            param_names = list(fsdp._parameter_metas.keys())
            print(f"    First 5 parameters with offsets:")
            for i, name in enumerate(param_names[:5]):
                begin = fsdp._parameter_begins_in_buffer[name]
                end = fsdp._parameter_ends_in_buffer[name]
                print(f"      {name}: [{begin}, {end})")

            # Show last 5 parameters
            if len(param_names) > 5:
                print(f"    Last 5 parameters with offsets:")
                for name in param_names[-5:]:
                    begin = fsdp._parameter_begins_in_buffer[name]
                    end = fsdp._parameter_ends_in_buffer[name]
                    print(f"      {name}: [{begin}, {end})")

    # Check tensor parallel parameters specifically
    print("\n" + "=" * 80)
    print("Tensor parallel parameters")
    print("=" * 80)

    tp_params = []
    for stage in loaded_model._stages:
        for fsdp in stage._fsdps:
            for name, meta in fsdp._parameter_metas.items():
                if meta.is_tensor_parallel:
                    tp_params.append((name, meta.tensor_parallel_dim, meta.tensor_parallel_size))

    print(f"\nTotal tensor parallel params: {len(tp_params)}")
    print("Sample (first 10):")
    for name, dim, size in tp_params[:10]:
        print(f"  {name}: dim={dim}, size={size}")


if __name__ == "__main__":
    main()
