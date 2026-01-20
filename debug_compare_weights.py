#!/usr/bin/env python3
"""
Debug script to compare actual weight values between distributed checkpoint and HF export.
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
HF_PATH = Path(
    "/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_sndist_mix1_1_lr3e-05_lfmse0.5_rkl0.5_fkl0.5_sl16384_iters500000/export/apriel2/22000"
)


def main():
    from fast_llm.engine.multi_stage.config import CheckpointMetadata

    # Load checkpoint metadata
    with open(CHECKPOINT_PATH / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)

    checkpoint_metadata = CheckpointMetadata.from_dict(metadata)

    # Find the HF model files
    hf_files = list(HF_PATH.glob("*.safetensors"))
    print(f"HF checkpoint files: {[f.name for f in hf_files]}")

    # Load HF weights
    hf_weights = {}
    for hf_file in hf_files:
        with safetensors.safe_open(hf_file, framework="pt") as f:
            for key in f.keys():
                hf_weights[key] = f.get_tensor(key)

    print(f"\nHF weights: {len(hf_weights)} tensors")
    print("Sample HF keys:")
    for key in list(hf_weights.keys())[:10]:
        print(f"  {key}: {hf_weights[key].shape}")

    # Load distributed checkpoint for rank 0
    print("\n" + "=" * 80)
    print("Loading distributed checkpoint rank 0")
    print("=" * 80)

    from fast_llm.models.multimodal.model import MultiModalModel

    loaded_config = checkpoint_metadata.config.to_copy({("distributed", "rank"): 0})
    loaded_model = MultiModalModel(
        loaded_config,
        optimizer_state_names=[],
        verbose=False,
    )

    # Load the shard
    rank_file = CHECKPOINT_PATH / "rank_0.safetensors"
    with safetensors.safe_open(rank_file, framework="pt") as f:
        weights_shard = f.get_tensor("weights_shard")

    print(f"Weights shard shape: {weights_shard.shape}")

    # Build a mapping from parameter name to global shard offset
    print("\n" + "=" * 80)
    print("Building parameter offset mapping")
    print("=" * 80)

    global_offset = 0
    param_global_offsets = {}  # param_name -> (global_start, global_end, stage_idx, fsdp_idx)

    for stage_idx, stage in enumerate(loaded_model._stages):
        for fsdp_idx, fsdp in enumerate(stage._fsdps):
            for param_name in fsdp._parameter_metas:
                begin = fsdp._parameter_begins_in_buffer[param_name]
                end = fsdp._parameter_ends_in_buffer[param_name]

                # Calculate which part is in this rank's shard
                shard_begin_in_buffer = fsdp._fsdp_dim.rank * fsdp._shard_size
                (fsdp._fsdp_dim.rank + 1) * fsdp._shard_size

                param_begin_in_shard = max(0, begin - shard_begin_in_buffer)
                param_end_in_shard = min(fsdp._shard_size, end - shard_begin_in_buffer)

                if param_begin_in_shard < param_end_in_shard:
                    global_start = global_offset + param_begin_in_shard
                    global_end = global_offset + param_end_in_shard
                    param_global_offsets[param_name] = (global_start, global_end, stage_idx, fsdp_idx)

            global_offset += fsdp._shard_size

    print(f"Total global offset: {global_offset} (should match shard size: {weights_shard.shape[0]})")

    # Now compare specific parameters
    print("\n" + "=" * 80)
    print("Comparing specific parameters")
    print("=" * 80)

    # Map from Fast-LLM param names to HF param names
    # Based on HF keys like: model.decoder.blocks.0.input_layernorm.weight
    comparisons = [
        ("head.final_norm.weight", "model.decoder.final_norm.weight"),
        ("embeddings.word_embeddings_weight", "model.embeddings.word_embeddings.weight"),
        ("decoder.0.norm_1.weight", "model.decoder.blocks.0.input_layernorm.weight"),
        (
            "decoder.0.mixer.mixers.attention.query.weight",
            "model.decoder.blocks.0.mixer.mixers.attention.q_proj.weight",
        ),
        ("decoder.0.mixer.mixers.gdn.norm.weight", "model.decoder.blocks.0.mixer.mixers.gdn.norm.weight"),
    ]

    for fl_name, hf_name in comparisons:
        if fl_name not in param_global_offsets:
            print(f"\n{fl_name}: NOT FOUND in rank 0 shard")
            continue

        global_start, global_end, stage_idx, fsdp_idx = param_global_offsets[fl_name]
        param_slice = weights_shard[global_start:global_end]

        print(f"\n{fl_name}:")
        print(f"  Global offset: [{global_start}, {global_end})")
        print(f"  Slice shape: {param_slice.shape}")
        print(f"  Slice values (first 10): {param_slice[:10].tolist()}")

        if hf_name in hf_weights:
            hf_param = hf_weights[hf_name].flatten()
            print(f"  HF {hf_name}:")
            print(f"    Flat shape: {hf_param.shape}")
            print(f"    Values (first 10): {hf_param[:10].tolist()}")

            # For tensor-parallel params, we only have a slice
            # For non-TP params, we should have the full thing
            fsdp = loaded_model._stages[stage_idx]._fsdps[fsdp_idx]
            meta = fsdp._parameter_metas[fl_name]
            print(f"  Is tensor parallel: {meta.is_tensor_parallel}")

            if param_slice.shape[0] == hf_param.shape[0]:
                diff = (param_slice.float() - hf_param.float()).abs().max().item()
                print(f"  Max diff: {diff}")
                if diff > 1e-3:
                    print("  WARNING: Values don't match!")
                else:
                    print("  Values match!")
        else:
            print(f"  HF {hf_name}: NOT FOUND")


if __name__ == "__main__":
    main()
