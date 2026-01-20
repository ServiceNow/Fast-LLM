#!/usr/bin/env python3
"""
Load a distributed Fast-LLM checkpoint for inspection.

Usage with 8 GPUs (fast):
    torchrun --nproc_per_node=8 load_distributed_checkpoint.py

Usage with single GPU (slower):
    python load_distributed_checkpoint.py
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Load distributed checkpoint")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_sndist_mix1_1_lr3e-05_lfmse0.5_rkl0.5_fkl0.5_sl16384_iters500000/checkpoint/22000",
    )
    parser.add_argument("--use-cpu", action="store_true", help="Load on CPU instead of GPU")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./distributed_debug_output",
        help="Directory to save weights/statistics outputs (one file per rank).",
    )
    args = parser.parse_args()

    # Import Fast-LLM components
    import fast_llm.models.auto  # noqa: F401 - registers model types
    from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, DistributedCheckpointFormat
    from fast_llm.engine.multi_stage.config import StageMode
    from fast_llm.models.multimodal.model import MultiModalModel

    logger.info("=" * 80)
    logger.info(f"Loading checkpoint: {args.checkpoint_path}")
    logger.info("=" * 80)

    # Configure checkpoint loading
    load_config = CheckpointLoadConfig(
        path=Path(args.checkpoint_path),
        format=DistributedCheckpointFormat,
        model_weights=True,
        optimizer_state=False,
    )
    load_config.setup(MultiModalModel.config_class)

    # Load the model
    logger.info("Loading model...")
    model = MultiModalModel.from_pretrained(
        load_config,
        mode=StageMode.weights,
        use_cpu=args.use_cpu,
    )
    logger.info("Model loaded successfully!")

    # Print model info
    logger.info("")
    logger.info("=" * 80)
    logger.info("Model Information")
    logger.info("=" * 80)
    logger.info(f"Number of stages: {len(model.stages)}")
    logger.info(f"Parameter names count: {len(model.parameter_names)}")

    # Iterate through parameters and print stats
    logger.info("")
    logger.info("Parameters:")
    total_params = 0
    total_sum = 0.0
    stats = {}
    file_index = 0
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    for name, shard_name, tensor in model.get_state_tensor_iterator(model.state_shard_names):
        param_count = tensor.numel()
        param_sum = tensor.sum(dtype=torch.float64).item()
        total_params += param_count
        total_sum += param_sum
        logger.info(f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, sum={param_sum:.4f}")
        file_name = f"weights_rank{rank}_ws{world_size}_{file_index}.safetensors"
        stats[name] = {
            "shard": shard_name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": param_count,
            "sum": param_sum,
            "file": file_name,
        }
        output_dir = Path(args.output_dir)
        # output_dir.mkdir(parents=True, exist_ok=True)
        # safetensors_path = output_dir / file_name
        # safetensors.torch.save_file({name: tensor}, safetensors_path)
        file_index += 1

    logger.info("")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Total parameter sum: {total_sum:.2f}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    stats["_summary"] = {
        "rank": rank,
        "world_size": world_size,
        "total_params": total_params,
        "total_sum": total_sum,
    }
    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stats_path = output_dir / f"weights_rank{rank}_ws{world_size}.json"
        logger.info(f"Saving stats to {stats_path}")
        stats_path.write_text(json.dumps(stats, indent=2))

    # Return model for interactive use
    return model


if __name__ == "__main__":
    model = main()
    print("\nModel loaded into 'model' variable. You can inspect it interactively.")
