#!/usr/bin/env python3
"""
Step 1: Download, dequantize, and truncate GPT-OSS model.
Saves the prepared checkpoint to a static directory.
"""

import pathlib

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import Mxfp4Config

# Configuration
MODEL_PATH = "openai/gpt-oss-20b"
OUTPUT_DIR = pathlib.Path("/home/ubuntu/Fast-LLM/test_gpt_oss_checkpoint")
NUM_LAYERS_TO_KEEP = 2


def main():
    print("=" * 80)
    print(f"Preparing GPT-OSS {MODEL_PATH} ({NUM_LAYERS_TO_KEEP}-layer variant)")
    print("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dequantized_path = OUTPUT_DIR / "dequantized_hf"

    print(f"\n1. Downloading HuggingFace model files...")
    print(f"   Source: {MODEL_PATH}")

    # Download the model files from HF Hub
    hf_local_path = snapshot_download(repo_id=MODEL_PATH, local_dir_use_symlinks=False)
    hf_local_path = pathlib.Path(hf_local_path)
    print(f"   Downloaded to: {hf_local_path}")

    print(f"\n2. Loading HuggingFace model with dequantization...")
    # Load with dequantization to convert MXFP4 quantized weights to float
    quantization_config = Mxfp4Config(dequantize=True)

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        hf_local_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    ).cuda()

    print(f"\n3. Trimming model to first {NUM_LAYERS_TO_KEEP} layers...")
    # Keep only first N transformer blocks to reduce memory
    original_num_layers = len(hf_model.model.layers)
    print(f"   Original layers: {original_num_layers}, keeping: {NUM_LAYERS_TO_KEEP}")
    hf_model.model.layers = hf_model.model.layers[:NUM_LAYERS_TO_KEEP]
    hf_model.config.num_hidden_layers = NUM_LAYERS_TO_KEEP

    # GPT-OSS has layer_types config that must match num_hidden_layers
    if hasattr(hf_model.config, "layer_types"):
        print(f"   Original layer_types length: {len(hf_model.config.layer_types)}")
        hf_model.config.layer_types = hf_model.config.layer_types[:NUM_LAYERS_TO_KEEP]
        print(f"   Trimmed layer_types: {hf_model.config.layer_types}")

    print(f"\n4. Saving trimmed dequantized model...")
    print(f"   Output: {dequantized_path}")
    hf_model.save_pretrained(dequantized_path)

    print(f"\n✅ Checkpoint prepared successfully!")
    print(f"   Location: {dequantized_path}")
    print(f"   Vocab size: {hf_model.config.vocab_size}")
    print(f"   Hidden size: {hf_model.config.hidden_size}")
    print(f"   Num layers: {hf_model.config.num_hidden_layers}")

    # Free memory
    del hf_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
