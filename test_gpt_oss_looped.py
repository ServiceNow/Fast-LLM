#!/usr/bin/env python3
"""
Test GPT-OSS forward pass using LOOPED MoE (not dropless) to isolate implementation differences.
"""

import os
import pathlib
import sys

import torch
import transformers

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, FastLLMCheckpointFormat, ModelConfigType
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
from fast_llm.models.gpt.model import GPTModel
from tests.utils.compare_tensor_logs import CompareConfig

# Set PyTorch memory allocator
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


sys.path.insert(0, "/home/ubuntu/Fast-LLM")

# Configuration
CHECKPOINT_DIR = pathlib.Path("/home/ubuntu/Fast-LLM/test_gpt_oss_checkpoint")
DEQUANTIZED_HF_PATH = CHECKPOINT_DIR / "dequantized_hf"
FAST_LLM_PATH = CHECKPOINT_DIR / "fast_llm"

print("=" * 80)
print("Testing GPT-OSS Forward Pass with LOOPED MoE")
print("=" * 80)

# Create test input
torch.manual_seed(42)
test_input = torch.randint(0, 201088, size=(1, 4), dtype=torch.int64, device="cuda")
print(f"\nTest input: {test_input}")

# ==============================================================================
# Part 1: HuggingFace Model
# ==============================================================================
print("\n" + "=" * 80)
print("Part 1: HuggingFace Model")
print("=" * 80)

hf_model = (
    transformers.AutoModelForCausalLM.from_pretrained(
        DEQUANTIZED_HF_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    .cuda()
    .eval()
)

with torch.no_grad():
    hf_output = hf_model(test_input)

hf_logits = hf_output.logits.clone().cpu()
print(f"HF logits shape: {hf_logits.shape}")
print(f"HF logits mean: {hf_logits.float().mean():.6f}, std: {hf_logits.float().std():.6f}")
print(f"HF logits [0, 0, :10]: {hf_logits[0, 0, :10].float()}")

del hf_model
torch.cuda.empty_cache()

# ==============================================================================
# Part 2: Fast-LLM Model with LOOPED MoE
# ==============================================================================
print("\n" + "=" * 80)
print("Part 2: Fast-LLM Model with LOOPED MoE (dropless=False)")
print("=" * 80)

# Load model
gpt_model = GPTModel.from_pretrained(
    CheckpointLoadConfig(
        path=FAST_LLM_PATH,
        format=FastLLMCheckpointFormat,
        load_config=ModelConfigType.model,
    )
)

# Override dropless setting to force looped implementation
decoder_config = gpt_model.config.base_model.decoder
print(f"\nDecoder type: {type(decoder_config).__name__}")
print(f"Original dropless setting (full): {decoder_config.blocks['full'].mlp.dropless}")
print(f"Original dropless setting (sliding): {decoder_config.blocks['sliding'].mlp.dropless}")
decoder_config.blocks["full"].mlp.dropless = False
decoder_config.blocks["sliding"].mlp.dropless = False
print(f"Modified dropless setting: {decoder_config.blocks['full'].mlp.dropless}")

# Re-initialize the MLP layers with the new config
# This is a bit hacky but necessary to apply the config change
for layer_idx, layer in enumerate(gpt_model.base_model.decoder):
    mlp = layer.mlp
    # Re-select the forward function based on updated config
    dropless_moe = mlp._config.dropless
    if dropless_moe and mlp._sequence_parallel:
        import warnings

        warnings.warn(
            "Dropless MoE not supported for sequence-tensor-parallel, falling back to looped implementation."
        )
        dropless_moe = False
    mlp._mlp_forward = mlp._forward_dropless if dropless_moe else mlp._forward_looped
    print(f"Layer {layer_idx}: Using {'dropless' if dropless_moe else 'looped'} MoE")

# Wrap with HuggingFace interface
fast_llm_model = HuggingfaceGPTModelForCausalLM(gpt_model)

with torch.no_grad():
    fast_llm_output = fast_llm_model(test_input)

fast_llm_logits = fast_llm_output.logits.clone()
print(f"\nFast-LLM logits shape: {fast_llm_logits.shape}")
print(f"Fast-LLM logits mean: {fast_llm_logits.float().mean():.6f}, std: {fast_llm_logits.float().std():.6f}")
print(f"Fast-LLM logits [0, 0, :10]: {fast_llm_logits[0, 0, :10].float()}")

# ==============================================================================
# Part 3: Comparison
# ==============================================================================
print("\n" + "=" * 80)
print("Part 3: Comparison")
print("=" * 80)

hf_logits_gpu = hf_logits.cuda()
errors = []
CompareConfig().compare_tensors(
    {"samples": hf_logits_gpu, "shape": hf_logits_gpu.shape, "step": 0},
    {"samples": fast_llm_logits, "shape": fast_llm_logits.shape, "step": 0},
    errors,
    "HuggingFace vs Fast-LLM (looped)",
    "logits",
)

if errors:
    print(f"\n❌ Comparison failed:")
    for error in errors:
        print(f"   {error}")
else:
    print(f"\n✅ Forward pass outputs match!")

print("\n" + "=" * 80)
