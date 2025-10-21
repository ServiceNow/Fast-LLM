#!/usr/bin/env python3
"""
Compare router outputs between HF and Fast-LLM to see if routing is consistent.
"""

import pathlib
import torch
import transformers

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, FastLLMCheckpointFormat, ModelConfigType
from fast_llm.models.gpt.model import GPTModel

CHECKPOINT_DIR = pathlib.Path("/home/ubuntu/Fast-LLM/test_gpt_oss_checkpoint")
DEQUANTIZED_HF_PATH = CHECKPOINT_DIR / "dequantized_hf"
FAST_LLM_PATH = CHECKPOINT_DIR / "fast_llm"

# Create test input
torch.manual_seed(42)
test_input_bf16 = torch.rand(1, 2880, device="cuda", dtype=torch.bfloat16)  # Single token for HF
test_input = test_input_bf16.float()  # Float32 for Fast-LLM

print("=" * 80)
print("Testing Router Outputs")
print("=" * 80)

# ================================================================================
# HF Model - Router
# ================================================================================
print("\n1. HuggingFace Model - Router for Layer 0")
hf_model = (
    transformers.AutoModelForCausalLM.from_pretrained(
        DEQUANTIZED_HF_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    .cuda()
    .eval()
)

layer0_mlp = hf_model.model.layers[0].mlp

with torch.no_grad():
    # Get router logits
    router_logits = test_input_bf16 @ layer0_mlp.router.weight.t() + layer0_mlp.router.bias
    print(f"  Router logits shape: {router_logits.shape}")
    print(f"  Router logits [:10]: {router_logits[0, :10].float()}")
    print(f"  Router logits [9]: {router_logits[0, 9].float()}")

    # Get top-k experts (k=4)
    router_probs = torch.nn.functional.softmax(router_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(router_probs, k=4, dim=-1)

    print(f"  Top-4 expert indices: {top_k_indices[0]}")
    print(f"  Top-4 expert probs: {top_k_probs[0].float()}")
    print(f"  Top-4 expert probs (normalized): {(top_k_probs / top_k_probs.sum(dim=-1, keepdim=True))[0].float()}")

del hf_model
torch.cuda.empty_cache()

# ================================================================================
# Fast-LLM Model - Router
# ================================================================================
print("\n2. Fast-LLM Model - Router for Layer 0")

gpt_model = GPTModel.from_pretrained(
    CheckpointLoadConfig(
        path=FAST_LLM_PATH,
        format=FastLLMCheckpointFormat,
        load_config=ModelConfigType.model,
    )
)

# Restore parameters
for stage in gpt_model._stages:
    stage.restore_parameters()

layer0_mlp_fast = gpt_model.base_model.decoder[0].mlp
router_weight = layer0_mlp_fast.router.weight
router_bias = layer0_mlp_fast.router.bias

print(f"  Router weight shape: {router_weight.shape}")
print(f"  Router bias shape: {router_bias.shape}")

with torch.no_grad():
    # Get router logits
    router_logits_fast = torch.nn.functional.linear(test_input, router_weight, router_bias)
    print(f"  Router logits shape: {router_logits_fast.shape}")
    print(f"  Router logits [:10]: {router_logits_fast[0, :10]}")
    print(f"  Router logits [9]: {router_logits_fast[0, 9]}")

    # Get top-k experts (k=4)
    router_probs_fast = torch.nn.functional.softmax(router_logits_fast, dim=-1)
    top_k_probs_fast, top_k_indices_fast = torch.topk(router_probs_fast, k=4, dim=-1)

    print(f"  Top-4 expert indices: {top_k_indices_fast[0]}")
    print(f"  Top-4 expert probs: {top_k_probs_fast[0]}")
    print(f"  Top-4 expert probs (normalized): {(top_k_probs_fast / top_k_probs_fast.sum(dim=-1, keepdim=True))[0]}")

print("\n" + "=" * 80)
print("Comparison:")
print("  Router outputs match!" if torch.allclose(router_logits.float(), router_logits_fast, rtol=1e-3) else "  Router outputs differ!")
print("=" * 80)
