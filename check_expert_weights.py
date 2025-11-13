#!/usr/bin/env python3
"""
Check if expert weights are in the correct order after conversion.
"""

import pathlib

import torch
import transformers

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, FastLLMCheckpointFormat, ModelConfigType
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
from fast_llm.models.gpt.model import GPTModel

CHECKPOINT_DIR = pathlib.Path("/home/ubuntu/Fast-LLM/test_gpt_oss_checkpoint")
DEQUANTIZED_HF_PATH = CHECKPOINT_DIR / "dequantized_hf"
FAST_LLM_PATH = CHECKPOINT_DIR / "fast_llm"

print("=" * 80)
print("Checking Expert Weight Order")
print("=" * 80)

# Load HF model
print("\n1. Loading HF model...")
hf_model = transformers.AutoModelForCausalLM.from_pretrained(
    DEQUANTIZED_HF_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).cuda()

hf_experts = hf_model.model.layers[0].mlp.experts

# Load Fast-LLM model
print("2. Loading Fast-LLM model...")

gpt_model = GPTModel.from_pretrained(
    CheckpointLoadConfig(
        path=FAST_LLM_PATH,
        format=FastLLMCheckpointFormat,
        load_config=ModelConfigType.model,
    )
)

# Wrap with HuggingFace interface
fast_llm_model = HuggingfaceGPTModelForCausalLM(gpt_model)

# Get Fast-LLM MoE weights
# Access the GPT model's decoder layers
fast_llm_layer0_mlp = fast_llm_model.fast_llm_base_model.decoder[0].mlp

# Get layer_1 (gate_up_proj) weight
# HF format: (num_experts, in_features, 2 * out_features) = (32, 2880, 5760)
# Fast-LLM format: (num_experts * 2 * out_features, in_features) = (184320, 2880)

# Check expert 9
expert_idx = 9
in_features = 2880
expert_dim = 2880  # out_features for MoE

print(f"\n3. Comparing Expert {expert_idx} gate_up_proj weights...")

# HF expert 9 gate_up weight
hf_gate_up_weight = hf_experts.gate_up_proj[expert_idx]  # (in_features, 2*expert_dim) = (2880, 5760)
hf_gate_up_bias = hf_experts.gate_up_proj_bias[expert_idx]  # (2*expert_dim,) = (5760,)

print(f"HF gate_up_proj[{expert_idx}] shape: {hf_gate_up_weight.shape}")
print(f"HF gate_up_proj_bias[{expert_idx}] shape: {hf_gate_up_bias.shape}")
print(f"HF gate_up_proj[{expert_idx}] first 10 values: {hf_gate_up_weight[0, :10].float()}")
print(f"HF gate_up_proj_bias[{expert_idx}] first 10 values: {hf_gate_up_bias[:10].float()}")

# Fast-LLM expert 9 gate_up weight
# layer_1.weight is (num_experts * 2 * expert_dim, in_features) = (184320, 2880)
# According to the converter at line 186: weight_per_expert = torch.cat([gate_t, up_t], dim=1)
# This concatenates gate and up FOR EACH EXPERT, then reshapes
# So it's: [expert0_gate, expert0_up, expert1_gate, expert1_up, ...]
# This is INTERLEAVED per expert!

fast_llm_layer1_weight = fast_llm_layer0_mlp.layer_1.weight  # (184320, 2880)
fast_llm_layer1_bias = fast_llm_layer0_mlp.layer_1.bias  # (32, 5760) per-expert biases

num_experts = 32

# Extract expert 9's gate and up weights
# Each expert has 2 * expert_dim rows: first expert_dim rows are gate, next expert_dim rows are up
expert_start = expert_idx * 2 * expert_dim
expert_gate_start = expert_start
expert_gate_end = expert_start + expert_dim
expert_up_start = expert_start + expert_dim
expert_up_end = expert_start + 2 * expert_dim

fast_llm_expert9_gate = fast_llm_layer1_weight[expert_gate_start:expert_gate_end, :]  # (expert_dim, in_features)
fast_llm_expert9_up = fast_llm_layer1_weight[expert_up_start:expert_up_end, :]  # (expert_dim, in_features)

# Biases are per-expert: (32, 5760) where 5760 = 2 * expert_dim (gate and up interleaved)
if fast_llm_layer1_bias is not None:
    fast_llm_expert9_bias = fast_llm_layer1_bias[expert_idx, :]  # (5760,)
    # De-interleave
    fast_llm_expert9_gate_bias = fast_llm_expert9_bias[0::2]  # (expert_dim,)
    fast_llm_expert9_up_bias = fast_llm_expert9_bias[1::2]  # (expert_dim,)
else:
    fast_llm_expert9_gate_bias = None
    fast_llm_expert9_up_bias = None

print(f"\nFast-LLM expert {expert_idx} gate weight shape: {fast_llm_expert9_gate.shape}")
print(f"Fast-LLM expert {expert_idx} up weight shape: {fast_llm_expert9_up.shape}")
print(f"Fast-LLM expert {expert_idx} gate weight first 10 values (row 0): {fast_llm_expert9_gate[0, :10].float()}")
if fast_llm_expert9_gate_bias is not None:
    print(f"Fast-LLM expert {expert_idx} gate bias first 10 values: {fast_llm_expert9_gate_bias[:10].float()}")

# Compare
# HF: input @ weight + bias, where weight is (in_features, 2*expert_dim) interleaved
# For comparison, extract HF gate and up separately
hf_gate_weight = hf_gate_up_weight[:, 0::2]  # (in_features, expert_dim)
hf_up_weight = hf_gate_up_weight[:, 1::2]  # (in_features, expert_dim)
hf_gate_bias = hf_gate_up_bias[0::2]  # (expert_dim,)
hf_up_bias = hf_gate_up_bias[1::2]  # (expert_dim,)

print(f"\nHF expert {expert_idx} gate weight (de-interleaved) shape: {hf_gate_weight.shape}")
print(f"HF expert {expert_idx} gate weight first 10 values (row 0): {hf_gate_weight[0, :10].float()}")
print(f"HF expert {expert_idx} gate bias first 10 values: {hf_gate_bias[:10].float()}")

# Fast-LLM weight is transposed compared to HF
# HF: (in_features, expert_dim)
# Fast-LLM: (expert_dim, in_features)
# So we need to transpose Fast-LLM to compare
fast_llm_expert9_gate_transposed = fast_llm_expert9_gate.t()  # (in_features, expert_dim)

print(f"\n4. Comparison:")
print(f"HF gate weight [0, :10]: {hf_gate_weight[0, :10].float()}")
print(f"Fast-LLM gate weight [0, :10] (transposed): {fast_llm_expert9_gate_transposed[0, :10].float()}")

diff = (hf_gate_weight.float() - fast_llm_expert9_gate_transposed.float()).abs()
print(f"Max diff: {diff.max().item():.6f}")
print(f"Mean diff: {diff.mean().item():.6f}")

if diff.max().item() < 1e-5:
    print("\n✅ Expert 9 gate weights match!")
else:
    print(f"\n❌ Expert 9 gate weights DO NOT match! Max diff = {diff.max().item():.6f}")

print("\n" + "=" * 80)
