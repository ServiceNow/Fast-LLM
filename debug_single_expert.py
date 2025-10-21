#!/usr/bin/env python3
"""
Debug single expert processing to find where HF and Fast-LLM diverge.
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
print("Testing Single Expert Processing")
print("=" * 80)

# ================================================================================
# HF Model - Expert 9
# ================================================================================
print("\n1. HuggingFace Model - Expert 9")
hf_model = (
    transformers.AutoModelForCausalLM.from_pretrained(
        DEQUANTIZED_HF_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    .cuda()
    .eval()
)

layer0 = hf_model.model.layers[0]
experts = layer0.mlp.experts
expert_idx = 9

with torch.no_grad():
    # gate_up_proj
    gate_up = test_input_bf16 @ experts.gate_up_proj[expert_idx] + experts.gate_up_proj_bias[expert_idx]
    print(f"  gate_up shape: {gate_up.shape}, mean: {gate_up.float().mean():.6f}")
    print(f"  gate_up [:10]: {gate_up[0, :10].float()}")

    # De-interleave
    gate = gate_up[..., 0::2]
    up = gate_up[..., 1::2]
    print(f"  gate [:10]: {gate[0, :10].float()}")
    print(f"  up [:10]: {up[0, :10].float()}")

    # Activation
    alpha = 1.702
    limit = 7.0
    gate_clamped = gate.clamp(max=limit)
    up_clamped = up.clamp(min=-limit, max=limit)
    glu = gate_clamped * torch.sigmoid(gate_clamped * alpha)
    activated = (up_clamped + 1) * glu

    print(f"  activated shape: {activated.shape}, mean: {activated.float().mean():.6f}")
    print(f"  activated [:10]: {activated[0, :10].float()}")

    # down_proj
    down_out = activated @ experts.down_proj[expert_idx] + experts.down_proj_bias[expert_idx]

    print(f"  down_out shape: {down_out.shape}, mean: {down_out.float().mean():.6f}")
    print(f"  down_out [:10]: {down_out[0, :10].float()}")

del hf_model
torch.cuda.empty_cache()

# ================================================================================
# Fast-LLM Model - Expert 9
# ================================================================================
print("\n2. Fast-LLM Model - Expert 9")

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

layer0_mlp = gpt_model.base_model.decoder[0].mlp
weight_1 = layer0_mlp.layer_1.weight
bias_1 = layer0_mlp.layer_1.bias
weight_2 = layer0_mlp.layer_2.weight
bias_2 = layer0_mlp.layer_2.bias

# Chunk to get expert 9
weight_1_chunks = weight_1.chunk(32)
bias_1_chunks = bias_1.chunk(32)
weight_2_chunks = weight_2.chunk(32)
bias_2_chunks = bias_2.chunk(32)

weight_1_expert9 = weight_1_chunks[9]  # (5760, 2880)
bias_1_expert9 = bias_1_chunks[9].squeeze(0)  # (5760,)
weight_2_expert9 = weight_2_chunks[9]  # (2880, 2880) - transposed
bias_2_expert9 = bias_2_chunks[9].squeeze(0)  # (2880,)

print(f"  weight_1_expert9 shape: {weight_1_expert9.shape}")
print(f"  bias_1_expert9 shape: {bias_1_expert9.shape}")
print(f"  weight_2_expert9 shape: {weight_2_expert9.shape}")
print(f"  bias_2_expert9 shape: {bias_2_expert9.shape}")

with torch.no_grad():
    # Layer 1: gate_up projection (weight is already concatenated, not interleaved)
    gate_up = torch.nn.functional.linear(test_input, weight_1_expert9, bias_1_expert9)
    print(f"  gate_up shape: {gate_up.shape}, mean: {gate_up.float().mean():.6f}")
    print(f"  gate_up [:10]: {gate_up[0, :10].float()}")

    # Split into gate and up (already concatenated in Fast-LLM format)
    gate, up = gate_up.chunk(2, dim=-1)
    print(f"  gate [:10]: {gate[0, :10].float()}")
    print(f"  up [:10]: {up[0, :10].float()}")

    # Activation (same as HF)
    alpha = 1.702
    limit = 7.0
    gate_clamped = gate.clamp(max=limit)
    up_clamped = up.clamp(min=-limit, max=limit)
    glu = gate_clamped * torch.sigmoid(gate_clamped * alpha)
    activated = (up_clamped + 1) * glu

    print(f"  activated shape: {activated.shape}, mean: {activated.float().mean():.6f}")
    print(f"  activated [:10]: {activated[0, :10].float()}")

    # Layer 2: down projection
    # Test both with and without transpose
    print(f"\n  Testing weight_2 transpose:")
    print(f"  weight_2_expert9 shape: {weight_2_expert9.shape}")

    # Option 1: With transpose
    down_out_with_t = torch.nn.functional.linear(activated, weight_2_expert9.t(), bias_2_expert9)
    print(f"  WITH transpose: down_out shape: {down_out_with_t.shape}, mean: {down_out_with_t.float().mean():.6f}")
    print(f"  WITH transpose: down_out [:10]: {down_out_with_t[0, :10].float()}")

    # Option 2: Without transpose (treating weight_2 as already transposed)
    down_out_no_t = activated @ weight_2_expert9.t() + bias_2_expert9
    print(f"  Matmul (@): down_out shape: {down_out_no_t.shape}, mean: {down_out_no_t.float().mean():.6f}")
    print(f"  Matmul (@): down_out [:10]: {down_out_no_t[0, :10].float()}")

    # Option 3: Direct use without any transpose
    down_out_direct = activated @ weight_2_expert9 + bias_2_expert9
    print(f"  Direct (no .t()): down_out shape: {down_out_direct.shape}, mean: {down_out_direct.float().mean():.6f}")
    print(f"  Direct (no .t()): down_out [:10]: {down_out_direct[0, :10].float()}")

print("\n" + "=" * 80)
print("Comparison complete!")
print("=" * 80)
