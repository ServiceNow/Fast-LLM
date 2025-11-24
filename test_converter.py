#!/usr/bin/env python3
"""Test that the gate_up_proj converter works correctly."""

import torch

# Simulate HF format: (num_experts, in_features, 2*expert_dim) interleaved
num_experts = 2
in_features = 4
expert_dim = 3
hf_gate_up = torch.randn(num_experts, in_features, 2 * expert_dim)

print("HF format shape:", hf_gate_up.shape)
print("HF gate_up[0, 0, :]:", hf_gate_up[0, 0, :])

# HF extraction
hf_gate = hf_gate_up[:, :, 0::2]  # even indices
hf_up = hf_gate_up[:, :, 1::2]  # odd indices

print("\nHF extracts:")
print("  gate[0, 0, :]:", hf_gate[0, 0, :])
print("  up[0, 0, :]:", hf_up[0, 0, :])

# My converter (import)
gate = hf_gate_up[:, :, 0::2]  # (num_experts, in_features, expert_dim) - even columns
up = hf_gate_up[:, :, 1::2]  # (num_experts, in_features, expert_dim) - odd columns

# Transpose each: (num_experts, expert_dim, in_features)
gate_t = gate.transpose(1, 2)
up_t = up.transpose(1, 2)

# For each expert, concatenate gate and up
# Result: (num_experts, 2 * expert_dim, in_features)
weight_per_expert = torch.cat([gate_t, up_t], dim=1)

# Reshape to (num_experts * 2 * expert_dim, in_features)
fast_llm_weight = weight_per_expert.reshape(num_experts * 2 * expert_dim, in_features)

print("\nFast-LLM format shape:", fast_llm_weight.shape)
print("First expert gate (transposed):", fast_llm_weight[:expert_dim, :])
print("First expert up (transposed):", fast_llm_weight[expert_dim : 2 * expert_dim, :])

# Now simulate Fast-LLM forward pass
# Input: (batch, seq, in_features) @ weight -> (batch, seq, expert_dim * 2) [concatenated gate, up]
input_vec = torch.randn(1, 1, in_features)
print("\nInput:", input_vec)

# Fast-LLM: matmul gives [gate, up] concatenated
fast_llm_output = input_vec @ fast_llm_weight[: 2 * expert_dim, :].t()  # First expert only
print("Fast-LLM output shape:", fast_llm_output.shape)
print("Fast-LLM output:", fast_llm_output)

# Split into gate and up
fl_gate, fl_up = fast_llm_output.chunk(2, dim=-1)
print("Fast-LLM gate:", fl_gate)
print("Fast-LLM up:", fl_up)

# HF: matmul gives [g0, u0, g1, u1, ...] interleaved
hf_output = input_vec @ hf_gate_up[0, :, :]  # First expert: (1, 1, in_features) @ (in_features, 2*expert_dim)
print("\nHF output shape:", hf_output.shape)
print("HF output:", hf_output)

# HF extracts
hf_gate_out = hf_output[:, :, 0::2]
hf_up_out = hf_output[:, :, 1::2]
print("HF gate:", hf_gate_out)
print("HF up:", hf_up_out)

# Compare
print("\nGate match:", torch.allclose(fl_gate, hf_gate_out, atol=1e-5))
print("Up match:", torch.allclose(fl_up, hf_up_out, atol=1e-5))
