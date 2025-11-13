#!/usr/bin/env python3
"""Test that the gpt_oss_glu activation matches HF implementation."""

import torch


# HF implementation (from the experts forward code)
def hf_activation(gate_up):
    """
    HF GPT-OSS activation.
    gate_up is interleaved [g0, u0, g1, u1, ...]
    """
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    alpha = 1.702
    limit = 7.0
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu


# Fast-LLM implementation (from config.py)
def fast_llm_activation(x):
    """
    Fast-LLM GPT-OSS activation.
    x is concatenated [gate..., up...]
    """
    gate, up = x.chunk(2, dim=-1)
    alpha = 1.702
    limit = 7.0
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1.0) * glu


# Test
torch.manual_seed(42)
batch, seq, dim = 2, 4, 8

# Create random gate and up
gate = torch.randn(batch, seq, dim)
up = torch.randn(batch, seq, dim)

# HF format: interleaved
hf_input = torch.stack([gate, up], dim=-1).reshape(batch, seq, 2 * dim)
print("HF input shape:", hf_input.shape)
print("HF input [0,0,:10]:", hf_input[0, 0, :10])

# Fast-LLM format: concatenated
fl_input = torch.cat([gate, up], dim=-1)
print("\nFL input shape:", fl_input.shape)
print("FL input [0,0,:10]:", fl_input[0, 0, :10])

# Run both activations
hf_output = hf_activation(hf_input)
fl_output = fast_llm_activation(fl_input)

print("\nHF output shape:", hf_output.shape)
print("HF output [0,0,:5]:", hf_output[0, 0, :5])

print("\nFL output shape:", fl_output.shape)
print("FL output [0,0,:5]:", fl_output[0, 0, :5])

# Compare
print("\nOutputs match:", torch.allclose(hf_output, fl_output, atol=1e-6))
print("Max diff:", (hf_output - fl_output).abs().max().item())
