#!/usr/bin/env python3
"""Test that Triton and Torch GPT-OSS GLU implementations match."""

import torch

from fast_llm.functional.config import ActivationType
from fast_llm.functional.triton.mlp import torch_mlp_activation, triton_mlp_activation_forward

# Set seed
torch.manual_seed(42)

# Create test input: concatenated [gate, up]
batch, seq, dim = 2, 4, 128
gate = torch.randn(batch, seq, dim, device="cuda")
up = torch.randn(batch, seq, dim, device="cuda")
input_concat = torch.cat([gate, up], dim=-1)  # shape: (batch, seq, 2*dim)

print(f"Input shape: {input_concat.shape}")
print(f"Gate [:5]: {gate[0, 0, :5]}")
print(f"Up [:5]: {up[0, 0, :5]}")

# Run torch implementation
torch_output = torch_mlp_activation(input_concat, gated=True, activation_type=ActivationType.gpt_oss_glu)

print(f"\nTorch output shape: {torch_output.shape}")
print(f"Torch output [0,0,:5]: {torch_output[0, 0, :5]}")

# Run triton implementation
# Make input contiguous for Triton
input_concat_contig = input_concat.contiguous()
triton_output, _ = triton_mlp_activation_forward(
    input_concat_contig, gated=True, activation_type=ActivationType.gpt_oss_glu
)

print(f"\nTriton output shape: {triton_output.shape}")
print(f"Triton output [0,0,:5]: {triton_output[0, 0, :5]}")

# Compare
print(f"\nOutputs match (atol=1e-5): {torch.allclose(torch_output, triton_output, atol=1e-5)}")
print(f"Max diff: {(torch_output - triton_output).abs().max().item()}")
print(f"RMS diff: {((torch_output - triton_output) ** 2).mean().sqrt().item()}")

# Also check individual values
print(f"\nDetailed comparison:")
for i in range(min(5, dim)):
    print(
        f"  dim {i}: torch={torch_output[0,0,i]:.6f}, triton={triton_output[0,0,i]:.6f}, diff={abs(torch_output[0,0,i] - triton_output[0,0,i]):.6e}"
    )
