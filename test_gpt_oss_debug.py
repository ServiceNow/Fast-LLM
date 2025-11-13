#!/usr/bin/env python3
"""
Debug GPT-OSS forward pass differences.
Compare a single token through both models to identify divergence point.
"""

import torch
import transformers

from fast_llm.models.gpt.huggingface import GPTHuggingfaceModel

# Set seed for reproducibility
torch.manual_seed(42)

print("Loading HF model...")
hf_model = (
    transformers.AutoModelForCausalLM.from_pretrained(
        "/home/ubuntu/Fast-LLM/test_gpt_oss_checkpoints_tywyhgh1/dequantized_hf",
        torch_dtype=torch.bfloat16,
    )
    .cuda()
    .eval()
)

print("Loading Fast-LLM model...")
fast_llm_model = (
    GPTHuggingfaceModel.from_pretrained(
        "/home/ubuntu/Fast-LLM/test_gpt_oss_checkpoints_tywyhgh1/fast_llm",
        torch_dtype=torch.bfloat16,
    )
    .cuda()
    .eval()
)

# Create a single token input
test_input = torch.tensor([[199635]], device="cuda")
print(f"\nTest input: {test_input}")

# Run HF model with hooks to capture intermediate values
hf_intermediates = {}


def make_hf_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output
        hf_intermediates[name] = output_tensor.detach().float()

    return hook


# Register hooks on first layer components
hf_model.model.embed_tokens.register_forward_hook(make_hf_hook("embeddings"))
hf_model.model.layers[0].input_layernorm.register_forward_hook(make_hf_hook("layer0_norm1"))
hf_model.model.layers[0].self_attn.register_forward_hook(make_hf_hook("layer0_attn"))
hf_model.model.layers[0].post_attention_layernorm.register_forward_hook(make_hf_hook("layer0_norm2"))
hf_model.model.layers[0].mlp.router.register_forward_hook(make_hf_hook("layer0_router"))
hf_model.model.layers[0].mlp.register_forward_hook(make_hf_hook("layer0_mlp"))

print("\nRunning HF model...")
with torch.no_grad():
    hf_output = hf_model(test_input)
hf_logits = hf_output.logits.float()

print("\n=== HF Intermediate Values ===")
for name, tensor in hf_intermediates.items():
    print(f"{name}: shape={tensor.shape}, mean={tensor.mean():.6f}, std={tensor.std():.6f}")
    if tensor.numel() <= 20:
        print(f"  values={tensor.flatten()[:20]}")

# Now check Fast-LLM embeddings manually
print("\n=== Fast-LLM Manual Check ===")
# Get embedding weight from Fast-LLM
fl_embed_weight = fast_llm_model._model._embedding.embedding.weight.data
print(f"Fast-LLM embedding weight shape: {fl_embed_weight.shape}")
print(f"Fast-LLM embedding for token {test_input[0,0]}: {fl_embed_weight[test_input[0,0], :10]}")

# Get HF embedding weight
hf_embed_weight = hf_model.model.embed_tokens.weight.data
print(f"HF embedding weight shape: {hf_embed_weight.shape}")
print(f"HF embedding for token {test_input[0,0]}: {hf_embed_weight[test_input[0,0], :10]}")

print(f"\nEmbedding weights match: {torch.allclose(fl_embed_weight.float(), hf_embed_weight.float(), atol=1e-3)}")

# Run Fast-LLM model
print("\nRunning Fast-LLM model...")
with torch.no_grad():
    fl_output = fast_llm_model(test_input)
fl_logits = fl_output.logits.float()

print(f"\n=== Output Comparison ===")
print(f"HF logits: shape={hf_logits.shape}, mean={hf_logits.mean():.6f}, std={hf_logits.std():.6f}")
print(f"FL logits: shape={fl_logits.shape}, mean={fl_logits.mean():.6f}, std={fl_logits.std():.6f}")
print(f"Logits match: {torch.allclose(hf_logits, fl_logits, atol=0.01)}")
print(f"Max diff: {(hf_logits - fl_logits).abs().max():.6f}")
print(f"RMS diff: {((hf_logits - fl_logits) ** 2).mean().sqrt():.6f}")
