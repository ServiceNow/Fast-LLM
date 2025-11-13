#!/usr/bin/env python3
"""
Debug MoE routing to understand expert selection and scoring differences.
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
print("Debug MoE Routing")
print("=" * 80)

# Create test input
torch.manual_seed(42)
test_input = torch.randint(0, 201088, size=(1, 4), dtype=torch.int64, device="cuda")
print(f"\nTest input: {test_input}")

# ==============================================================================
# Part 1: HF Model Router
# ==============================================================================
print("\n" + "=" * 80)
print("Part 1: HuggingFace Model Router")
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

# Get embeddings and first norm
with torch.no_grad():
    hidden_states = hf_model.model.embed_tokens(test_input)  # (1, 4, 2880)
    hidden_states = hf_model.model.layers[0].input_layernorm(hidden_states)

    # Attention
    attn_output = hf_model.model.layers[0].self_attn(hidden_states)[0]
    hidden_states = hidden_states + attn_output

    # Pre-MLP norm
    residual = hidden_states
    hidden_states = hf_model.model.layers[0].post_attention_layernorm(hidden_states)

    print(f"\nMLP input shape: {hidden_states.shape}")
    print(f"MLP input [0, 0, :10]: {hidden_states[0, 0, :10].float()}")

    # Router
    router = hf_model.model.layers[0].mlp.router
    print(f"\nRouter weight shape: {router.weight.shape}")
    print(f"Router bias shape: {router.bias.shape if router.bias is not None else None}")

    # Flatten for router
    hidden_states_flat = hidden_states.flatten(0, 1)  # (4, 2880)
    router_logits, router_indices = router(hidden_states_flat)

    print(f"\nRouter logits shape: {router_logits.shape}")
    print(f"Router indices shape: {router_indices.shape}")
    print(f"\nFirst token router logits (all 32): {router_logits[0].float()}")
    print(f"First token top-4 indices: {router_indices[0]}")
    print(f"First token top-4 scores: {router_logits[0, router_indices[0]].float()}")

del hf_model
torch.cuda.empty_cache()

# ==============================================================================
# Part 2: Fast-LLM Model Router
# ==============================================================================
print("\n" + "=" * 80)
print("Part 2: Fast-LLM Model Router")
print("=" * 80)

gpt_model = GPTModel.from_pretrained(
    CheckpointLoadConfig(
        path=FAST_LLM_PATH,
        format=FastLLMCheckpointFormat,
        load_config=ModelConfigType.model,
    )
)
fast_llm_model = HuggingfaceGPTModelForCausalLM(gpt_model)

# Run forward to get internal activations
with torch.no_grad():
    output = fast_llm_model(test_input)

print(f"\nFast-LLM model config:")
print(f"  experts: {gpt_model.config.base_model.decoder.blocks.full.mlp.experts}")
print(f"  experts_per_token: {gpt_model.config.base_model.decoder.blocks.full.mlp.experts_per_token}")

print("\n" + "=" * 80)
