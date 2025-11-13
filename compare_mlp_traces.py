#!/usr/bin/env python3
"""
Compare MLP component traces between HF and Fast-LLM using instrumented code.
"""

import pathlib

import torch
import transformers

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, FastLLMCheckpointFormat, ModelConfigType
from fast_llm.functional.triton import mlp as mlp_module
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
from fast_llm.models.gpt.model import GPTModel

CHECKPOINT_DIR = pathlib.Path("/home/ubuntu/Fast-LLM/test_gpt_oss_checkpoint")
DEQUANTIZED_HF_PATH = CHECKPOINT_DIR / "dequantized_hf"
FAST_LLM_PATH = CHECKPOINT_DIR / "fast_llm"

print("=" * 80)
print("Comparing MLP Traces: HF vs Fast-LLM")
print("=" * 80)

# Create small test input for detailed tracing
torch.manual_seed(42)
test_input = torch.randint(0, 201088, size=(1, 4), dtype=torch.int64, device="cuda")
print(f"\nTest input: {test_input}")

# ==============================================================================
# Part 1: HF Model - Manual Tracing
# ==============================================================================
print("\n" + "=" * 80)
print("Part 1: HuggingFace Model - Manual Component Tracing")
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

hf_traces = {}


def make_hook(name):
    def hook(module, input, output):
        if isinstance(input, tuple):
            hf_traces[f"{name}_input"] = input[0].detach().float()
        else:
            hf_traces[f"{name}_input"] = input.detach().float()
        if isinstance(output, tuple):
            hf_traces[f"{name}_output"] = output[0].detach().float()
        else:
            hf_traces[f"{name}_output"] = output.detach().float()

    return hook


layer0 = hf_model.model.layers[0]
layer0.post_attention_layernorm.register_forward_hook(make_hook("norm2"))
layer0.mlp.register_forward_hook(make_hook("mlp"))
layer0.mlp.experts.register_forward_hook(make_hook("experts"))

with torch.no_grad():
    hf_output = hf_model(test_input)

mlp_input = hf_traces["norm2_output"]

print(f"\n1. MLP Input (after norm2):")
print(f"   shape={mlp_input.shape}, mean={mlp_input.float().mean():.6f}, std={mlp_input.float().std():.6f}")
print(f"   first token [0, 0, :10]: {mlp_input[0, 0, :10].float()}")

# Manual MLP forward to trace components
mlp = layer0.mlp
experts = mlp.experts

with torch.no_grad():

    # Router (convert back to bfloat16 for HF model operations)
    mlp_input_bf16 = mlp_input.to(torch.bfloat16)
    router_scores, router_indices = mlp.router(mlp_input_bf16.flatten(0, 1))

    print(f"\n2. Router:")
    print(f"   scores shape={router_scores.shape}, indices shape={router_indices.shape}")
    print(f"   first token top-4 experts: {router_indices[0]}")
    print(f"   first token top-4 scores: {router_scores[0]}")

    # Process first token through first expert
    first_token = mlp_input_bf16[0, 0:1, :]  # (1, hidden_size)
    expert_idx = router_indices[0, 0].item()
    expert_score = router_scores[0, expert_idx].item()  # Get score for this specific expert

    print(f"\n3. Processing token through expert {expert_idx}:")

    # gate_up_proj
    gate_up = first_token @ experts.gate_up_proj[expert_idx] + experts.gate_up_proj_bias[expert_idx]
    print(f"   gate_up shape={gate_up.shape}, mean={gate_up.float().mean():.6f}, std={gate_up.float().std():.6f}")
    print(f"   gate_up [0, :10]: {gate_up[0, :10].float()}")

    # De-interleave
    gate = gate_up[..., 0::2]
    up = gate_up[..., 1::2]
    print(f"   gate mean={gate.float().mean():.6f}, std={gate.float().std():.6f}")
    print(f"   gate [0, :10]: {gate[0, :10].float()}")
    print(f"   up mean={up.float().mean():.6f}, std={up.float().std():.6f}")
    print(f"   up [0, :10]: {up[0, :10].float()}")

    # Activation
    alpha = 1.702
    limit = 7.0
    gate_clamped = gate.clamp(min=None, max=limit)
    up_clamped = up.clamp(min=-limit, max=limit)
    glu = gate_clamped * torch.sigmoid(gate_clamped * alpha)
    activated = (up_clamped + 1) * glu

    print(f"   activated mean={activated.float().mean():.6f}, std={activated.float().std():.6f}")
    print(f"   activated [0, :10]: {activated[0, :10].float()}")

    # down_proj
    down_out = activated @ experts.down_proj[expert_idx] + experts.down_proj_bias[expert_idx]
    weighted_out = down_out * expert_score

    print(f"   down_proj mean={down_out.float().mean():.6f}, std={down_out.float().std():.6f}")
    print(f"   down_proj [0, :10]: {down_out[0, :10].float()}")
    print(f"   weighted (score={expert_score:.4f}) [0, :10]: {weighted_out[0, :10].float()}")

    # Full MLP
    mlp_out, _ = mlp(mlp_input_bf16.flatten(0, 1))
    mlp_out = mlp_out.view_as(mlp_input_bf16)

    print(f"\n4. Full MLP output:")
    print(f"   shape={mlp_out.shape}, mean={mlp_out.float().mean():.6f}, std={mlp_out.float().std():.6f}")
    print(f"   first token [0, 0, :10]: {mlp_out[0, 0, :10].float()}")

del hf_model
torch.cuda.empty_cache()

# ==============================================================================
# Part 2: Fast-LLM Model - Using Instrumented Code
# ==============================================================================
print("\n" + "=" * 80)
print("Part 2: Fast-LLM Model - Instrumented Tracing")
print("=" * 80)

# Clear traces
mlp_module._MLP_DEBUG_TRACES.clear()


# Load GPT model first, then wrap
gpt_model = GPTModel.from_pretrained(
    CheckpointLoadConfig(
        path=FAST_LLM_PATH,
        format=FastLLMCheckpointFormat,
        load_config=ModelConfigType.model,
    )
)
fast_llm_model = HuggingfaceGPTModelForCausalLM(gpt_model)

with torch.no_grad():
    fl_output = fast_llm_model(test_input)

# Print Fast-LLM traces
traces = mlp_module._MLP_DEBUG_TRACES

print(f"\nFast-LLM traced:")
print(f"  - {len(traces.get('looped_inputs', []))} looped MLP calls")
print(f"  - {len(traces.get('mlp_inputs', []))} mlp_forward calls")

if traces.get("looped_inputs"):
    print(f"\n1. Looped MLP Input (first call, first token):")
    looped_in = traces["looped_inputs"][0]
    hidden = looped_in["hidden_states"]
    scores = looped_in["scores"]
    top_experts = looped_in["top_experts"]

    print(f"   hidden_states shape={hidden.shape}, mean={hidden.mean():.6f}, std={hidden.std():.6f}")
    print(f"   hidden_states [0, :10]: {hidden[0, :10]}")
    print(f"   top_experts: {top_experts[0]}")
    print(f"   scores: {scores[0]}")

if traces.get("looped_outputs"):
    print(f"\n2. Looped MLP Output (first call, first token):")
    looped_out = traces["looped_outputs"][0]
    print(f"   shape={looped_out.shape}, mean={looped_out.mean():.6f}, std={looped_out.std():.6f}")
    print(f"   [0, :10]: {looped_out[0, :10]}")

if traces.get("mlp_inputs"):
    print(f"\n1. MLP Forward Input (first call):")
    mlp_in = traces["mlp_inputs"][0]
    input_tensor = mlp_in["input"]
    scores_tensor = mlp_in["scores"]
    sparse_used = mlp_in["sparse_map_used"]

    print(f"   input shape={input_tensor.shape}, mean={input_tensor.mean():.6f}, std={input_tensor.std():.6f}")
    print(f"   input [0, :10]: {input_tensor[0, :10]}")
    if scores_tensor is not None:
        print(f"   scores shape={scores_tensor.shape}: {scores_tensor[0]}")
    print(f"   sparse_map used: {sparse_used}")

if traces.get("mlp_outputs"):
    print(f"\n2. MLP Forward Output (first call):")
    mlp_out = traces["mlp_outputs"][0]
    output_tensor = mlp_out["output"]
    out_shape = mlp_out["shape"]

    print(f"   output full shape={out_shape}")
    print(
        f"   output (first token) shape={output_tensor.shape}, mean={output_tensor.mean():.6f}, std={output_tensor.std():.6f}"
    )
    print(f"   output [0, :10]: {output_tensor[0, :10]}")

print("\n" + "=" * 80)
print("✅ Tracing complete! Compare the values above.")
print("=" * 80)
