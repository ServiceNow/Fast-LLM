#!/usr/bin/env python3
"""
Add hooks to both HF and Fast-LLM MLP to trace intermediate values.
"""

import pathlib

import torch
import transformers

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, FastLLMCheckpointFormat, ModelConfigType

# Monkey-patch the mlp_autograd_looped to add tracing
from fast_llm.functional.triton import mlp as mlp_module
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM

CHECKPOINT_DIR = pathlib.Path("/home/ubuntu/Fast-LLM/test_gpt_oss_checkpoint")
DEQUANTIZED_HF_PATH = CHECKPOINT_DIR / "dequantized_hf"
FAST_LLM_PATH = CHECKPOINT_DIR / "fast_llm"

print("=" * 80)
print("Tracing MLP Components with Hooks")
print("=" * 80)

# Create test input
torch.manual_seed(42)
test_input = torch.randint(0, 201088, size=(1, 4), dtype=torch.int64, device="cuda")  # Smaller for detailed tracing
print(f"\nTest input shape: {test_input.shape}")
print(f"Test input: {test_input}")

# ==============================================================================
# Part 1: Trace HuggingFace Model
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

hf_traces = {}


# Hook the MLP experts to trace gate_up and activation
def make_hf_experts_hook():
    def hook(module, input, output):
        # Save the input to experts
        hf_traces["experts_input"] = input[0].detach().float()
        hf_traces["experts_output"] = output.detach().float()

    return hook


hf_model.model.layers[0].mlp.experts.register_forward_hook(make_hf_experts_hook())

print("\nRunning HF model...")
with torch.no_grad():
    hf_output = hf_model(test_input)

print(
    f"HF experts input: shape={hf_traces['experts_input'].shape}, mean={hf_traces['experts_input'].mean():.6f}, std={hf_traces['experts_input'].std():.6f}"
)
print(
    f"HF experts output: shape={hf_traces['experts_output'].shape}, mean={hf_traces['experts_output'].mean():.6f}, std={hf_traces['experts_output'].std():.6f}"
)
print(
    f"HF final logits: shape={hf_output.logits.shape}, mean={hf_output.logits.mean():.6f}, std={hf_output.logits.std():.6f}"
)

# Save for comparison
hf_logits = hf_output.logits.clone().cpu()

del hf_model
torch.cuda.empty_cache()

# ==============================================================================
# Part 2: Trace Fast-LLM Model
# ==============================================================================
print("\n" + "=" * 80)
print("Part 2: Fast-LLM Model")
print("=" * 80)


original_mlp_autograd_looped = mlp_module.mlp_autograd_looped
fl_traces = {}


def traced_mlp_autograd_looped(
    hidden_states,
    scores,
    top_experts,
    weight_1,
    weight_2,
    num_experts,
    gated,
    activation_type,
    group,
    sequence_parallel,
    training,
    recompute_level,
    bias_1=None,
    bias_2=None,
):
    # Save inputs
    fl_traces["mlp_input"] = hidden_states.detach().clone().cpu()
    fl_traces["scores"] = scores.detach().clone().cpu()
    fl_traces["top_experts"] = top_experts.detach().clone().cpu()

    # Call original
    result = original_mlp_autograd_looped(
        hidden_states,
        scores,
        top_experts,
        weight_1,
        weight_2,
        num_experts,
        gated,
        activation_type,
        group,
        sequence_parallel,
        training,
        recompute_level,
        bias_1,
        bias_2,
    )

    # Save output
    fl_traces["mlp_output"] = result.detach().clone().cpu()

    return result


mlp_module.mlp_autograd_looped = traced_mlp_autograd_looped

fast_llm_model = HuggingfaceGPTModelForCausalLM.from_pretrained(
    CheckpointLoadConfig(
        path=FAST_LLM_PATH,
        format=FastLLMCheckpointFormat,
        load_config=ModelConfigType.model,
    )
)

print("\nRunning Fast-LLM model...")
with torch.no_grad():
    fl_output = fast_llm_model(test_input)

print(
    f"FL MLP input: shape={fl_traces['mlp_input'].shape}, mean={fl_traces['mlp_input'].mean():.6f}, std={fl_traces['mlp_input'].std():.6f}"
)
print(
    f"FL scores: shape={fl_traces['scores'].shape}, mean={fl_traces['scores'].mean():.6f}, std={fl_traces['scores'].std():.6f}"
)
print(f"FL top_experts: shape={fl_traces['top_experts'].shape}")
print(f"FL top_experts: {fl_traces['top_experts']}")
print(
    f"FL MLP output: shape={fl_traces['mlp_output'].shape}, mean={fl_traces['mlp_output'].mean():.6f}, std={fl_traces['mlp_output'].std():.6f}"
)
print(
    f"FL final logits: shape={fl_output.logits.shape}, mean={fl_output.logits.mean():.6f}, std={fl_output.logits.std():.6f}"
)

# Compare
print("\n" + "=" * 80)
print("Comparison")
print("=" * 80)

print(f"\nMLP input mean: HF={hf_traces['experts_input'].mean():.6f}, FL={fl_traces['mlp_input'].mean():.6f}")
print(f"MLP input std: HF={hf_traces['experts_input'].std():.6f}, FL={fl_traces['mlp_input'].std():.6f}")
print(f"MLP output mean: HF={hf_traces['experts_output'].mean():.6f}, FL={fl_traces['mlp_output'].mean():.6f}")
print(f"MLP output std: HF={hf_traces['experts_output'].std():.6f}, FL={fl_traces['mlp_output'].std():.6f}")

fl_logits = fl_output.logits.cpu()
hf_logits = hf_logits.cuda()
fl_logits_gpu = fl_output.logits

print(f"\nFinal logits mean: HF={hf_logits.float().mean():.6f}, FL={fl_logits.mean():.6f}")
print(f"Final logits std: HF={hf_logits.float().std():.6f}, FL={fl_logits.std():.6f}")
print(f"Logits max diff: {(hf_logits.float() - fl_logits_gpu.float()).abs().max():.6f}")
print(f"Logits RMS diff: {((hf_logits.float() - fl_logits_gpu.float()) ** 2).mean().sqrt():.6f}")
