#!/usr/bin/env python3
"""Compare dev model (with placement switching) against fixed architecture models.

This validates that setting the dev model's placement to match a fixed model
produces equivalent outputs.

Usage:
    python test_placement_comparison.py
"""

import gc
import sys
from pathlib import Path

import torch
import triton

def _triton_allocator(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device='cuda').data_ptr()

triton.set_allocator(_triton_allocator)

from vllm import LLM, SamplingParams, ModelRegistry
from vllm.config import CompilationConfig
from vllm.config.compilation import CompilationMode
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)


class Apriel2TextModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def _get_first_attention_block(self):
        decoder = getattr(self.hf_text_config, 'decoder', {})
        decoder_type = decoder.get('type', 'fixed')
        if decoder_type == 'fixed':
            block = decoder.get('block', {})
            mixer = block.get('mixer', {})
            mixer_type = mixer.get('type', 'attention')
            if mixer_type == 'stochastic':
                main_mixer_name = mixer.get('main_mixer_name', 'attention')
                return mixer.get('mixers', {}).get(main_mixer_name, {})
            elif mixer_type == 'attention':
                return mixer
        elif decoder_type == 'pattern':
            blocks = decoder.get('blocks', {})
            pattern = decoder.get('pattern', [])
            for block_name in pattern:
                block = blocks.get(block_name, {})
                mixer = block.get('mixer', {})
                if mixer.get('type') == 'attention':
                    return mixer
        return {}

    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, 'decoder', {}).get('num_blocks', 0)

    def get_total_num_attention_heads(self) -> int:
        return self._get_first_attention_block().get('heads', 0)

    def get_total_num_kv_heads(self) -> int:
        return self._get_first_attention_block().get('head_groups', self.get_total_num_attention_heads())

    def get_head_size(self) -> int:
        return self._get_first_attention_block().get('head_size', 0)


MODEL_ARCH_CONFIG_CONVERTORS['apriel2_text'] = Apriel2TextModelArchConfigConvertor
MODEL_ARCH_CONFIG_CONVERTORS['apriel2'] = Apriel2TextModelArchConfigConvertor


# Ensure the parent package is importable
_script_dir = Path(__file__).parent
_package_root = _script_dir.parent.parent.parent
if str(_package_root) not in sys.path:
    sys.path.insert(0, str(_package_root))

from fast_llm_external_models.apriel2.vllm.modeling_apriel2 import Apriel2ForCausalLM
ModelRegistry.register_model(
    "Apriel2ForCausalLM",
    "fast_llm_external_models.apriel2.vllm:Apriel2ForCausalLM",
)


def run_inference(llm, prompts, max_tokens=10):
    """Run inference and return generated text and token IDs."""
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0, logprobs=5)
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for out in outputs:
        results.append({
            "text": out.outputs[0].text,
            "token_ids": list(out.outputs[0].token_ids),
            "logprobs": out.outputs[0].logprobs,
        })
    return results


def compare_results(results1, results2, name1, name2):
    """Compare two sets of results."""
    print(f"\n{'='*70}")
    print(f"Comparing {name1} vs {name2}")
    print(f"{'='*70}")

    matches = 0
    total = len(results1)

    for i, (r1, r2) in enumerate(zip(results1, results2)):
        text_match = r1["text"] == r2["text"]
        token_match = r1["token_ids"] == r2["token_ids"]

        if text_match and token_match:
            matches += 1
            print(f"  Prompt {i}: MATCH - '{r1['text'][:50]}...'")
        else:
            print(f"  Prompt {i}: DIFF")
            print(f"    {name1}: {r1['token_ids'][:5]} -> '{r1['text'][:30]}'")
            print(f"    {name2}: {r2['token_ids'][:5]} -> '{r2['text'][:30]}'")

            # Compare logprobs for first token
            if r1["logprobs"] and r2["logprobs"]:
                lp1 = r1["logprobs"][0]
                lp2 = r2["logprobs"][0]

                # Find common tokens and compare
                common = set(lp1.keys()) & set(lp2.keys())
                if common:
                    diffs = []
                    for tid in list(common)[:5]:
                        diff = abs(lp1[tid].logprob - lp2[tid].logprob)
                        diffs.append(diff)
                    print(f"    Logprob diff (top-5 common): avg={sum(diffs)/len(diffs):.4f}, max={max(diffs):.4f}")

    print(f"\nMatch rate: {matches}/{total} ({100*matches/total:.1f}%)")
    return matches == total


def main():
    # Test prompts
    prompts = [
        "The capital of France is",
        "In machine learning, the gradient descent algorithm",
        "The quick brown fox jumps over",
        "def fibonacci(n):\n    if n <= 1:\n        return",
        "To solve this equation, we need to",
    ]

    dev_model = "/tmp/apriel2-0.5b-dev"
    fixed_model = "/tmp/apriel2-0.5b-every2nd-gdn"

    # Every 2nd layer is GDN: attention, gdn, attention, gdn, ...
    every2nd_placement = ["attention", "gdn"] * 12

    compilation_config = CompilationConfig(mode=CompilationMode.NONE)

    # ========== Run fixed model first ==========
    print(f"\n{'#'*70}")
    print(f"# Loading FIXED model: {fixed_model}")
    print(f"{'#'*70}")

    llm_fixed = LLM(
        model=fixed_model,
        trust_remote_code=True,
        gpu_memory_utilization=0.3,
        max_model_len=512,
        dtype="bfloat16",
        compilation_config=compilation_config,
        disable_log_stats=True,
        enable_prefix_caching=False,
    )

    print("\nRunning inference on fixed model...")
    fixed_results = run_inference(llm_fixed, prompts)

    del llm_fixed
    gc.collect()
    torch.cuda.empty_cache()

    # ========== Run dev model with placement switching ==========
    print(f"\n{'#'*70}")
    print(f"# Loading DEV model: {dev_model}")
    print(f"# Will set placement to: every 2nd GDN")
    print(f"{'#'*70}")

    llm_dev = LLM(
        model=dev_model,
        trust_remote_code=True,
        gpu_memory_utilization=0.3,
        max_model_len=512,
        dtype="bfloat16",
        compilation_config=compilation_config,
        disable_log_stats=True,
        enable_prefix_caching=False,
    )

    # Get initial placement
    initial_placement = llm_dev.collective_rpc("get_layer_placements")
    print(f"\nInitial placement: all {list(initial_placement[0].values())[0]}")

    # Switch to every2nd-gdn pattern
    print(f"Switching to every2nd-gdn pattern...")
    llm_dev.collective_rpc("set_layer_placements", args=(every2nd_placement,))

    # Verify
    new_placement = llm_dev.collective_rpc("get_layer_placements")
    attn_count = sum(1 for v in new_placement[0].values() if v == "attention")
    gdn_count = sum(1 for v in new_placement[0].values() if v == "gdn")
    print(f"New placement: {attn_count} attention, {gdn_count} gdn")

    print("\nRunning inference on dev model (with every2nd-gdn placement)...")
    dev_results = run_inference(llm_dev, prompts)

    del llm_dev
    gc.collect()
    torch.cuda.empty_cache()

    # ========== Compare ==========
    all_match = compare_results(
        fixed_results, dev_results,
        "fixed-every2nd-gdn", "dev-with-every2nd-placement"
    )

    print(f"\n{'='*70}")
    if all_match:
        print("SUCCESS: Dev model with placement switching matches fixed model!")
    else:
        print("WARNING: Some differences detected between models.")
        print("This may be expected if the weights differ between checkpoints.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
