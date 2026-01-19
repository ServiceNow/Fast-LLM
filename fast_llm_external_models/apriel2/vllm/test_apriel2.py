#!/usr/bin/env python3
"""Test script for Apriel2 vLLM implementation.

This script tests coherence and numerical correctness of Apriel2 models
by comparing vLLM outputs with the reference Transformers implementation.

Usage:
    # Test coherence (generation quality)
    python test_apriel2.py coherence /path/to/model
    python test_apriel2.py coherence /path/to/model1 /path/to/model2

    # Compare logits between vLLM and Transformers
    python test_apriel2.py logits /path/to/model
    python test_apriel2.py logits /path/to/model --prompt "Custom prompt"

    # Run both tests
    python test_apriel2.py all /path/to/model
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
from vllm import LLM, ModelRegistry, SamplingParams
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)

# Ensure the parent package is importable
_script_dir = Path(__file__).parent
_package_root = _script_dir.parent.parent.parent
if str(_package_root) not in sys.path:
    sys.path.insert(0, str(_package_root))

# Register the Apriel2 model class at module level (required for subprocess)
from fast_llm_external_models.apriel2.vllm.modeling_apriel2 import Apriel2ForCausalLM  # noqa: E402
ModelRegistry.register_model(
    "Apriel2ForCausalLM",
    "fast_llm_external_models.apriel2.vllm:Apriel2ForCausalLM",
)


# Register config convertor at module level
class Apriel2TextModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def _get_first_attention_block(self):
        decoder = getattr(self.hf_text_config, 'decoder', {})
        decoder_type = decoder.get('type', 'fixed')
        if decoder_type == 'fixed':
            block = decoder.get('block', {})
            mixer = block.get('mixer', {})
            if mixer.get('type') == 'attention':
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


def setup_transformers():
    """Register Apriel2 model with Transformers."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2TextConfig
    from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForCausalLM

    AutoConfig.register("apriel2_text", Apriel2TextConfig)
    AutoModelForCausalLM.register(Apriel2TextConfig, Apriel2ForCausalLM)


def test_coherence_vllm(model_paths: list[str], prompts: list[str], max_tokens: int = 50):
    """Test generation coherence with vLLM."""
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0)

    results = {}
    for model_path in model_paths:
        model_name = Path(model_path).name
        print(f"\n{'#'*70}")
        print(f"# vLLM: {model_name}")
        print(f"{'#'*70}")

        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.4,
            max_model_len=2048,
        )

        outputs = llm.generate(prompts, sampling_params)
        results[model_name] = {}

        for output in outputs:
            prompt = output.prompt
            generated = output.outputs[0].text
            results[model_name][prompt] = generated
            print(f"\nPrompt: {prompt!r}")
            print(f"Output: {prompt + generated!r}")

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    return results


def test_coherence_transformers(model_paths: list[str], prompts: list[str], max_tokens: int = 50):
    """Test generation coherence with Transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup_transformers()

    results = {}
    for model_path in model_paths:
        model_name = Path(model_path).name
        print(f"\n{'#'*70}")
        print(f"# Transformers: {model_name}")
        print(f"{'#'*70}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        model.eval()

        results[model_name] = {}
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # Extract just the generated part (remove prompt)
            generated_only = generated[len(prompt):]
            results[model_name][prompt] = generated_only
            print(f"\nPrompt: {prompt!r}")
            print(f"Output: {generated!r}")

        del model
        torch.cuda.empty_cache()

    return results


def compare_logits(model_path: str, prompt: str, max_tokens: int = 1, dtype: str = "bfloat16"):
    """Compare logits between vLLM and Transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup_transformers()

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    print(f"\n{'='*70}")
    print(f"Model: {model_path}")
    print(f"Prompt: {prompt!r}")
    print(f"Dtype: {dtype}")
    print(f"{'='*70}\n")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"Input tokens: {input_ids.shape[1]}")
    print(f"Token IDs: {input_ids[0].tolist()}")

    # --- vLLM ---
    print(f"\n--- vLLM ({dtype}) ---")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        dtype=dtype,
        # enforce_eager=True,  # Disable torch.compile and CUDA graphs for debugging
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0,
        logprobs=20,
    )

    outputs = llm.generate([prompt], sampling_params)
    output = outputs[0]

    vllm_text = output.outputs[0].text
    vllm_token_ids = output.outputs[0].token_ids
    vllm_logprobs = output.outputs[0].logprobs

    print(f"Generated text: {vllm_text!r}")
    print(f"Generated token IDs: {vllm_token_ids}")

    vllm_first_token_id = None
    if vllm_token_ids:
        vllm_first_token_id = vllm_token_ids[0]
        vllm_first_token = tokenizer.decode([vllm_first_token_id])
        print(f"First generated token: {vllm_first_token!r} (id={vllm_first_token_id})")

        if vllm_logprobs and len(vllm_logprobs) > 0:
            print("Top-5 by logprob:")
            first_logprobs = vllm_logprobs[0]
            sorted_logprobs = sorted(first_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)[:5]
            for tid, lp in sorted_logprobs:
                token = tokenizer.decode([tid])
                print(f"  {token!r} (id={tid}): logprob={lp.logprob:.4f}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # --- Transformers ---
    # Use flash_attention_2 to match vLLM's attention backend (bf16 only)
    attn_impl = "flash_attention_2" if dtype == "bfloat16" else "eager"
    print(f"\n--- Transformers ({dtype}, {attn_impl}) ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.eval()

    with torch.no_grad():
        tf_outputs = model(input_ids.to("cuda"))
        tf_logits = tf_outputs.logits.cpu()

    print(f"Logits shape: {tf_logits.shape}")

    tf_next_token_logits = tf_logits[0, -1, :]
    tf_next_token_id = tf_next_token_logits.argmax().item()
    tf_next_token = tokenizer.decode([tf_next_token_id])
    print(f"Predicted next token: {tf_next_token!r} (id={tf_next_token_id})")

    tf_logprobs = torch.log_softmax(tf_next_token_logits.float(), dim=-1)
    print("Top-5 by logprob:")
    tf_top5 = tf_logprobs.topk(5)
    for i in range(5):
        tid = tf_top5.indices[i].item()
        lp = tf_top5.values[i].item()
        token = tokenizer.decode([tid])
        print(f"  {token!r} (id={tid}): logprob={lp:.4f}")

    del model
    torch.cuda.empty_cache()

    # --- Comparison ---
    print("\n--- Comparison ---")
    match = False
    if vllm_first_token_id is not None:
        if vllm_first_token_id == tf_next_token_id:
            print("MATCH: Both models predict the same next token!")
            match = True
        else:
            vllm_first_token = tokenizer.decode([vllm_first_token_id])
            print(f"MISMATCH: Transformers predicts {tf_next_token!r}, vLLM predicts {vllm_first_token!r}")

            tf_topk = tf_logprobs.topk(10)
            if vllm_first_token_id in tf_topk.indices.tolist():
                rank = tf_topk.indices.tolist().index(vllm_first_token_id)
                print(f"  vLLM's token is rank {rank+1} in transformers' predictions")
            else:
                print(f"  vLLM's token is NOT in transformers' top-10")

        # Compare logprobs
        if vllm_logprobs and len(vllm_logprobs) > 0:
            print("\n--- Logprob Comparison ---")
            first_logprobs = vllm_logprobs[0]

            common_tokens = set(first_logprobs.keys()) & set(range(len(tf_logprobs)))
            if common_tokens:
                diffs = []
                for tid in list(common_tokens)[:10]:
                    vllm_lp = first_logprobs[tid].logprob
                    tf_lp = tf_logprobs[tid].item()
                    diff = abs(vllm_lp - tf_lp)
                    diffs.append(diff)
                    if diff > 0.1:
                        token = tokenizer.decode([tid])
                        print(f"  {token!r}: vLLM={vllm_lp:.4f}, TF={tf_lp:.4f}, diff={diff:.4f}")

                avg_diff = sum(diffs) / len(diffs) if diffs else 0
                max_diff = max(diffs) if diffs else 0
                print(f"\n  Average logprob diff: {avg_diff:.6f}")
                print(f"  Max logprob diff: {max_diff:.6f}")

    return match, vllm_text, tf_next_token


def cmd_coherence(args):
    """Run coherence test."""
    prompts = [
        "The capital of France is",
        "To solve this math problem, I need to",
        "Once upon a time, there was a",
    ]

    print("\n" + "="*70)
    print("COHERENCE TEST: vLLM")
    print("="*70)
    vllm_results = test_coherence_vllm(args.model_paths, prompts, args.max_tokens)

    print("\n" + "="*70)
    print("COHERENCE TEST: Transformers")
    print("="*70)
    tf_results = test_coherence_transformers(args.model_paths, prompts, args.max_tokens)

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    for model_name in vllm_results:
        print(f"\n{model_name}:")
        for prompt in prompts:
            vllm_out = vllm_results[model_name].get(prompt, "")
            tf_out = tf_results[model_name].get(prompt, "")
            # Compare first 20 chars
            vllm_start = vllm_out[:20].strip()
            tf_start = tf_out[:20].strip()
            match = "MATCH" if vllm_start == tf_start else "DIFF"
            print(f"  [{match}] {prompt[:30]!r}...")
            if match == "DIFF":
                print(f"         vLLM: {vllm_start!r}...")
                print(f"         TF:   {tf_start!r}...")


def cmd_logits(args):
    """Run logits comparison test."""
    for model_path in args.model_paths:
        compare_logits(model_path, args.prompt, args.max_tokens, args.dtype)


def cmd_all(args):
    """Run all tests."""
    cmd_coherence(args)
    print("\n\n")
    cmd_logits(args)


def main():
    parser = argparse.ArgumentParser(
        description="Test Apriel2 vLLM implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Coherence test
    p_coherence = subparsers.add_parser("coherence", help="Test generation coherence")
    p_coherence.add_argument("model_paths", nargs="+", help="Path(s) to model checkpoint(s)")
    p_coherence.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    p_coherence.set_defaults(func=cmd_coherence)

    # Logits comparison
    p_logits = subparsers.add_parser("logits", help="Compare logits between vLLM and Transformers")
    p_logits.add_argument("model_paths", nargs="+", help="Path(s) to model checkpoint(s)")
    p_logits.add_argument("--prompt", default="The capital of France is", help="Input prompt")
    p_logits.add_argument("--max-tokens", type=int, default=1, help="Max tokens to generate")
    p_logits.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16", help="Data type")
    p_logits.set_defaults(func=cmd_logits)

    # All tests
    p_all = subparsers.add_parser("all", help="Run all tests")
    p_all.add_argument("model_paths", nargs="+", help="Path(s) to model checkpoint(s)")
    p_all.add_argument("--prompt", default="The capital of France is", help="Input prompt for logits test")
    p_all.add_argument("--max-tokens", type=int, default=50, help="Max tokens for coherence test")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
