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

    # Statistical comparison with many prompts (for rigorous testing)
    python test_apriel2.py stats /path/to/model --num-prompts 128
    python test_apriel2.py stats /path/to/model --num-prompts 64 --min-tokens 256 --no-compile

    # Run both tests
    python test_apriel2.py all /path/to/model
"""

import argparse
import gc
import sys
from pathlib import Path
import numpy as np

import torch
import triton

# Set a triton allocator to avoid "no allocator was set" errors
def _triton_allocator(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device='cuda').data_ptr()

triton.set_allocator(_triton_allocator)

from vllm import LLM, ModelRegistry, SamplingParams
from vllm.config import CompilationConfig
from vllm.config.compilation import CompilationMode
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


def compare_logits(model_path: str, prompt: str, max_tokens: int = 1, dtype: str = "bfloat16", no_compile: bool = False, revision: str | None = None, debug_gdn: bool = False):
    """Compare logits between vLLM and Transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup_transformers()

    # Enable GDN debug if requested
    if debug_gdn:
        # vLLM GDN class
        from fast_llm_external_models.apriel2.vllm.modeling_apriel2 import Apriel2GatedDeltaNet as VLLMGatedDeltaNet
        VLLMGatedDeltaNet._debug_global_enable = True
        print("GDN debug mode enabled for vLLM")

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    print(f"\n{'='*70}")
    print(f"Model: {model_path}")
    print(f"Revision: {revision}")
    print(f"Prompt: {prompt!r}")
    print(f"Dtype: {dtype}")
    print(f"No compile: {no_compile}")
    print(f"Debug GDN: {debug_gdn}")
    print(f"{'='*70}\n")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, trust_remote_code=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"Input tokens: {input_ids.shape[1]}")
    print(f"Token IDs: {input_ids[0].tolist()}")

    # --- vLLM ---
    compile_label = "no-compile" if no_compile else "compiled"
    print(f"\n--- vLLM ({dtype}, {compile_label}) ---")
    compilation_config = CompilationConfig(mode=CompilationMode.NONE) if no_compile else None
    llm = LLM(
        model=model_path,
        revision=revision,
        trust_remote_code=True,
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        dtype=dtype,
        compilation_config=compilation_config,
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
        revision=revision,
        torch_dtype=torch_dtype,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.eval()

    # Enable debug on transformers GDN layers if requested
    if debug_gdn:
        for name, module in model.named_modules():
            if module.__class__.__name__ == "Apriel2GatedDeltaNet":
                module._debug_enabled = True  # Enable at instance level (TF doesn't have warmup filtering)
                print(f"Enabled debug on {name}")

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


def compare_comprehensive(
    model_path: str,
    prompt_sizes: list[int],
    decode_lengths: list[int],
    batch_sizes: list[int],
    dtype: str = "bfloat16",
    no_compile: bool = True,
    revision: str | None = None,
):
    """Compare vLLM and Transformers across various configurations.

    Returns a list of result dicts with keys:
        prompt_size, decode_length, batch_size, avg_diff, max_diff, all_match
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup_transformers()

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, trust_remote_code=True)

    # Generate prompts of different sizes using a base text
    base_text = (
        "The study of artificial intelligence has evolved significantly over the past decades. "
        "Machine learning, a subset of AI, focuses on developing algorithms that can learn from data. "
        "Deep learning, in turn, uses neural networks with many layers to model complex patterns. "
        "Natural language processing enables computers to understand and generate human language. "
        "Computer vision allows machines to interpret and analyze visual information from the world. "
        "Reinforcement learning trains agents to make decisions by rewarding desired behaviors. "
        "The field continues to advance rapidly, with new breakthroughs occurring frequently. "
        "Applications range from autonomous vehicles to medical diagnosis and beyond. "
        "Ethical considerations around AI development have become increasingly important. "
        "Researchers work to ensure AI systems are fair, transparent, and beneficial to society. "
    )

    # Repeat base text to get enough tokens
    long_text = (base_text * 20)[:8000]  # Plenty of text

    def get_prompt_with_tokens(target_tokens: int) -> str:
        """Get a prompt with approximately target_tokens tokens."""
        # Binary search for right length
        low, high = 1, len(long_text)
        while low < high:
            mid = (low + high) // 2
            test_prompt = long_text[:mid]
            num_tokens = len(tokenizer.encode(test_prompt))
            if num_tokens < target_tokens:
                low = mid + 1
            else:
                high = mid
        return long_text[:low]

    results = []

    # Load vLLM once
    print(f"\n{'='*70}")
    print(f"Loading vLLM model: {model_path}")
    print(f"{'='*70}")

    compilation_config = CompilationConfig(mode=CompilationMode.NONE) if no_compile else None
    llm = LLM(
        model=model_path,
        revision=revision,
        trust_remote_code=True,
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        dtype=dtype,
        compilation_config=compilation_config,
    )

    # Load Transformers once
    print(f"\nLoading Transformers model...")
    attn_impl = "flash_attention_2" if dtype == "bfloat16" else "eager"
    tf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        torch_dtype=torch_dtype,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    tf_model.eval()

    print(f"\n{'='*70}")
    print(f"Running comparisons...")
    print(f"{'='*70}\n")

    # Header
    print(f"{'Prompt':<8} {'Decode':<8} {'Batch':<8} {'Avg Diff':<12} {'Max Diff':<12} {'Match':<8}")
    print("-" * 60)

    for prompt_size in prompt_sizes:
        prompt = get_prompt_with_tokens(prompt_size)
        actual_tokens = len(tokenizer.encode(prompt))

        for decode_length in decode_lengths:
            for batch_size in batch_sizes:
                # Create batch of prompts
                prompts = [prompt] * batch_size

                # vLLM inference
                sampling_params = SamplingParams(
                    max_tokens=decode_length,
                    temperature=0,
                    logprobs=20,
                )
                vllm_outputs = llm.generate(prompts, sampling_params)

                # Transformers inference
                input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to("cuda")

                with torch.no_grad():
                    if decode_length == 1:
                        # Just get logits for next token prediction
                        tf_outputs = tf_model(input_ids)
                        tf_logits = tf_outputs.logits
                    else:
                        # Generate multiple tokens
                        tf_output_ids = tf_model.generate(
                            input_ids,
                            max_new_tokens=decode_length,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            return_dict_in_generate=True,
                            output_logits=True,
                        )
                        # Stack logits from generation steps
                        tf_logits = torch.stack(tf_output_ids.logits, dim=1)

                # Compare logprobs for each position and batch element
                all_diffs = []
                all_match = True

                for b in range(batch_size):
                    vllm_out = vllm_outputs[b]
                    vllm_logprobs_list = vllm_out.outputs[0].logprobs or []
                    vllm_token_ids = vllm_out.outputs[0].token_ids

                    for pos in range(min(decode_length, len(vllm_logprobs_list))):
                        vllm_logprobs = vllm_logprobs_list[pos]

                        if decode_length == 1:
                            # For prefill, use last position
                            tf_pos_logits = tf_logits[b, -1, :]
                        else:
                            # For generation, use corresponding position
                            tf_pos_logits = tf_logits[b, pos, :]

                        tf_pos_logprobs = torch.log_softmax(tf_pos_logits.float(), dim=-1)

                        # Get TF's predicted token
                        tf_pred_token = tf_pos_logprobs.argmax().item()
                        vllm_pred_token = vllm_token_ids[pos] if pos < len(vllm_token_ids) else None

                        if vllm_pred_token != tf_pred_token:
                            if all_match:  # First mismatch
                                vllm_lp_for_tok = vllm_logprobs.get(vllm_pred_token, None)
                                vllm_lp_val = vllm_lp_for_tok.logprob if vllm_lp_for_tok else "N/A"
                                tf_lp_vllm_tok = tf_pos_logprobs[vllm_pred_token].item() if vllm_pred_token and vllm_pred_token < len(tf_pos_logprobs) else "N/A"
                                tf_lp_tf_tok = tf_pos_logprobs[tf_pred_token].item()
                                print(f"  FIRST MISMATCH at pos {pos}: vLLM tok={vllm_pred_token} (lp={vllm_lp_val}), TF tok={tf_pred_token} (lp={tf_lp_tf_tok:.4f})")
                                print(f"    TF logprob for vLLM's token: {tf_lp_vllm_tok}")
                            all_match = False

                        # Compare logprobs for common tokens
                        for tid, lp_info in vllm_logprobs.items():
                            if tid < len(tf_pos_logprobs):
                                vllm_lp = lp_info.logprob
                                tf_lp = tf_pos_logprobs[tid].item()
                                diff = abs(vllm_lp - tf_lp)
                                all_diffs.append(diff)

                avg_diff = sum(all_diffs) / len(all_diffs) if all_diffs else 0
                max_diff = max(all_diffs) if all_diffs else 0
                match_str = "YES" if all_match else "NO"

                result = {
                    "prompt_size": actual_tokens,
                    "decode_length": decode_length,
                    "batch_size": batch_size,
                    "avg_diff": avg_diff,
                    "max_diff": max_diff,
                    "all_match": all_match,
                }
                results.append(result)

                print(f"{actual_tokens:<8} {decode_length:<8} {batch_size:<8} {avg_diff:<12.6f} {max_diff:<12.6f} {match_str:<8}")

    # Cleanup
    del llm
    del tf_model
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_avg = sum(r["avg_diff"] for r in results) / len(results)
    all_max = max(r["max_diff"] for r in results)
    all_matched = all(r["all_match"] for r in results)
    print(f"Overall average diff: {all_avg:.6f}")
    print(f"Overall max diff: {all_max:.6f}")
    print(f"All predictions match: {'YES' if all_matched else 'NO'}")

    return results


def cmd_compare(args):
    """Run comprehensive comparison across configurations."""
    prompt_sizes = [int(x) for x in args.prompt_sizes.split(",")]
    decode_lengths = [int(x) for x in args.decode_lengths.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    for model_path in args.model_paths:
        compare_comprehensive(
            model_path,
            prompt_sizes=prompt_sizes,
            decode_lengths=decode_lengths,
            batch_sizes=batch_sizes,
            dtype=args.dtype,
            no_compile=args.no_compile,
            revision=getattr(args, 'revision', None),
        )


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


# ============================================================================
# Statistical Testing Infrastructure (v2)
# ============================================================================
#
# Design goals:
# 1. Dataset-based prompts (C4) for reproducibility
# 2. Controlled tokenization - same token IDs to both backends
# 3. Per-position statistics (prefill + each decode step)
# 4. Configurable Transformers kernel selection
# 5. Full parameter space: prompts, prompt_length, decode_length, batch_size, compile, kernels

from dataclasses import dataclass, field
from itertools import islice


@dataclass
class TokenComparison:
    """Comparison data for a single token position."""
    prompt_idx: int
    position: int  # 0 = prefill (last token), 1+ = decode steps
    vllm_token_id: int
    tf_token_id: int
    token_match: bool
    avg_logprob_diff: float
    max_logprob_diff: float
    top_k_diffs: list[float] = field(default_factory=list)


def load_and_tokenize_prompts(
    num_prompts: int,
    prompt_length: int,
    tokenizer,
    seed: int = 42,
) -> list[list[int]]:
    """Load prompts from C4 dataset and tokenize to exact length.

    Streams through shuffled dataset until we find exactly num_prompts
    that have at least prompt_length tokens.

    Args:
        num_prompts: Number of prompts to collect
        prompt_length: Exact number of tokens per prompt
        tokenizer: Tokenizer to use
        seed: Random seed for shuffling

    Returns:
        List of token ID lists, all exactly prompt_length long
    """
    from datasets import load_dataset

    print(f"Loading C4 dataset (streaming, seed={seed})...")
    dataset = load_dataset('allenai/c4', 'en', split='train', streaming=True)

    # Shuffle with seed for reproducibility
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    token_ids_list = []
    samples_checked = 0

    for sample in dataset:
        samples_checked += 1
        text = sample['text']

        # Tokenize and check length
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= prompt_length:
            token_ids_list.append(tokens[:prompt_length])

            if len(token_ids_list) >= num_prompts:
                break

        # Progress every 100 samples
        if samples_checked % 100 == 0:
            print(f"  Checked {samples_checked} samples, found {len(token_ids_list)}/{num_prompts} valid prompts", end="\r")

    print(f"  Checked {samples_checked} samples, found {len(token_ids_list)}/{num_prompts} valid prompts")

    if len(token_ids_list) < num_prompts:
        print(f"  Warning: Only found {len(token_ids_list)} prompts with >= {prompt_length} tokens")

    return token_ids_list


def set_transformers_kernels(model_path: str, kernel_config: str) -> None:
    """Set kernel configuration in Transformers modeling file.

    Args:
        model_path: Path to model (to find modeling file)
        kernel_config: 'upstream' or 'vllm'
    """
    import importlib
    import sys

    # The Transformers model uses the local modeling_apriel2.py in the checkpoint
    # We need to modify its flags before loading
    modeling_path = Path(model_path) / "modeling_apriel2.py"
    if not modeling_path.exists():
        print(f"  Warning: No modeling_apriel2.py found at {model_path}")
        return

    # Read the file
    content = modeling_path.read_text()

    # Set the flags based on kernel_config
    if kernel_config == "upstream":
        new_content = content.replace("USE_VLLM_CONV = True", "USE_VLLM_CONV = False")
        new_content = new_content.replace("USE_VLLM_GDN_OPS = True", "USE_VLLM_GDN_OPS = False")
        new_content = new_content.replace("USE_VLLM_GATED_NORM = True", "USE_VLLM_GATED_NORM = False")
    elif kernel_config == "vllm":
        new_content = content.replace("USE_VLLM_CONV = False", "USE_VLLM_CONV = True")
        new_content = new_content.replace("USE_VLLM_GDN_OPS = False", "USE_VLLM_GDN_OPS = True")
        new_content = new_content.replace("USE_VLLM_GATED_NORM = False", "USE_VLLM_GATED_NORM = True")
    else:
        raise ValueError(f"Unknown kernel_config: {kernel_config}")

    if new_content != content:
        modeling_path.write_text(new_content)
        print(f"  Set Transformers kernels to: {kernel_config}")

        # Clear any cached imports
        modules_to_remove = [k for k in sys.modules if 'apriel2' in k.lower()]
        for mod in modules_to_remove:
            del sys.modules[mod]


def run_vllm_inference(
    model_path: str,
    token_ids_list: list[list[int]],
    decode_length: int,
    dtype: str,
    no_compile: bool,
    revision: str | None,
) -> tuple[list[list[int]], list[list[dict]]]:
    """Run vLLM inference and return generated tokens and logprobs.

    Returns:
        - generated_tokens: list of list of token IDs (one list per prompt)
        - logprobs_per_position: list of list of logprob dicts (one list per prompt)
    """
    from vllm import TokensPrompt

    compile_label = "no-compile" if no_compile else "compiled"
    print(f"\nLoading vLLM model ({compile_label})...")
    compilation_config = CompilationConfig(mode=CompilationMode.NONE) if no_compile else None

    llm = LLM(
        model=model_path,
        revision=revision,
        trust_remote_code=True,
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        dtype=dtype,
        compilation_config=compilation_config,
    )

    # Create TokensPrompt for each prompt
    vllm_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in token_ids_list]

    sampling_params = SamplingParams(
        max_tokens=decode_length,
        temperature=0,
        logprobs=20,
    )

    print(f"Running vLLM inference on {len(vllm_prompts)} prompts (decode_length={decode_length})...")
    outputs = llm.generate(vllm_prompts, sampling_params)

    # Extract results
    generated_tokens = []
    logprobs_per_position = []

    for output in outputs:
        tokens = list(output.outputs[0].token_ids) if output.outputs[0].token_ids else []
        generated_tokens.append(tokens)

        # Logprobs for each position
        lps = []
        if output.outputs[0].logprobs:
            for pos_lps in output.outputs[0].logprobs:
                lps.append(pos_lps if pos_lps else {})
        logprobs_per_position.append(lps)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return generated_tokens, logprobs_per_position


def run_transformers_inference(
    model_path: str,
    token_ids_list: list[list[int]],
    decode_length: int,
    dtype: str,
    revision: str | None,
) -> tuple[list[list[int]], list[list[torch.Tensor]]]:
    """Run Transformers inference and return generated tokens and logprobs.

    Returns:
        - generated_tokens: list of list of token IDs (one list per prompt)
        - logprobs_per_position: list of list of logprob tensors (one list per prompt)
    """
    from transformers import AutoModelForCausalLM

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    attn_impl = "flash_attention_2" if dtype == "bfloat16" else "eager"

    print(f"\nLoading Transformers model ({attn_impl})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        torch_dtype=torch_dtype,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.eval()

    generated_tokens = []
    logprobs_per_position = []

    print(f"Running Transformers inference on {len(token_ids_list)} prompts...")

    for i, token_ids in enumerate(token_ids_list):
        input_ids = torch.tensor([token_ids], device="cuda")
        prompt_tokens = []
        prompt_logprobs = []

        with torch.no_grad():
            # Generate tokens one at a time to get logprobs at each step
            for step in range(decode_length):
                outputs = model(input_ids)
                logits = outputs.logits[:, -1, :]  # Last position
                logprobs = torch.log_softmax(logits.float(), dim=-1).cpu()

                next_token = logits.argmax(dim=-1).item()
                prompt_tokens.append(next_token)
                prompt_logprobs.append(logprobs[0])

                # Append to input for next step
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device="cuda")], dim=1)

        generated_tokens.append(prompt_tokens)
        logprobs_per_position.append(prompt_logprobs)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(token_ids_list)} prompts", end="\r")

    print()

    del model
    torch.cuda.empty_cache()

    return generated_tokens, logprobs_per_position


def compute_comparisons(
    vllm_tokens: list[list[int]],
    vllm_logprobs: list[list[dict]],
    tf_tokens: list[list[int]],
    tf_logprobs: list[list[torch.Tensor]],
) -> list[TokenComparison]:
    """Compute per-position comparisons between vLLM and Transformers."""
    comparisons = []

    for prompt_idx, (vt, vl, tt, tl) in enumerate(zip(vllm_tokens, vllm_logprobs, tf_tokens, tf_logprobs)):
        # Compare each position
        for pos in range(min(len(vt), len(tt), len(vl), len(tl))):
            vllm_token = vt[pos]
            tf_token = tt[pos]
            vllm_lps = vl[pos]
            tf_lps = tl[pos]

            # Compute logprob differences for top-K tokens
            diffs = []
            if vllm_lps:
                for tid, lp_info in list(vllm_lps.items())[:20]:
                    vllm_lp = lp_info.logprob
                    tf_lp = tf_lps[tid].item()
                    diffs.append(abs(vllm_lp - tf_lp))

            avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
            max_diff = max(diffs) if diffs else 0.0

            comparisons.append(TokenComparison(
                prompt_idx=prompt_idx,
                position=pos,
                vllm_token_id=vllm_token,
                tf_token_id=tf_token,
                token_match=(vllm_token == tf_token),
                avg_logprob_diff=avg_diff,
                max_logprob_diff=max_diff,
                top_k_diffs=diffs,
            ))

    return comparisons


def print_stats_report(comparisons: list[TokenComparison], title: str = "Statistics"):
    """Print comprehensive statistics from comparisons."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")

    if not comparisons:
        print("No comparisons to report.")
        return {}

    # Group by position
    by_position: dict[int, list[TokenComparison]] = {}
    for c in comparisons:
        by_position.setdefault(c.position, []).append(c)

    # Overall stats
    all_avg_diffs = np.array([c.avg_logprob_diff for c in comparisons])
    all_max_diffs = np.array([c.max_logprob_diff for c in comparisons])
    all_matches = np.array([c.token_match for c in comparisons])

    n_total = len(comparisons)
    n_prompts = len(set(c.prompt_idx for c in comparisons))
    n_positions = len(by_position)

    print(f"\nTotal comparisons: {n_total} ({n_prompts} prompts x {n_positions} positions)")
    print(f"Token match rate: {all_matches.sum()}/{n_total} ({100*all_matches.mean():.1f}%)")

    # Per-position stats
    print(f"\n--- Per-Position Statistics ---")
    print(f"{'Pos':>4} {'N':>6} {'Match%':>8} {'AvgDiff':>10} {'p50':>8} {'p95':>8} {'Max':>8}")
    print("-" * 60)

    position_stats = {}
    for pos in sorted(by_position.keys()):
        pos_comparisons = by_position[pos]
        pos_diffs = np.array([c.avg_logprob_diff for c in pos_comparisons])
        pos_matches = np.array([c.token_match for c in pos_comparisons])

        stats = {
            "n": len(pos_comparisons),
            "match_rate": pos_matches.mean(),
            "avg_diff_mean": pos_diffs.mean(),
            "avg_diff_p50": np.percentile(pos_diffs, 50),
            "avg_diff_p95": np.percentile(pos_diffs, 95),
            "avg_diff_max": pos_diffs.max(),
        }
        position_stats[pos] = stats

        pos_label = "prefill" if pos == 0 else f"decode{pos}"
        print(f"{pos_label:>4} {stats['n']:>6} {100*stats['match_rate']:>7.1f}% "
              f"{stats['avg_diff_mean']:>10.4f} {stats['avg_diff_p50']:>8.4f} "
              f"{stats['avg_diff_p95']:>8.4f} {stats['avg_diff_max']:>8.4f}")

    # Overall distribution
    print(f"\n--- Overall Avg Logprob Diff Distribution ---")
    print(f"  Mean:   {all_avg_diffs.mean():.6f}")
    print(f"  Std:    {all_avg_diffs.std():.6f}")
    print(f"  p10:    {np.percentile(all_avg_diffs, 10):.6f}")
    print(f"  p50:    {np.percentile(all_avg_diffs, 50):.6f}")
    print(f"  p90:    {np.percentile(all_avg_diffs, 90):.6f}")
    print(f"  p95:    {np.percentile(all_avg_diffs, 95):.6f}")
    print(f"  p99:    {np.percentile(all_avg_diffs, 99):.6f}")
    print(f"  Max:    {all_avg_diffs.max():.6f}")

    # Outliers
    outlier_threshold = 1.0
    outliers = [c for c in comparisons if c.avg_logprob_diff > outlier_threshold]
    if outliers:
        print(f"\n--- Outliers (avg diff > {outlier_threshold}) ---")
        print(f"  Count: {len(outliers)} ({100*len(outliers)/n_total:.1f}%)")
        # Show by position
        outlier_positions = {}
        for o in outliers:
            outlier_positions.setdefault(o.position, []).append(o)
        for pos, pos_outliers in sorted(outlier_positions.items()):
            pos_label = "prefill" if pos == 0 else f"decode{pos}"
            print(f"  Position {pos_label}: {len(pos_outliers)} outliers")

    return {
        "n_total": n_total,
        "n_prompts": n_prompts,
        "n_positions": n_positions,
        "match_rate": all_matches.mean(),
        "avg_diff_mean": all_avg_diffs.mean(),
        "avg_diff_p50": np.percentile(all_avg_diffs, 50),
        "avg_diff_p95": np.percentile(all_avg_diffs, 95),
        "avg_diff_max": all_avg_diffs.max(),
        "n_outliers": len(outliers),
        "position_stats": position_stats,
    }


def cmd_stats(args):
    """Run statistical comparison with many prompts."""
    from transformers import AutoTokenizer

    setup_transformers()

    for model_path in args.model_paths:
        print(f"\n{'#'*70}")
        print(f"# Statistical Comparison: {Path(model_path).name}")
        print(f"# Prompts: {args.num_prompts}, prompt_length: {args.prompt_length}, decode_length: {args.decode_length}")
        print(f"# Mode: {'no-compile' if args.no_compile else 'compiled'}, TF kernels: {args.tf_kernels}")
        print(f"{'#'*70}")

        revision = getattr(args, 'revision', None)

        # Set Transformers kernel configuration
        set_transformers_kernels(model_path, args.tf_kernels)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, trust_remote_code=True)

        # Load and tokenize prompts from dataset
        print(f"\nLoading {args.num_prompts} prompts from C4 (exactly {args.prompt_length} tokens each)...")
        token_ids_list = load_and_tokenize_prompts(
            args.num_prompts,
            args.prompt_length,
            tokenizer,
            seed=args.seed,
        )
        print(f"  Prepared {len(token_ids_list)} token sequences")

        # Run vLLM inference
        vllm_tokens, vllm_logprobs = run_vllm_inference(
            model_path, token_ids_list, args.decode_length,
            args.dtype, args.no_compile, revision
        )

        # Run Transformers inference
        tf_tokens, tf_logprobs = run_transformers_inference(
            model_path, token_ids_list, args.decode_length,
            args.dtype, revision
        )

        # Compute comparisons
        comparisons = compute_comparisons(vllm_tokens, vllm_logprobs, tf_tokens, tf_logprobs)

        # Print statistics
        mode_label = "no-compile" if args.no_compile else "compiled"
        stats = print_stats_report(
            comparisons,
            f"Results ({mode_label}, TF={args.tf_kernels})"
        )

        print(f"\n{'='*70}")
        print(f" SUMMARY: {Path(model_path).name}")
        print(f"{'='*70}")
        print(f"  Mode: {mode_label}")
        print(f"  TF kernels: {args.tf_kernels}")
        print(f"  Token match rate: {100*stats['match_rate']:.1f}%")
        print(f"  Avg diff (mean): {stats['avg_diff_mean']:.4f}")
        print(f"  Avg diff (p95):  {stats['avg_diff_p95']:.4f}")
        print(f"  Avg diff (max):  {stats['avg_diff_max']:.4f}")
        if stats['n_outliers'] > 0:
            print(f"  WARNING: {stats['n_outliers']} outliers detected (avg diff > 1.0)")
        print()


def cmd_logits(args):
    """Run logits comparison test."""
    revision = getattr(args, 'revision', None)
    debug_gdn = getattr(args, 'debug_gdn', False)
    for model_path in args.model_paths:
        compare_logits(model_path, args.prompt, args.max_tokens, args.dtype, args.no_compile, revision, debug_gdn)


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
    p_logits.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    p_logits.add_argument("--revision", default=None, help="Model revision")
    p_logits.add_argument("--debug-gdn", action="store_true", help="Enable GDN debug output")
    p_logits.set_defaults(func=cmd_logits)

    # Comprehensive comparison
    p_compare = subparsers.add_parser("compare", help="Compare across prompt sizes, decode lengths, and batch sizes")
    p_compare.add_argument("model_paths", nargs="+", help="Path(s) to model checkpoint(s)")
    p_compare.add_argument("--prompt-sizes", default="5,50,200", help="Comma-separated prompt sizes in tokens")
    p_compare.add_argument("--decode-lengths", default="1,5,10", help="Comma-separated decode lengths")
    p_compare.add_argument("--batch-sizes", default="1,2,4", help="Comma-separated batch sizes")
    p_compare.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16", help="Data type")
    p_compare.add_argument("--no-compile", action="store_true", help="Disable torch.compile (default: compile enabled)")
    p_compare.add_argument("--revision", default=None, help="Model revision")
    p_compare.set_defaults(func=cmd_compare)

    # Statistical comparison
    p_stats = subparsers.add_parser("stats", help="Statistical comparison with many prompts (per-position analysis)")
    p_stats.add_argument("model_paths", nargs="+", help="Path(s) to model checkpoint(s)")
    p_stats.add_argument("--num-prompts", type=int, default=128, help="Number of prompts to test")
    p_stats.add_argument("--prompt-length", type=int, default=256, help="Number of tokens to prefill")
    p_stats.add_argument("--decode-length", type=int, default=10, help="Number of tokens to decode")
    p_stats.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16", help="Data type")
    p_stats.add_argument("--no-compile", action="store_true", help="Disable torch.compile (default: compile enabled)")
    p_stats.add_argument("--tf-kernels", choices=["upstream", "vllm"], default="upstream",
                        help="Transformers kernel config: upstream FLA or vLLM forks")
    p_stats.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p_stats.add_argument("--revision", default=None, help="Model revision")
    p_stats.set_defaults(func=cmd_stats)

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
