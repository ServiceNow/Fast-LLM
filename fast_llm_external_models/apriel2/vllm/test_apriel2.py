#!/usr/bin/env python3
"""Test script for Apriel2 vLLM implementation.

This script tests coherence and numerical correctness of Apriel2 models
by comparing vLLM outputs with the reference Transformers implementation.

Usage:
    # Test coherence (generation quality)
    python test_apriel2.py coherence /path/to/model
    python test_apriel2.py coherence /path/to/model --placement every2nd-gdn

    # Compare logits between vLLM and Transformers
    python test_apriel2.py logits /path/to/model
    python test_apriel2.py logits /path/to/model --placement all-gdn

    # Statistical comparison with many prompts (for rigorous testing)
    python test_apriel2.py stats /path/to/model --num-prompts 128
    python test_apriel2.py stats /path/to/model --placement every3rd-gdn

    # Multi-placement sweep (loads model once, tests multiple placements)
    python test_apriel2.py stats /path/to/model --placements all-attention all-gdn every2nd-gdn
    python test_apriel2.py stats /path/to/model --placement-sweep  # Test common placements

    # Compare standalone model vs supernet with matching placement
    # (Diagnose if placement switching is working correctly)
    python test_apriel2.py compare-placement \
        --standalone /tmp/apriel2-0.5b-every2nd-gdn \
        --supernet /tmp/apriel2-0.5b-dev \
        --placement every2nd-gdn \
        --framework both

    # Run both tests
    python test_apriel2.py all /path/to/model

Placement patterns:
    all-X                   All layers use mixer X (attention, gdn, kda, swa)
    everyNth-X              Every Nth layer uses mixer X, others use attention
                            (e.g., every2nd-gdn, every3rd-kda, every4th-swa)
    bookend-attn-X          First 2 and last 2 layers use attention, middle uses X
                            (e.g., bookend-attn-gdn, bookend-attn-kda)
    attn,gdn,attn,...       Explicit comma-separated list

Examples:
    --placement all-attention       All layers use attention
    --placement all-gdn             All layers use GDN
    --placement every2nd-gdn        Every 2nd layer is GDN (1=attn, 2=gdn, 3=attn, ...)
    --placement every4th-kda        Every 4th layer is KDA
    --placement bookend-attn-gdn    Attention at ends, GDN in middle
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
import triton

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.config.compilation import CompilationMode

# Apriel2 model registration is handled automatically via vLLM's plugin system
# (see fast-llm setup.cfg entry_points for vllm.general_plugins)


# Set a triton allocator to avoid "no allocator was set" errors
def _triton_allocator(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device='cuda').data_ptr()


triton.set_allocator(_triton_allocator)


def normalize_mixer_name(name: str) -> str:
    """Normalize mixer name aliases to canonical names."""
    # Map shorthand names to canonical names
    aliases = {
        "swa": "sliding_window",
        "attn": "attention",
    }
    return aliases.get(name, name)


def parse_placement(placement_str: str, num_layers: int) -> list[str]:
    """Parse placement string into a list of mixer names.

    Args:
        placement_str: Either a pattern name or comma-separated mixer names.
            Patterns:
                all-X: All layers use mixer X (attention, gdn, kda, swa/sliding_window)
                everyNth-X: Every Nth layer uses mixer X, others use attention
                bookend-attn-X: First 2 and last 2 layers use attention, middle uses X
            Explicit: attention,gdn,attention,gdn,...
        num_layers: Number of layers in the model.

    Returns:
        List of mixer names, one per layer.
    """
    placement_str = placement_str.strip().lower()

    # Handle all-X patterns (e.g., all-attention, all-gdn, all-kda, all-swa)
    if placement_str.startswith("all-"):
        mixer = normalize_mixer_name(placement_str[4:])  # Extract mixer name after "all-"
        return [mixer] * num_layers

    # Handle bookend-attn-X patterns (e.g., bookend-attn-gdn, bookend-attn-kda)
    if placement_str.startswith("bookend-attn-"):
        mixer = normalize_mixer_name(placement_str[13:])  # Extract mixer name after "bookend-attn-"
        placement = []
        for i in range(num_layers):
            if i < 2 or i >= num_layers - 2:
                placement.append("attention")
            else:
                placement.append(mixer)
        return placement

    # Handle everyNth-X patterns (e.g., every2nd-gdn, every3rd-kda, every4th-swa)
    if placement_str.startswith("every"):
        # Find where the mixer name starts (after the dash)
        dash_idx = placement_str.rfind("-")
        if dash_idx > 5:  # "every" is 5 chars, need at least that plus ordinal
            n_str = placement_str[5:dash_idx]  # Extract "2nd", "3rd", etc.
            n_str = n_str.rstrip("ndrdth")  # Remove ordinal suffix
            mixer = normalize_mixer_name(placement_str[dash_idx + 1:])  # Extract mixer name
            try:
                n = int(n_str)
                placement = []
                for i in range(num_layers):
                    if (i + 1) % n == 0:  # Every nth layer uses the specified mixer
                        placement.append(mixer)
                    else:
                        placement.append("attention")
                return placement
            except ValueError:
                pass  # Fall through to explicit list or error

    # Explicit comma-separated list
    if "," in placement_str:
        placement = [normalize_mixer_name(m.strip()) for m in placement_str.split(",")]
        if len(placement) != num_layers:
            raise ValueError(f"Placement has {len(placement)} entries but model has {num_layers} layers")
        return placement

    raise ValueError(f"Unknown placement pattern: {placement_str}")


def apply_placement(llm: "LLM", placement_str: str | None) -> None:
    """Apply placement to a vLLM model if specified.

    Args:
        llm: vLLM LLM instance.
        placement_str: Placement string or None to skip.
    """
    if placement_str is None:
        return

    # Get current placements to determine num_layers
    placements = llm.collective_rpc("get_layer_placements")
    if not placements or not placements[0]:
        print(f"  Model does not support placement switching, ignoring --placement")
        return

    num_layers = len(placements[0])
    current = list(placements[0].values())
    print(f"  Current placement: {current[0]} (all {num_layers} layers)")

    new_placement = parse_placement(placement_str, num_layers)
    llm.collective_rpc("set_layer_placements", args=(new_placement,))

    # Verify
    placements_after = llm.collective_rpc("get_layer_placements")
    counts = {}
    for v in placements_after[0].values():
        counts[v] = counts.get(v, 0) + 1
    counts_str = ", ".join(f"{c} {m}" for m, c in sorted(counts.items()))
    print(f"  Applied placement '{placement_str}': {counts_str}")


def apply_placement_transformers(model, placement: list[str], verbose: bool = True) -> None:
    """Apply placement to Transformers model by setting main_mixer_name for stochastic layers.

    Args:
        model: Transformers model instance.
        placement: List of mixer names, one per layer.
        verbose: Whether to print status messages.
    """
    # Apriel2 uses model.model.decoder.blocks instead of model.model.layers
    blocks = model.model.decoder.blocks
    num_layers = len(blocks)
    if len(placement) != num_layers:
        raise ValueError(f"Placement has {len(placement)} entries but model has {num_layers} layers")

    applied = 0
    for layer_idx, mixer_name in enumerate(placement):
        block = blocks[layer_idx]
        # Check if block has a stochastic mixer with main_mixer_name
        if hasattr(block, 'mixer') and hasattr(block.mixer, 'main_mixer_name'):
            block.mixer.main_mixer_name = mixer_name
            applied += 1

    if verbose:
        counts = {}
        for m in placement:
            counts[m] = counts.get(m, 0) + 1
        counts_str = ", ".join(f"{c} {m}" for m, c in sorted(counts.items()))
        print(f"  Applied placement to Transformers: {counts_str} (applied to {applied} layers)")


def setup_transformers():
    """Register Apriel2 model with Transformers."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2TextConfig
    from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForCausalLM

    AutoConfig.register("apriel2_text", Apriel2TextConfig)
    AutoModelForCausalLM.register(Apriel2TextConfig, Apriel2ForCausalLM)


def test_coherence_vllm(model_paths: list[str], prompts: list[str], max_tokens: int = 50, placement: str | None = None):
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

        apply_placement(llm, placement)

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


def compare_logits(model_path: str, prompt: str, max_tokens: int = 1, dtype: str = "bfloat16", no_compile: bool = False, revision: str | None = None, debug_gdn: bool = False, placement: str | None = None):
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
    print(f"Placement: {placement}")
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

    apply_placement(llm, placement)

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
    no_compile: bool = False,
    revision: str | None = None,
    placement: str | None = None,
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

    apply_placement(llm, placement)

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
            placement=getattr(args, 'placement', None),
        )


def cmd_coherence(args):
    """Run coherence test."""
    prompts = [
        "The capital of France is",
        "To solve this math problem, I need to",
        "Once upon a time, there was a",
    ]

    placement = getattr(args, 'placement', None)

    print("\n" + "="*70)
    print("COHERENCE TEST: vLLM")
    print("="*70)
    vllm_results = test_coherence_vllm(args.model_paths, prompts, args.max_tokens, placement=placement)

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


# Common placement patterns for sweep testing
SWEEP_PLACEMENTS = [
    "all-attention",
    "all-gdn",
    "all-kda",
    "bookend-attn-gdn",   # 2x attn, middle layers gdn, 2x attn
    "bookend-attn-kda",   # 2x attn, middle layers kda, 2x attn
    "every2nd-gdn",
    "every2nd-kda",
    "every2nd-swa",       # sliding window attention
    "every4th-gdn",
    "every4th-kda",
    "every4th-swa",
]


def resolve_placements(args) -> list[str] | None:
    """Resolve placement arguments into a list of placement patterns.

    Handles --placement, --placements, and --placement-sweep arguments.
    Returns None if no placement is specified (use model default).

    Args:
        args: Parsed command-line arguments.

    Returns:
        List of placement pattern strings, or None if no placement specified.
    """
    placements = []

    # Check --placement-sweep first (highest priority)
    if getattr(args, 'placement_sweep', False):
        placements.extend(SWEEP_PLACEMENTS)

    # Then check --placements (list)
    if getattr(args, 'placements', None):
        placements.extend(args.placements)

    # Finally check --placement (single, deprecated)
    if getattr(args, 'placement', None):
        if placements:
            print("  Warning: --placement is deprecated when using --placements or --placement-sweep")
        else:
            placements.append(args.placement)

    return placements if placements else None


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


def load_vllm_model(
    model_path: str,
    batch_size: int,
    dtype: str,
    no_compile: bool,
    revision: str | None,
) -> "LLM":
    """Load vLLM model.

    Args:
        model_path: Path to model checkpoint.
        batch_size: Max concurrent sequences.
        dtype: Data type (bfloat16 or float32).
        no_compile: Whether to disable torch.compile.
        revision: Model revision.

    Returns:
        Loaded LLM instance.
    """
    compile_label = "no-compile" if no_compile else "compiled"
    print(f"\nLoading vLLM model ({compile_label}, batch_size={batch_size})...")
    compilation_config = CompilationConfig(mode=CompilationMode.NONE) if no_compile else None

    llm = LLM(
        model=model_path,
        revision=revision,
        trust_remote_code=True,
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        dtype=dtype,
        compilation_config=compilation_config,
        max_num_seqs=batch_size,  # Control max concurrent sequences
        enable_prefix_caching=False,  # Disable for hybrid models
    )

    return llm


def run_vllm_inference_with_model(
    llm: "LLM",
    token_ids_list: list[list[int]],
    decode_length: int,
    placement: list[str] | None = None,
    verbose: bool = True,
) -> tuple[list[list[int]], list[list[dict]]]:
    """Run vLLM inference with an already-loaded model.

    Args:
        llm: Loaded LLM instance.
        token_ids_list: List of token ID sequences for prompts.
        decode_length: Number of tokens to decode.
        placement: Placement list (already parsed), or None to skip.
        verbose: Whether to print progress.

    Returns:
        - generated_tokens: list of list of token IDs (one list per prompt)
        - logprobs_per_position: list of list of logprob dicts (one list per prompt)
    """
    from vllm import TokensPrompt

    # Apply placement if specified
    if placement is not None:
        llm.collective_rpc("set_layer_placements", args=(placement,))

    # Create TokensPrompt for each prompt
    vllm_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in token_ids_list]

    sampling_params = SamplingParams(
        max_tokens=decode_length,
        temperature=0,
        logprobs=20,
    )

    if verbose:
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

    return generated_tokens, logprobs_per_position


def run_vllm_inference(
    model_path: str,
    token_ids_list: list[list[int]],
    decode_length: int,
    batch_size: int,
    dtype: str,
    no_compile: bool,
    revision: str | None,
    placement: str | None = None,
) -> tuple[list[list[int]], list[list[dict]]]:
    """Run vLLM inference and return generated tokens and logprobs.

    This is a convenience wrapper that loads the model, runs inference, and cleans up.

    Returns:
        - generated_tokens: list of list of token IDs (one list per prompt)
        - logprobs_per_position: list of list of logprob dicts (one list per prompt)
    """
    llm = load_vllm_model(model_path, batch_size, dtype, no_compile, revision)

    # Parse and apply placement if specified
    placement_list = None
    if placement is not None:
        # Get num_layers from model
        placements = llm.collective_rpc("get_layer_placements")
        if placements and placements[0]:
            num_layers = len(placements[0])
            placement_list = parse_placement(placement, num_layers)
            apply_placement(llm, placement)

    generated_tokens, logprobs_per_position = run_vllm_inference_with_model(
        llm, token_ids_list, decode_length, placement_list
    )

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return generated_tokens, logprobs_per_position


def load_transformers_model(
    model_path: str,
    dtype: str,
    revision: str | None,
):
    """Load Transformers model.

    Args:
        model_path: Path to model checkpoint.
        dtype: Data type (bfloat16 or float32).
        revision: Model revision.

    Returns:
        Loaded model instance.
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

    return model


def run_transformers_inference_with_model(
    model,
    token_ids_list: list[list[int]],
    decode_length: int,
    placement: list[str] | None = None,
    verbose: bool = True,
) -> tuple[list[list[int]], list[list[torch.Tensor]]]:
    """Run Transformers inference with an already-loaded model.

    Args:
        model: Loaded Transformers model.
        token_ids_list: List of token ID sequences for prompts.
        decode_length: Number of tokens to decode.
        placement: Placement list (already parsed), or None to skip.
        verbose: Whether to print progress.

    Returns:
        - generated_tokens: list of list of token IDs (one list per prompt)
        - logprobs_per_position: list of list of logprob tensors (one list per prompt)
    """
    # Apply placement if specified
    if placement is not None:
        apply_placement_transformers(model, placement, verbose=verbose)

    generated_tokens = []
    logprobs_per_position = []

    if verbose:
        print(f"Running Transformers inference on {len(token_ids_list)} prompts...")

    for idx, token_ids in enumerate(token_ids_list):
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

        if verbose and ((idx + 1) % 10 == 0 or (idx + 1) == len(token_ids_list)):
            print(f"  Processed {idx + 1}/{len(token_ids_list)} prompts", end="\r")

    if verbose:
        print()

    return generated_tokens, logprobs_per_position


def run_transformers_inference(
    model_path: str,
    token_ids_list: list[list[int]],
    decode_length: int,
    batch_size: int,
    dtype: str,
    revision: str | None,
    placement: str | None = None,
) -> tuple[list[list[int]], list[list[torch.Tensor]]]:
    """Run Transformers inference and return generated tokens and logprobs.

    This is a convenience wrapper that loads the model, runs inference, and cleans up.

    Args:
        batch_size: Unused (kept for API compatibility).

    Returns:
        - generated_tokens: list of list of token IDs (one list per prompt)
        - logprobs_per_position: list of list of logprob tensors (one list per prompt)
    """
    model = load_transformers_model(model_path, dtype, revision)

    # Parse placement if specified
    placement_list = None
    if placement is not None:
        # Apriel2 uses model.model.decoder.blocks instead of model.model.layers
        num_layers = len(model.model.decoder.blocks)
        placement_list = parse_placement(placement, num_layers)

    generated_tokens, logprobs_per_position = run_transformers_inference_with_model(
        model, token_ids_list, decode_length, placement_list
    )

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


def compute_same_framework_comparisons(
    model1_tokens: list[list[int]],
    model1_logprobs: list[list],  # Either list[dict] (vLLM) or list[torch.Tensor] (TF)
    model2_tokens: list[list[int]],
    model2_logprobs: list[list],  # Same type as model1_logprobs
    is_vllm: bool = False,
) -> list[TokenComparison]:
    """Compute per-position comparisons between two models using the same framework.

    This handles both vLLM-style logprobs (list of dicts) and Transformers-style
    logprobs (list of tensors).

    Args:
        model1_tokens: Generated tokens from model 1.
        model1_logprobs: Logprobs from model 1.
        model2_tokens: Generated tokens from model 2.
        model2_logprobs: Logprobs from model 2.
        is_vllm: If True, logprobs are vLLM-style dicts; if False, Transformers-style tensors.

    Returns:
        List of TokenComparison objects.
    """
    comparisons = []

    for prompt_idx, (t1, lp1, t2, lp2) in enumerate(zip(
        model1_tokens, model1_logprobs, model2_tokens, model2_logprobs
    )):
        for pos in range(min(len(t1), len(t2), len(lp1), len(lp2))):
            token1 = t1[pos]
            token2 = t2[pos]
            logprobs1 = lp1[pos]
            logprobs2 = lp2[pos]

            diffs = []
            if is_vllm:
                # vLLM: logprobs are dicts with token_id -> LogprobInfo
                if logprobs1 and logprobs2:
                    # Compare logprobs for tokens in model1's top-K
                    for tid, lp_info1 in list(logprobs1.items())[:20]:
                        lp1_val = lp_info1.logprob
                        lp_info2 = logprobs2.get(tid)
                        if lp_info2 is not None:
                            lp2_val = lp_info2.logprob
                            diffs.append(abs(lp1_val - lp2_val))
            else:
                # Transformers: logprobs are tensors
                if logprobs1 is not None and logprobs2 is not None:
                    # Get top-K token IDs from model1
                    top_k = logprobs1.topk(20)
                    for i in range(min(20, len(top_k.indices))):
                        tid = top_k.indices[i].item()
                        lp1_val = logprobs1[tid].item()
                        lp2_val = logprobs2[tid].item()
                        diffs.append(abs(lp1_val - lp2_val))

            avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
            max_diff = max(diffs) if diffs else 0.0

            comparisons.append(TokenComparison(
                prompt_idx=prompt_idx,
                position=pos,
                vllm_token_id=token1,  # Model 1 token (reusing field name)
                tf_token_id=token2,    # Model 2 token (reusing field name)
                token_match=(token1 == token2),
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


def print_multi_placement_report(results: dict[str, dict], model_name: str, decode_length: int) -> None:
    """Print comparison table across placements.

    Args:
        results: Dict mapping placement_str -> stats dict from print_stats_report.
        model_name: Model name for header.
        decode_length: Number of decode positions.
    """
    print(f"\n{'='*90}")
    print(f" PLACEMENT COMPARISON: {model_name}")
    print(f"{'='*90}")

    if not results:
        print("No placements to compare.")
        return

    # Header
    header = f"{'Placement':<20} {'Prefill%':>10} {'Decode1%':>10}"
    if decode_length >= 15:
        header += f" {'Decode15%':>10}"
    header += f" {'AvgDiff':>10} {'Outliers':>10}"
    print(header)
    print("-" * 90)

    for placement_str, stats in results.items():
        position_stats = stats.get("position_stats", {})

        # Get prefill match rate (position 0)
        prefill_stats = position_stats.get(0, {})
        prefill_match = prefill_stats.get("match_rate", 0.0) * 100

        # Get decode1 match rate (position 1)
        decode1_stats = position_stats.get(1, {})
        decode1_match = decode1_stats.get("match_rate", 0.0) * 100

        row = f"{placement_str:<20} {prefill_match:>9.1f}% {decode1_match:>9.1f}%"

        # Get decode15 match rate if available
        if decode_length >= 15:
            decode15_stats = position_stats.get(14, {})  # 0-indexed, so position 14 is 15th
            decode15_match = decode15_stats.get("match_rate", 0.0) * 100
            row += f" {decode15_match:>9.1f}%"

        avg_diff = stats.get("avg_diff_mean", 0.0)
        n_outliers = stats.get("n_outliers", 0)

        row += f" {avg_diff:>10.4f} {n_outliers:>10}"
        print(row)

    print()


def cmd_stats(args):
    """Run statistical comparison with many prompts.

    Supports comparing vLLM vs Transformers across multiple placements.
    When multiple placements are specified, models are loaded once and
    placements are switched dynamically.
    """
    from transformers import AutoTokenizer

    setup_transformers()

    # Resolve placements list
    placements = resolve_placements(args)

    for model_path in args.model_paths:
        model_name = Path(model_path).name
        mode_label = "no-compile" if args.no_compile else "compiled"

        print(f"\n{'#'*70}")
        print(f"# Statistical Comparison: {model_name}")
        print(f"# Prompts: {args.num_prompts}, prompt_length: {args.prompt_length}, decode_length: {args.decode_length}")
        print(f"# Mode: {mode_label}, TF kernels: {args.tf_kernels}")
        if placements:
            print(f"# Placements: {len(placements)} ({', '.join(placements[:3])}{'...' if len(placements) > 3 else ''})")
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

        # Single placement mode (backwards compatible)
        if placements is None or len(placements) == 1:
            placement_str = placements[0] if placements else None

            # Run vLLM inference
            vllm_tokens, vllm_logprobs = run_vllm_inference(
                model_path, token_ids_list, args.decode_length,
                args.batch_size, args.dtype, args.no_compile, revision, placement_str
            )

            # Run Transformers inference (with same placement if specified)
            tf_tokens, tf_logprobs = run_transformers_inference(
                model_path, token_ids_list, args.decode_length,
                args.batch_size, args.dtype, revision, placement_str
            )

            # Compute comparisons
            comparisons = compute_comparisons(vllm_tokens, vllm_logprobs, tf_tokens, tf_logprobs)

            # Print statistics
            stats = print_stats_report(
                comparisons,
                f"Results ({mode_label}, TF={args.tf_kernels})"
            )

            print(f"\n{'='*70}")
            print(f" SUMMARY: {model_name}")
            print(f"{'='*70}")
            print(f"  Mode: {mode_label}")
            print(f"  TF kernels: {args.tf_kernels}")
            print(f"  Batch size: {args.batch_size}")
            print(f"  Dtype: {args.dtype}")
            if placement_str:
                print(f"  Placement: {placement_str}")
            if revision:
                print(f"  Revision: {revision}")
            print(f"  Token match rate: {100*stats['match_rate']:.1f}%")
            print(f"  Avg diff (mean): {stats['avg_diff_mean']:.4f}")
            print(f"  Avg diff (p95):  {stats['avg_diff_p95']:.4f}")
            print(f"  Avg diff (max):  {stats['avg_diff_max']:.4f}")
            if stats['n_outliers'] > 0:
                print(f"  WARNING: {stats['n_outliers']} outliers detected (avg diff > 1.0)")
            print()
            continue

        # Multi-placement sweep mode: load models ONCE and iterate placements
        print(f"\n--- Multi-placement sweep mode ({len(placements)} placements) ---")

        # Load vLLM model once
        llm = load_vllm_model(
            model_path, args.batch_size, args.dtype, args.no_compile, revision
        )

        # Get num_layers from vLLM model
        vllm_placements = llm.collective_rpc("get_layer_placements")
        if not vllm_placements or not vllm_placements[0]:
            print(f"  WARNING: Model does not support placement switching, skipping sweep")
            del llm
            gc.collect()
            torch.cuda.empty_cache()
            continue

        num_layers = len(vllm_placements[0])
        print(f"  Model has {num_layers} layers")

        # Load Transformers model once
        tf_model = load_transformers_model(model_path, args.dtype, revision)

        # Run sweep over placements
        all_results: dict[str, dict] = {}

        for placement_idx, placement_str in enumerate(placements):
            print(f"\n--- Placement {placement_idx + 1}/{len(placements)}: {placement_str} ---")

            try:
                placement_list = parse_placement(placement_str, num_layers)
            except ValueError as e:
                print(f"  ERROR: {e}, skipping")
                continue

            # Run vLLM with this placement
            vllm_tokens, vllm_logprobs = run_vllm_inference_with_model(
                llm, token_ids_list, args.decode_length, placement_list, verbose=True
            )

            # Run Transformers with SAME placement
            tf_tokens, tf_logprobs = run_transformers_inference_with_model(
                tf_model, token_ids_list, args.decode_length, placement_list, verbose=True
            )

            # Compute comparisons
            comparisons = compute_comparisons(vllm_tokens, vllm_logprobs, tf_tokens, tf_logprobs)

            # Print statistics for this placement
            stats = print_stats_report(comparisons, f"Results: {placement_str}")
            all_results[placement_str] = stats

        # Print combined comparison table
        print_multi_placement_report(all_results, model_name, args.decode_length)

        # Cleanup
        del llm, tf_model
        gc.collect()
        torch.cuda.empty_cache()


def cmd_compare_placement(args):
    """Compare standalone model vs supernet with matching placement.

    This command diagnoses whether placement switching works correctly by comparing:
    - A standalone model (e.g., /tmp/apriel2-0.5b-every2nd-gdn) that was trained/converted
      with a fixed architecture
    - A supernet model (e.g., /tmp/apriel2-0.5b-dev) with placement dynamically set to
      match the standalone model's architecture

    If the two models produce different outputs, placement switching is broken for
    that framework.
    """
    from transformers import AutoTokenizer

    setup_transformers()

    standalone_path = args.standalone
    supernet_path = args.supernet
    placement_str = args.placement
    framework = args.framework
    num_prompts = args.num_prompts
    prompt_length = args.prompt_length
    decode_length = args.decode_length
    dtype = args.dtype
    no_compile = args.no_compile
    revision = getattr(args, 'revision', None)

    standalone_name = Path(standalone_path).name
    supernet_name = Path(supernet_path).name

    print(f"\n{'#'*70}")
    print(f"# COMPARE-PLACEMENT: Standalone vs Supernet")
    print(f"#")
    print(f"# Standalone: {standalone_name}")
    print(f"# Supernet:   {supernet_name}")
    print(f"# Placement:  {placement_str}")
    print(f"# Framework:  {framework}")
    print(f"# Prompts:    {num_prompts}, length={prompt_length}, decode={decode_length}")
    print(f"{'#'*70}")

    # Load tokenizer from standalone model (should be compatible)
    tokenizer = AutoTokenizer.from_pretrained(standalone_path, revision=revision, trust_remote_code=True)

    # Load and tokenize prompts from dataset
    print(f"\nLoading {num_prompts} prompts from C4 (exactly {prompt_length} tokens each)...")
    token_ids_list = load_and_tokenize_prompts(
        num_prompts,
        prompt_length,
        tokenizer,
        seed=args.seed,
    )
    print(f"  Prepared {len(token_ids_list)} token sequences")

    results = {}

    # =========================================================================
    # Transformers comparison
    # =========================================================================
    if framework in ("transformers", "both"):
        print(f"\n{'='*70}")
        print(f" TRANSFORMERS COMPARISON")
        print(f"{'='*70}")

        # Load standalone model (no placement switching needed)
        print(f"\n--- Loading standalone Transformers model: {standalone_name} ---")
        standalone_tf = load_transformers_model(standalone_path, dtype, revision)

        # Run inference on standalone (no placement - uses its native architecture)
        print(f"  Running inference on standalone model (native architecture)...")
        standalone_tf_tokens, standalone_tf_logprobs = run_transformers_inference_with_model(
            standalone_tf, token_ids_list, decode_length, placement=None, verbose=True
        )

        del standalone_tf
        gc.collect()
        torch.cuda.empty_cache()

        # Load supernet model
        print(f"\n--- Loading supernet Transformers model: {supernet_name} ---")
        supernet_tf = load_transformers_model(supernet_path, dtype, revision)

        # Get num_layers and parse placement
        num_layers = len(supernet_tf.model.decoder.blocks)
        placement_list = parse_placement(placement_str, num_layers)
        print(f"  Supernet has {num_layers} layers")

        # Run inference on supernet with placement
        print(f"  Running inference on supernet model with placement '{placement_str}'...")
        supernet_tf_tokens, supernet_tf_logprobs = run_transformers_inference_with_model(
            supernet_tf, token_ids_list, decode_length, placement=placement_list, verbose=True
        )

        del supernet_tf
        gc.collect()
        torch.cuda.empty_cache()

        # Compare standalone vs supernet (both Transformers)
        print(f"\n--- Comparing Standalone vs Supernet (Transformers) ---")
        comparisons = compute_same_framework_comparisons(
            standalone_tf_tokens, standalone_tf_logprobs,
            supernet_tf_tokens, supernet_tf_logprobs,
            is_vllm=False,
        )
        stats = print_stats_report(comparisons, "Transformers: Standalone vs Supernet")
        results["transformers"] = stats

    # =========================================================================
    # vLLM comparison
    # =========================================================================
    if framework in ("vllm", "both"):
        print(f"\n{'='*70}")
        print(f" VLLM COMPARISON")
        print(f"{'='*70}")

        # Load standalone vLLM model (no placement switching needed)
        print(f"\n--- Loading standalone vLLM model: {standalone_name} ---")
        standalone_llm = load_vllm_model(standalone_path, args.batch_size, dtype, no_compile, revision)

        # Check if standalone supports placement (it shouldn't need it)
        standalone_placements = standalone_llm.collective_rpc("get_layer_placements")
        if standalone_placements and standalone_placements[0]:
            standalone_num_layers = len(standalone_placements[0])
            print(f"  Standalone has {standalone_num_layers} layers (stochastic mixers detected)")
        else:
            print(f"  Standalone model does not have stochastic mixers (expected for standalone)")

        # Run inference on standalone (no placement - uses native architecture)
        print(f"  Running inference on standalone model (native architecture)...")
        standalone_vllm_tokens, standalone_vllm_logprobs = run_vllm_inference_with_model(
            standalone_llm, token_ids_list, decode_length, placement=None, verbose=True
        )

        del standalone_llm
        gc.collect()
        torch.cuda.empty_cache()

        # Load supernet vLLM model
        print(f"\n--- Loading supernet vLLM model: {supernet_name} ---")
        supernet_llm = load_vllm_model(supernet_path, args.batch_size, dtype, no_compile, revision)

        # Get num_layers and parse placement
        supernet_placements = supernet_llm.collective_rpc("get_layer_placements")
        if not supernet_placements or not supernet_placements[0]:
            print(f"  ERROR: Supernet does not support placement switching!")
            del supernet_llm
            gc.collect()
            torch.cuda.empty_cache()
        else:
            num_layers = len(supernet_placements[0])
            placement_list = parse_placement(placement_str, num_layers)
            print(f"  Supernet has {num_layers} layers")

            # Run inference on supernet with placement
            print(f"  Running inference on supernet model with placement '{placement_str}'...")
            supernet_vllm_tokens, supernet_vllm_logprobs = run_vllm_inference_with_model(
                supernet_llm, token_ids_list, decode_length, placement=placement_list, verbose=True
            )

            del supernet_llm
            gc.collect()
            torch.cuda.empty_cache()

            # Compare standalone vs supernet (both vLLM)
            print(f"\n--- Comparing Standalone vs Supernet (vLLM) ---")
            comparisons = compute_same_framework_comparisons(
                standalone_vllm_tokens, standalone_vllm_logprobs,
                supernet_vllm_tokens, supernet_vllm_logprobs,
                is_vllm=True,
            )
            stats = print_stats_report(comparisons, "vLLM: Standalone vs Supernet")
            results["vllm"] = stats

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'#'*70}")
    print(f"# SUMMARY: Compare-Placement Results")
    print(f"#")
    print(f"# Standalone: {standalone_name}")
    print(f"# Supernet:   {supernet_name}")
    print(f"# Placement:  {placement_str}")
    print(f"{'#'*70}")

    for fw, stats in results.items():
        avg_diff = stats.get("avg_diff_mean", 0.0)
        match_rate = stats.get("match_rate", 0.0) * 100
        n_outliers = stats.get("n_outliers", 0)

        # Determine if results are acceptable (avg diff < 1.0 is good)
        status = "OK" if avg_diff < 1.0 else "BROKEN"
        emoji = "✓" if avg_diff < 1.0 else "✗"

        print(f"\n  {fw.upper():>12}: [{status}] {emoji}")
        print(f"               Match rate: {match_rate:.1f}%")
        print(f"               Avg diff:   {avg_diff:.4f}")
        print(f"               Outliers:   {n_outliers}")

        if avg_diff >= 1.0:
            print(f"               >>> PLACEMENT SWITCHING IS BROKEN IN {fw.upper()} <<<")

    print()


def cmd_logits(args):
    """Run logits comparison test."""
    revision = getattr(args, 'revision', None)
    debug_gdn = getattr(args, 'debug_gdn', False)
    placement = getattr(args, 'placement', None)
    for model_path in args.model_paths:
        compare_logits(model_path, args.prompt, args.max_tokens, args.dtype, args.no_compile, revision, debug_gdn, placement)


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

    # Placement help text (used by multiple subcommands)
    placement_help = (
        "Mixer placement: 'all-X' (all-attention, all-gdn, all-kda), "
        "'everyNth-X' (every2nd-gdn, every4th-kda), "
        "'bookend-attn-X' (bookend-attn-gdn), "
        "or comma-separated list like 'attention,gdn,attention,...'"
    )

    # Coherence test
    p_coherence = subparsers.add_parser("coherence", help="Test generation coherence")
    p_coherence.add_argument("model_paths", nargs="+", help="Path(s) to model checkpoint(s)")
    p_coherence.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    p_coherence.add_argument("--placement", default=None, help=placement_help)
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
    p_logits.add_argument("--placement", default=None, help=placement_help)
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
    p_compare.add_argument("--placement", default=None, help=placement_help)
    p_compare.set_defaults(func=cmd_compare)

    # Statistical comparison
    p_stats = subparsers.add_parser("stats", help="Statistical comparison with many prompts (per-position analysis)")
    p_stats.add_argument("model_paths", nargs="+", help="Path(s) to model checkpoint(s)")
    p_stats.add_argument("--num-prompts", type=int, default=64, help="Number of prompts to test")
    p_stats.add_argument("--prompt-length", type=int, default=256, help="Number of tokens to prefill")
    p_stats.add_argument("--decode-length", type=int, default=10, help="Number of tokens to decode")
    p_stats.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    p_stats.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16", help="Data type")
    p_stats.add_argument("--no-compile", action="store_true", help="Disable torch.compile (default: compile enabled)")
    p_stats.add_argument("--tf-kernels", choices=["upstream", "vllm"], default="upstream",
                        help="Transformers kernel config: upstream FLA or vLLM forks")
    p_stats.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p_stats.add_argument("--revision", default=None, help="Model revision")
    # Placement arguments - support single, multiple, and sweep modes
    p_stats.add_argument("--placement", default=None, help=f"(Deprecated) Single placement - use --placements instead. {placement_help}")
    p_stats.add_argument("--placements", nargs="+", default=None,
                        help="Placement patterns to test (e.g., all-attention all-gdn every2nd-gdn)")
    p_stats.add_argument("--placement-sweep", action="store_true",
                        help=f"Test common placements: {', '.join(SWEEP_PLACEMENTS[:4])}...")
    p_stats.set_defaults(func=cmd_stats)

    # Compare-placement: standalone vs supernet comparison
    p_compare_placement = subparsers.add_parser(
        "compare-placement",
        help="Compare standalone model vs supernet with matching placement",
        description=(
            "Diagnose whether placement switching works correctly by comparing a standalone "
            "model (with fixed architecture) against a supernet model with dynamically set "
            "placement. If outputs differ significantly, placement switching is broken."
        ),
    )
    p_compare_placement.add_argument("--standalone", required=True,
                                     help="Path to standalone model (e.g., /tmp/apriel2-0.5b-every2nd-gdn)")
    p_compare_placement.add_argument("--supernet", required=True,
                                     help="Path to supernet model (e.g., /tmp/apriel2-0.5b-dev)")
    p_compare_placement.add_argument("--placement", required=True,
                                     help=f"Placement to apply to supernet. {placement_help}")
    p_compare_placement.add_argument("--framework", choices=["transformers", "vllm", "both"],
                                     default="both", help="Framework(s) to test")
    p_compare_placement.add_argument("--num-prompts", type=int, default=32,
                                     help="Number of prompts to test")
    p_compare_placement.add_argument("--prompt-length", type=int, default=256,
                                     help="Number of tokens to prefill")
    p_compare_placement.add_argument("--decode-length", type=int, default=10,
                                     help="Number of tokens to decode")
    p_compare_placement.add_argument("--batch-size", type=int, default=1,
                                     help="Batch size for vLLM inference")
    p_compare_placement.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                                     help="Data type")
    p_compare_placement.add_argument("--no-compile", action="store_true",
                                     help="Disable torch.compile for vLLM")
    p_compare_placement.add_argument("--seed", type=int, default=42,
                                     help="Random seed for reproducibility")
    p_compare_placement.add_argument("--revision", default=None, help="Model revision")
    p_compare_placement.set_defaults(func=cmd_compare_placement)

    # All tests
    p_all = subparsers.add_parser("all", help="Run all tests")
    p_all.add_argument("model_paths", nargs="+", help="Path(s) to model checkpoint(s)")
    p_all.add_argument("--prompt", default="The capital of France is", help="Input prompt for logits test")
    p_all.add_argument("--max-tokens", type=int, default=50, help="Max tokens for coherence test")
    p_all.add_argument("--placement", default=None, help=placement_help)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
