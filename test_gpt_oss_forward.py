#!/usr/bin/env python3
"""
Test that Fast-LLM and HuggingFace GPT-OSS models produce equivalent outputs.

Based on test_huggingface_model from tests/models/test_checkpoint.py
"""

import os
import pathlib
import sys
import tempfile

# Set PyTorch memory allocator to use expandable segments to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import transformers

from fast_llm.engine.checkpoint.config import (
    CheckpointLoadConfig,
    CheckpointSaveConfig,
    FastLLMCheckpointFormat,
    ModelConfigType,
)
from fast_llm.engine.checkpoint.convert import ConvertConfig
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.models.gpt.conversion.config import GptOssCheckpointFormat
from tests.utils.compare_tensor_logs import CompareConfig

sys.path.insert(0, "/home/ubuntu/Fast-LLM")

# Number of transformer layers to keep for testing (reduce to save memory)
NUM_LAYERS_TO_KEEP = 2


def test_gpt_oss_20b_forward_equivalence():
    """Test that HuggingFace and Fast-LLM produce equivalent outputs for GPT-OSS 20B."""
    print("=" * 80)
    print(f"Testing GPT-OSS 20B Forward Pass Equivalence ({NUM_LAYERS_TO_KEEP}-layer model)")
    print("=" * 80)

    model_path = "openai/gpt-oss-20b"

    try:
        # Create persistent directory for conversion
        with tempfile.TemporaryDirectory(dir="/home/ubuntu/Fast-LLM", prefix="test_gpt_oss_checkpoints_", delete=False) as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            hf_local_path = tmpdir / "hf_model"
            fast_llm_path = tmpdir / "fast_llm"

            print(f"\nUsing checkpoint directory: {tmpdir}")

            print(f"\n1. Downloading HuggingFace model files...")
            print(f"   Source: {model_path}")

            # Download the model files from HF Hub without instantiating
            from huggingface_hub import snapshot_download

            hf_local_path = snapshot_download(repo_id=model_path, local_dir_use_symlinks=False)
            hf_local_path = pathlib.Path(hf_local_path)

            print(f"   Downloaded to: {hf_local_path}")

            print(f"\n2. Loading HuggingFace model with dequantization...")
            # Load the model with dequantization enabled
            # This converts the MXFP4 quantized weights (blocks/scales) to standard float weights
            from transformers import Mxfp4Config

            quantization_config = Mxfp4Config(dequantize=True)

            hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                hf_local_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
            ).cuda()

            print(f"\n3. Trimming model to first {NUM_LAYERS_TO_KEEP} layers...")
            # Keep only first N transformer blocks to reduce memory usage
            # This is sufficient to validate the conversion works correctly
            # GptOssForCausalLM has structure: model.layers (not transformer.h)
            original_num_layers = len(hf_model.model.layers)
            print(f"   Original layers: {original_num_layers}, keeping: {NUM_LAYERS_TO_KEEP}")
            hf_model.model.layers = hf_model.model.layers[:NUM_LAYERS_TO_KEEP]
            hf_model.config.num_hidden_layers = NUM_LAYERS_TO_KEEP

            # GPT-OSS also has a layer_types config that must match num_hidden_layers
            if hasattr(hf_model.config, 'layer_types'):
                print(f"   Original layer_types length: {len(hf_model.config.layer_types)}")
                hf_model.config.layer_types = hf_model.config.layer_types[:NUM_LAYERS_TO_KEEP]
                print(f"   Trimmed layer_types: {hf_model.config.layer_types}")

            print(f"\n4. Saving trimmed dequantized model...")
            dequantized_path = tmpdir / "dequantized_hf"
            hf_model.save_pretrained(dequantized_path)

            # Save vocab size and config before freeing the model
            vocab_size = hf_model.config.vocab_size

            # Free HuggingFace model to save memory
            del hf_model
            torch.cuda.empty_cache()

            print(f"\n5. Converting to Fast-LLM format...")
            print(f"   Source: {dequantized_path}")
            print(f"   Target: {fast_llm_path}")

            # Convert dequantized HF model to Fast-LLM format
            ConvertConfig(
                input=CheckpointLoadConfig(
                    path=dequantized_path,
                    format=GptOssCheckpointFormat,
                    load_config=ModelConfigType.model,
                ),
                output=CheckpointSaveConfig(
                    path=fast_llm_path,
                    format=FastLLMCheckpointFormat,
                ),
                model=GPTModelConfig,
            ).run()

            print(f"\n6. Creating test input...")
            # Set seed for reproducibility
            torch.manual_seed(42)
            test_input = torch.randint(
                0,
                vocab_size,
                size=(2, 32),  # Small batch and sequence length
                dtype=torch.int64,
                device="cuda",
            )
            print(f"   Input shape: {test_input.shape}")
            print(f"   Vocab size: {vocab_size}")
            print(f"   First 10 token IDs: {test_input[0, :10].tolist()}")

            print(f"\n7. Loading HuggingFace model and running forward pass...")
            # Reload HuggingFace model just for inference
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                dequantized_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).cuda()

            # Add forward hooks for debugging
            hf_activations = {}

            def make_hf_hook(name):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        output_tensor = output[0]
                    else:
                        output_tensor = output
                    hf_activations[name] = output_tensor.detach()
                    print(f"   HF {name}: shape={output_tensor.shape}, mean={output_tensor.mean().item():.6f}, std={output_tensor.std().item():.6f}")
                return hook

            hf_model.model.embed_tokens.register_forward_hook(make_hf_hook("embeddings"))
            hf_model.model.layers[0].self_attn.register_forward_hook(make_hf_hook("layer0_attn"))
            hf_model.model.layers[0].mlp.register_forward_hook(make_hf_hook("layer0_mlp"))
            hf_model.model.layers[0].register_forward_hook(make_hf_hook("layer0_output"))
            if len(hf_model.model.layers) > 1:
                hf_model.model.layers[1].register_forward_hook(make_hf_hook("layer1_output"))
            hf_model.model.norm.register_forward_hook(make_hf_hook("final_norm"))
            hf_model.lm_head.register_forward_hook(make_hf_hook("lm_head"))

            print(f"   Running HuggingFace model...")
            with torch.no_grad():
                hf_output = hf_model(test_input)

            # Save the output and free the model
            hf_logits = hf_output.logits.clone().cpu()
            del hf_model, hf_output
            torch.cuda.empty_cache()

            # Additional memory cleanup to ensure PyTorch releases reserved memory
            import gc
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Try to force PyTorch to release all cached memory
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

            # Multiple rounds of cleanup
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()

            print(f"   GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
                  f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

            print(f"\n8. Loading Fast-LLM model and running forward pass...")
            # Get the HuggingFace wrapper class from Fast-LLM
            # This wraps Fast-LLM model to match HF interface
            from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
            from fast_llm.logging import set_model_debug_level
            from fast_llm.engine.config_utils.logging import TensorLogs, TensorLogsConfig

            # Initialize TensorLogs and enable debug mode
            TensorLogs.reset(TensorLogsConfig(save=False, show=True, max_elements=8))
            set_model_debug_level(3)  # Level 3 shows tensor samples

            # Verify debug level is set
            from fast_llm.logging import get_model_debug_level
            print(f"   Debug level set to: {get_model_debug_level()}")

            # Enable Fast-LLM logging
            import logging
            logging.basicConfig(level=logging.INFO, format='%(message)s')
            logging.getLogger('fast_llm').setLevel(logging.INFO)

            fast_llm_model = HuggingfaceGPTModelForCausalLM.from_pretrained(
                CheckpointLoadConfig(
                    path=fast_llm_path,
                    format=FastLLMCheckpointFormat,
                    load_config=ModelConfigType.model,
                )
            )

            # Fast-LLM uses internal debug logging - we'll parse that output
            # to extract intermediate activations
            print(f"   Running Fast-LLM model (internal debug logging enabled)...")

            # Capture the debug output
            import io
            import contextlib

            # Redirect stdout to capture Fast-LLM's debug output
            fast_llm_debug_output = io.StringIO()
            with contextlib.redirect_stdout(fast_llm_debug_output):
                with torch.no_grad():
                    fast_llm_output = fast_llm_model(test_input)

            # Parse the debug output to extract key activations
            debug_lines = fast_llm_debug_output.getvalue().split('\n')
            fast_llm_activations = {}

            for line in debug_lines:
                # Look for lines like: "Global : decoder.0 mixer output    shape=(2, 32, 2880)    ...    mu=0.0122    std=0.8114"
                if 'Global :' in line and 'mu=' in line and 'std=' in line:
                    parts = line.split()
                    # Extract the layer name (e.g., "decoder.0 mixer output")
                    try:
                        name_start = line.index('Global :') + len('Global : ')
                        name_end = line.index('shape=')
                        name = line[name_start:name_end].strip()

                        # Extract mean and std
                        mu_idx = parts.index('mu=')
                        std_idx = parts.index('std=')
                        mu = float(parts[mu_idx].split('=')[1])
                        std = float(parts[std_idx].split('=')[1])

                        fast_llm_activations[name] = {'mean': mu, 'std': std}
                        print(f"   Fast-LLM {name}: mean={mu:.6f}, std={std:.6f}")
                    except (ValueError, IndexError):
                        pass

            # Save the output
            fast_llm_logits = fast_llm_output.logits.clone()

            print(f"\n9. Comparing intermediate activations...")
            # Map HF layer names to Fast-LLM layer names
            layer_mapping = {
                'embeddings': None,  # Not captured in Fast-LLM debug
                'layer0_attn': 'decoder.0 mixer output',
                'layer0_mlp': 'decoder.0 MLP output',
                'layer0_output': 'decoder.0 MLP residual',  # After residual connection
                'layer1_output': 'decoder.1 MLP residual',
                'final_norm': None,  # Check if captured
                'lm_head': 'head Language model logits',
            }

            # Compare the activations where we have both
            for hf_name, fast_llm_name in layer_mapping.items():
                if hf_name in hf_activations and fast_llm_name and fast_llm_name in fast_llm_activations:
                    hf_act = hf_activations[hf_name]
                    fl_stats = fast_llm_activations[fast_llm_name]
                    hf_mean = hf_act.mean().item()
                    hf_std = hf_act.std().item()
                    fl_mean = fl_stats['mean']
                    fl_std = fl_stats['std']

                    mean_diff = abs(hf_mean - fl_mean)
                    std_diff = abs(hf_std - fl_std)

                    status = "✓" if (mean_diff < 0.01 and std_diff < 0.1) else "✗"
                    print(f"   {status} {hf_name:15s}: HF(μ={hf_mean:.4f}, σ={hf_std:.4f}) vs FL(μ={fl_mean:.4f}, σ={fl_std:.4f}) | Δμ={mean_diff:.4f}, Δσ={std_diff:.4f}")

            print(f"\n10. Comparing final outputs...")
            # Move HF logits back to GPU for comparison
            hf_logits = hf_logits.cuda()

            print(f"   HF output shape: {hf_logits.shape}")
            print(f"   Fast-LLM output shape: {fast_llm_logits.shape}")
            print(f"   HF output dtype: {hf_logits.dtype}")
            print(f"   Fast-LLM output dtype: {fast_llm_logits.dtype}")

            # Compare using Fast-LLM's comparison utility
            errors = []
            CompareConfig().compare_tensors(
                {"samples": hf_logits, "shape": hf_logits.shape, "step": 0},
                {"samples": fast_llm_logits, "shape": fast_llm_logits.shape, "step": 0},
                errors,
                "HuggingFace vs Fast-LLM",
                "logits",
            )

            if errors:
                print(f"\n❌ Comparison failed:")
                for error in errors:
                    print(f"   {error}")
                return False

            # Print statistics
            print(f"\n   Statistics:")
            print(f"   HF logits mean: {hf_logits.mean().item():.4f}")
            print(f"   Fast-LLM logits mean: {fast_llm_logits.mean().item():.4f}")
            print(
                f"   Absolute difference mean: {(hf_logits - fast_llm_logits).abs().mean().item():.6f}"
            )
            print(f"   Max absolute difference: {(hf_logits - fast_llm_logits).abs().max().item():.6f}")

            print(f"\n✅ Forward pass equivalence test passed!")
            return True

    except Exception as e:
        print(f"\n❌ Test failed:")
        print(f"   Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gpt_oss_20b_forward_equivalence()
    sys.exit(0 if success else 1)
