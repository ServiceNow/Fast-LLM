#!/usr/bin/env python3
"""
Test that Fast-LLM and HuggingFace GPT-OSS models produce equivalent outputs.

Based on test_huggingface_model from tests/models/test_checkpoint.py
"""

import pathlib
import sys
import tempfile

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


def test_gpt_oss_20b_forward_equivalence():
    """Test that HuggingFace and Fast-LLM produce equivalent outputs for GPT-OSS 20B."""
    print("=" * 80)
    print("Testing GPT-OSS 20B Forward Pass Equivalence")
    print("=" * 80)

    model_path = "openai/gpt-oss-20b"

    try:
        # Create temporary directory for conversion
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            hf_local_path = tmpdir / "hf_model"
            fast_llm_path = tmpdir / "fast_llm"

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

            print(f"\n3. Saving dequantized model...")
            dequantized_path = tmpdir / "dequantized_hf"
            hf_model.save_pretrained(dequantized_path)

            # Save vocab size and config before freeing the model
            vocab_size = hf_model.config.vocab_size

            # Free HuggingFace model to save memory
            del hf_model
            torch.cuda.empty_cache()

            print(f"\n4. Converting to Fast-LLM format...")
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

            print(f"\n5. Creating test input...")
            test_input = torch.randint(
                0,
                vocab_size,
                size=(2, 32),  # Small batch and sequence length
                dtype=torch.int64,
                device="cuda",
            )
            print(f"   Input shape: {test_input.shape}")
            print(f"   Vocab size: {vocab_size}")

            print(f"\n6. Loading HuggingFace model and running forward pass...")
            # Reload HuggingFace model just for inference
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                dequantized_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).cuda()

            print(f"   Running HuggingFace model...")
            with torch.no_grad():
                hf_output = hf_model(test_input)

            # Save the output and free the model
            hf_logits = hf_output.logits.clone()
            del hf_model, hf_output
            torch.cuda.empty_cache()

            print(f"\n7. Loading Fast-LLM model and running forward pass...")
            # Get the HuggingFace wrapper class from Fast-LLM
            # This wraps Fast-LLM model to match HF interface
            from fast_llm.models.gpt.huggingface import GPTHuggingfaceModel

            fast_llm_model = GPTHuggingfaceModel.from_pretrained(
                CheckpointLoadConfig(
                    path=fast_llm_path,
                    format=FastLLMCheckpointFormat,
                    load_config=ModelConfigType.model,
                )
            )

            # Run Fast-LLM model
            print(f"   Running Fast-LLM model...")
            with torch.no_grad():
                fast_llm_output = fast_llm_model(test_input)

            # Save the output
            fast_llm_logits = fast_llm_output.logits.clone()

            print(f"\n8. Comparing outputs...")
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
