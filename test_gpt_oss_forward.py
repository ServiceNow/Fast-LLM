#!/usr/bin/env python3
"""
Test that Fast-LLM and HuggingFace GPT-OSS models produce equivalent outputs.

Based on test_huggingface_model from tests/models/test_checkpoint.py
"""

import sys
import pathlib
import tempfile

sys.path.insert(0, '/home/ubuntu/Fast-LLM')

import torch
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, CheckpointSaveConfig, ModelConfigType
from fast_llm.engine.checkpoint.convert import ConvertConfig
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.models.gpt.conversion.config import GptOssCheckpointFormat
from fast_llm.engine.checkpoint.config import FastLLMCheckpointFormat
from tests.utils.compare_tensor_logs import CompareConfig
import transformers


def test_gpt_oss_20b_forward_equivalence():
    """Test that HuggingFace and Fast-LLM produce equivalent outputs for GPT-OSS 20B."""
    print("="*80)
    print("Testing GPT-OSS 20B Forward Pass Equivalence")
    print("="*80)

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

            print(f"\n2. Converting to Fast-LLM format...")
            print(f"   Source: {hf_local_path}")
            print(f"   Target: {fast_llm_path}")

            # Convert HF model to Fast-LLM format
            ConvertConfig(
                input=CheckpointLoadConfig(
                    path=hf_local_path,
                    format=GptOssCheckpointFormat,
                    load_config=ModelConfigType.model,
                ),
                output=CheckpointSaveConfig(
                    path=fast_llm_path,
                    format=FastLLMCheckpointFormat,
                ),
                model=GPTModelConfig,
            ).run()

            print(f"\n3. Loading HuggingFace model...")
            # Load the model from the downloaded files
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                hf_local_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).cuda()

            print(f"\n4. Loading Fast-LLM model (from converted checkpoint)...")
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

            print(f"\n5. Creating test input...")
            vocab_size = hf_model.config.vocab_size
            test_input = torch.randint(
                0,
                vocab_size,
                size=(2, 32),  # Small batch and sequence length
                dtype=torch.int64,
                device="cuda",
            )
            print(f"   Input shape: {test_input.shape}")
            print(f"   Vocab size: {vocab_size}")

            print(f"\n6. Running forward passes...")

            # Run HuggingFace model
            print(f"   Running HuggingFace model...")
            with torch.no_grad():
                hf_output = hf_model(test_input)

            # Run Fast-LLM model
            print(f"   Running Fast-LLM model...")
            with torch.no_grad():
                fast_llm_output = fast_llm_model(test_input)

            print(f"\n7. Comparing outputs...")
            print(f"   HF output shape: {hf_output.logits.shape}")
            print(f"   Fast-LLM output shape: {fast_llm_output.logits.shape}")
            print(f"   HF output dtype: {hf_output.logits.dtype}")
            print(f"   Fast-LLM output dtype: {fast_llm_output.logits.dtype}")

            # Compare using Fast-LLM's comparison utility
            errors = []
            CompareConfig().compare_tensors(
                {"samples": hf_output.logits, "shape": hf_output.logits.shape, "step": 0},
                {"samples": fast_llm_output.logits, "shape": fast_llm_output.logits.shape, "step": 0},
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
            print(f"   HF logits mean: {hf_output.logits.mean().item():.4f}")
            print(f"   Fast-LLM logits mean: {fast_llm_output.logits.mean().item():.4f}")
            print(f"   Absolute difference mean: {(hf_output.logits - fast_llm_output.logits).abs().mean().item():.6f}")
            print(f"   Max absolute difference: {(hf_output.logits - fast_llm_output.logits).abs().max().item():.6f}")

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
