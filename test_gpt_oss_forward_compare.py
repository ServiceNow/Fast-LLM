#!/usr/bin/env python3
"""
Step 2: Convert checkpoint and compare forward passes between HF and Fast-LLM.
"""

import os
import pathlib
import sys

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

# Set PyTorch memory allocator to use expandable segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


sys.path.insert(0, "/home/ubuntu/Fast-LLM")

# Configuration
CHECKPOINT_DIR = pathlib.Path("/home/ubuntu/Fast-LLM/test_gpt_oss_checkpoint")
DEQUANTIZED_HF_PATH = CHECKPOINT_DIR / "dequantized_hf"
FAST_LLM_PATH = CHECKPOINT_DIR / "fast_llm"


def test_gpt_oss_forward_equivalence():
    """Test that HuggingFace and Fast-LLM produce equivalent outputs."""
    print("=" * 80)
    print("Testing GPT-OSS Forward Pass Equivalence")
    print("=" * 80)

    if not DEQUANTIZED_HF_PATH.exists():
        print(f"\n❌ Error: Checkpoint not found at {DEQUANTIZED_HF_PATH}")
        print(f"   Please run prepare_gpt_oss_checkpoint.py first!")
        return False

    try:
        # Load config to get vocab size
        config = transformers.AutoConfig.from_pretrained(DEQUANTIZED_HF_PATH)
        vocab_size = config.vocab_size

        print(f"\n1. Converting to Fast-LLM format...")
        print(f"   Source: {DEQUANTIZED_HF_PATH}")
        print(f"   Target: {FAST_LLM_PATH}")

        ConvertConfig(
            input=CheckpointLoadConfig(
                path=DEQUANTIZED_HF_PATH,
                format=GptOssCheckpointFormat,
                load_config=ModelConfigType.model,
            ),
            output=CheckpointSaveConfig(
                path=FAST_LLM_PATH,
                format=FastLLMCheckpointFormat,
            ),
            model=GPTModelConfig,
        ).run()

        print(f"\n2. Creating test input...")
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

        print(f"\n3. Loading HuggingFace model and running forward pass...")
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            DEQUANTIZED_HF_PATH,
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
                print(
                    f"   HF {name}: shape={output_tensor.shape}, mean={output_tensor.mean().item():.6f}, std={output_tensor.std().item():.6f}"
                )

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

        # Memory cleanup
        import gc

        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()

        print(f"   GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

        print(f"\n4. Loading Fast-LLM model and running forward pass...")
        from fast_llm.engine.config_utils.logging import TensorLogs, TensorLogsConfig
        from fast_llm.logging import set_model_debug_level
        from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
        from fast_llm.models.gpt.model import GPTModel

        # Initialize TensorLogs and enable debug mode
        TensorLogs.reset(TensorLogsConfig(save=False, show=True, max_elements=8))
        set_model_debug_level(3)

        print(f"   Debug level set to: 3")

        # Load the base GPT model first
        gpt_model = GPTModel.from_pretrained(
            CheckpointLoadConfig(
                path=FAST_LLM_PATH,
                format=FastLLMCheckpointFormat,
                load_config=ModelConfigType.model,
            )
        )

        # Then wrap it with the HuggingFace interface
        fast_llm_model = HuggingfaceGPTModelForCausalLM(gpt_model)

        print(f"   Running Fast-LLM model...")
        with torch.no_grad():
            fast_llm_output = fast_llm_model(test_input)

        fast_llm_logits = fast_llm_output.logits.clone()

        print(f"\n5. Comparing outputs...")
        hf_logits = hf_logits.cuda()

        print(f"   HF output shape: {hf_logits.shape}, dtype: {hf_logits.dtype}")
        print(f"   Fast-LLM output shape: {fast_llm_logits.shape}, dtype: {fast_llm_logits.dtype}")

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

        print(f"\n✅ Forward pass equivalence test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed:")
        print(f"   Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gpt_oss_forward_equivalence()
    sys.exit(0 if success else 1)
