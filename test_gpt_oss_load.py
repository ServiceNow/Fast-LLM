#!/usr/bin/env python3
"""
Test loading GPT-OSS 20B model from HuggingFace.
"""

import sys
sys.path.insert(0, '/home/ubuntu/Fast-LLM')

import torch
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig
from fast_llm.engine.config_utils.run import log_pipeline_parallel_main_rank
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.models.gpt.conversion.config import GptOssCheckpointFormat


def test_load_gpt_oss_20b():
    """Test loading GPT-OSS 20B model."""
    print("="*80)
    print("Testing GPT-OSS 20B Model Loading")
    print("="*80)

    # Model path
    model_path = "openai/gpt-oss-20b"
    print(f"\nModel: {model_path}")

    try:
        print("\n1. Loading model configuration from HuggingFace...")

        # Load the HuggingFace config
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Get the checkpoint handler
        from fast_llm.models.gpt.conversion.gpt_oss import GptOssHuggingfaceCheckpointHandler
        handler_class = GptOssHuggingfaceCheckpointHandler
        print(f"   Handler: {handler_class.__name__}")

        print(f"\n2. HuggingFace Config loaded:")
        print(f"   Architecture: {hf_config.architectures}")
        print(f"   Hidden size: {hf_config.hidden_size}")
        print(f"   Num layers: {hf_config.num_hidden_layers}")
        print(f"   Num experts: {hf_config.num_local_experts}")
        print(f"   Experts per token: {hf_config.num_experts_per_tok}")
        print(f"   Vocab size: {hf_config.vocab_size}")
        print(f"   Has layer_types: {hasattr(hf_config, 'layer_types')}")
        if hasattr(hf_config, 'layer_types'):
            print(f"   Layer types: {hf_config.layer_types[:10]}..." if len(hf_config.layer_types) > 10 else f"   Layer types: {hf_config.layer_types}")

        print(f"\n3. Converting to Fast-LLM config...")
        # Convert HuggingFace config to Fast-LLM config
        fast_llm_config_dict = handler_class.base_model_converter_class.import_config(hf_config.to_dict())

        print(f"\n4. Fast-LLM Config structure:")
        print(f"   Hidden size: {fast_llm_config_dict.get('hidden_size')}")
        print(f"   Decoder type: {fast_llm_config_dict.get('decoder', {}).get('type')}")
        if 'decoder' in fast_llm_config_dict:
            decoder = fast_llm_config_dict['decoder']
            if 'blocks' in decoder:
                print(f"   Block types: {list(decoder['blocks'].keys())}")
                print(f"   Pattern: {decoder.get('pattern', 'N/A')}")
            print(f"   Num blocks: {decoder.get('num_blocks')}")

        print(f"\n5. Checking MLP config...")
        if 'decoder' in fast_llm_config_dict:
            decoder = fast_llm_config_dict['decoder']
            if 'blocks' in decoder:
                for block_name, block_config in decoder['blocks'].items():
                    mlp_config = block_config.get('mlp', {})
                    print(f"   Block '{block_name}' MLP:")
                    print(f"      Type: {mlp_config.get('type')}")
                    print(f"      Experts: {mlp_config.get('experts')}")
                    print(f"      Experts per token: {mlp_config.get('experts_per_token')}")
            elif 'block' in decoder:
                mlp_config = decoder['block'].get('mlp', {})
                print(f"   MLP:")
                print(f"      Type: {mlp_config.get('type')}")
                print(f"      Experts: {mlp_config.get('experts')}")
                print(f"      Experts per token: {mlp_config.get('experts_per_token')}")

        print(f"\n✅ Successfully loaded and converted GPT-OSS 20B config!")
        return True

    except Exception as e:
        print(f"\n❌ Failed to load GPT-OSS 20B:")
        print(f"   Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_load_gpt_oss_20b()
    sys.exit(0 if success else 1)
