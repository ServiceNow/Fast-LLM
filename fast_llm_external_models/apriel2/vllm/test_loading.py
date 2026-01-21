#!/usr/bin/env python3
"""Test script for loading Apriel2 stochastic mixer models in vLLM.

Focused on testing model loading and (eventually) runtime mixer switching.

Usage:
    python test_loading.py /path/to/model
    python test_loading.py /path/to/model --no-compile
"""

import argparse
import sys
from pathlib import Path

import torch
import triton

def _triton_allocator(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device='cuda').data_ptr()

triton.set_allocator(_triton_allocator)

from vllm import LLM, ModelRegistry
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

from fast_llm_external_models.apriel2.vllm.modeling_apriel2 import Apriel2ForCausalLM
ModelRegistry.register_model(
    "Apriel2ForCausalLM",
    "fast_llm_external_models.apriel2.vllm:Apriel2ForCausalLM",
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


def main():
    parser = argparse.ArgumentParser(description="Test Apriel2 stochastic model loading")
    parser.add_argument("model_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")

    compilation_config = CompilationConfig(mode=CompilationMode.NONE) if args.no_compile else None

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.3,
        max_model_len=512,
        dtype="bfloat16",
        compilation_config=compilation_config,
        disable_log_stats=True,
        enable_prefix_caching=False,
    )

    # Model loaded successfully
    print("\nModel loaded successfully!")

    # Note: Model inspection requires different approach in vLLM v1
    # The model is in a subprocess, so direct access isn't available here

    print("\nLoad test passed!")


if __name__ == "__main__":
    main()
