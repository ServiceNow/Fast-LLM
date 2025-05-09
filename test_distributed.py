# distributed_example.py
import os
import torch
import torch.distributed as dist

from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.models.gpt.config import LlamaGPTHuggingfaceCheckpointFormat, Qwen2GPTHuggingfaceCheckpointFormat
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM


def run(
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    checkpoint="/mnt/checkpoints/pretrained_models/Qwen2-1.5B-Instruct/",
):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    updates = {
        ("base_model", "transformer", "use_flash_attention"): attn_implementation is not None
        and attn_implementation == "flash_attention_2",
        ("distributed", "tensor_parallel"): 2,
        ("distributed", "pipeline_parallel"): 1,
        ("distributed", "sequence_data_parallel"): 1,
    }

    if torch_dtype is not None and torch_dtype == torch.bfloat16:
        updates[("distributed", "training_dtype")] = "bf16"

    print("aupdatesgs", updates)

    model_fm = HuggingfaceGPTModelForCausalLM.from_pretrained(
        CheckpointLoadConfig(
            path=checkpoint,
            format=Qwen2GPTHuggingfaceCheckpointFormat,
        ),
        updates,
    )

    input_ids = torch.randint(1, tokenizer.vocab_size, (10, 100), dtype=torch.int64, generator=torch.Generator().manual_seed(42))

    res = model_fm.forward(input_ids, use_cache=False)
    print(res.logits.shape, res.logits.sum().item())


def main():
    run()


if __name__ == "__main__":
    main()
