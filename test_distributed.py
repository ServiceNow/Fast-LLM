# distributed_example.py
import os
import torch
import torch.distributed as dist

from dataclasses import dataclass
import functools
import time

from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.models.gpt.config import LlamaGPTHuggingfaceCheckpointFormat, Qwen2GPTHuggingfaceCheckpointFormat
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM

from fast_llm.core.distributed import scatter_object_list, gather_object


def run(
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    checkpoint="/mnt/checkpoints/pretrained_models/Qwen2-1.5B-Instruct/",
):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    updates = {
        ("base_model", "transformer", "use_flash_attention"): attn_implementation is not None
        and attn_implementation == "flash_attention_2",
        ("distributed", "tensor_parallel"): 1,
        ("distributed", "pipeline_parallel"): 1,
        ("distributed", "sequence_data_parallel"): 1,
        # ("distributed", "sequence_tensor_parallel"): True,
    }

    if torch_dtype is not None and torch_dtype == torch.bfloat16:
        updates[("distributed", "training_dtype")] = "bf16"

    print("aupdatesgs", updates)

    model_fm = HuggingfaceGPTModelForCausalLM.from_pretrained(
        CheckpointLoadConfig(
            path=checkpoint,
            format=Qwen2GPTHuggingfaceCheckpointFormat,
            model_weights=True,
        ),
        updates,
    )

    device = model_fm._inference_runner._fast_llm_model.distributed.device
    rank = model_fm._inference_runner._fast_llm_model.distributed._config.rank
    word_size = model_fm._inference_runner._fast_llm_model.distributed._config.world_size

    if rank == 0:
        batch_size = 32
        length = 20

        num_batches = 10
        t0 = time.time()
        for i in range(num_batches):
            input_ids = torch.randint(
                1,
                tokenizer.vocab_size,
                (batch_size, length),
                dtype=torch.int64,
                generator=torch.Generator().manual_seed(42+i),
            ).to(device)

            step = batch_size // word_size
            scatter_list = [(input_ids[i * step: (i + 1) * step], True) for i in range(word_size)]

            params = [None]
            scatter_object_list(device, params, scatter_list, model_fm._inference_runner._fast_llm_model.distributed.world_group, 0)
            input_ids = params[0][0]

            #res = model_fm.generate(input_ids, max_new_tokens=50, use_cache=False)

            res = input_ids
            res = res.to("cpu")
            

            global_res = [None] * word_size
            gather_object(device, res, global_res, model_fm._inference_runner._fast_llm_model.distributed.world_group, 0)

            res = torch.cat(global_res, dim=0)
            print(res.shape, res.sum().item())
        
        scatter_list = [(None, False) for i in range(word_size)]
        params = [None]
        scatter_object_list(device, params, scatter_list, model_fm._inference_runner._fast_llm_model.distributed.world_group, 0)

        print(time.time() - t0)

    else:
        while True:
            scatter_list = None

            params = [None]
            scatter_object_list(device, params, scatter_list, model_fm._inference_runner._fast_llm_model.distributed.world_group, 0)
            input_ids, continue_generate = params[0]
            if not continue_generate:
                break

            #res = model_fm.generate(input_ids, max_new_tokens=50, use_cache=False)

            res = input_ids
            res = res.to("cpu")

            if rank == 0:
                global_res = [None] * word_size
            else:
                global_res = None
            gather_object(device, res, global_res, model_fm._inference_runner._fast_llm_model.distributed.world_group, 0)

    
    # res = model_fm.forward(input_ids, use_cache=False)
    # if res.logits is not None:
    #     print(res.logits.shape, res.logits.sum().item())
    #     print(res.logits.argmax(dim=2, keepdim=False))

    # else:
    #     print("None")


def main():
    run()


if __name__ == "__main__":
    main()
    print("exiting")
