from transformers import AutoTokenizer, AutoModelForCausalLM
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig
from fast_llm.models.gpt.config import LlamaGPTHuggingfaceCheckpointFormat

import torch


def diff_flm_hf(tokenizer, flm_tokens, hf_tokens):
    print("+++++++++++++++fast_llm:+++++++++++++++++++++++++++++++++++++++++++++++++")
    fllm_str = tokenizer.decode(flm_tokens)
    print(fllm_str)
    print("---------------hugging_face:---------------------------------------------")
    hf_str = tokenizer.decode(hf_tokens)
    print(hf_str)
    print(
        f"==============================({"Same" if fllm_str==hf_str else "Different"})====================================="
    )


def run_test(attn_implementation, torch_dtype, is_batch_size2, use_fm_changes):
    checkpoint = "/mnt/checkpoints/pretrained_models/SmolLM2-135M-Instruct"

    device = "cuda"  # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
    hf_kwards = {}
    if attn_implementation is not None and attn_implementation == "flash_attention_2":
        hf_kwards["attn_implementation"] = "flash_attention_2"
    if torch_dtype is not None:
        hf_kwards["torch_dtype"] = torch_dtype

    print("hf_kwards", hf_kwards)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, **hf_kwards).to(device)

    messages = [
        {"role": "user", "content": "What is gravity?"},
    ]
    if is_batch_size2:
        messages += [
            {"role": "user", "content": "Who is the president of EU?"},
        ]

    input_text = [tokenizer.apply_chat_template([el], tokenize=False) for el in messages]

    tokenizer.padding_side = "left"
    inputs = tokenizer(input_text, padding="longest", return_tensors="pt").to(device)

    outputs_hf = model.generate(**inputs, max_new_tokens=50, use_cache=False)
    # print(tokenizer.decode(outputs_hf[0]))

    fm_kwards = {}
    if attn_implementation is not None and attn_implementation == "flash_attention_2":
        fm_kwards["attn_implementation"] = "flash_attention_2"
    else:
        fm_kwards["attn_implementation"] = "fuse"
    if torch_dtype is not None and torch_dtype == torch.bfloat16:
        fm_kwards["torch_dtype"] = "bf16"

    print("fm_kwards", fm_kwards)

    fm_model = HuggingfaceGPTModelForCausalLM.from_pretrained(
        CheckpointLoadConfig(
            path=checkpoint,
            format=LlamaGPTHuggingfaceCheckpointFormat,
        ),
        use_fm_changes=use_fm_changes,
        **fm_kwards,
    )

    outputs = fm_model.generate(**inputs, max_new_tokens=50, use_cache=False)

    diff_flm_hf(tokenizer, outputs[0][inputs["input_ids"].shape[1] :], outputs_hf[0][inputs["input_ids"].shape[1] :])
    if len(outputs) > 1:
        diff_flm_hf(
            tokenizer, outputs[1][inputs["input_ids"].shape[1] :], outputs_hf[1][inputs["input_ids"].shape[1] :]
        )


def main():
    run_test(attn_implementation=None, torch_dtype=None, is_batch_size2=False, use_fm_changes=False)


if __name__ == "__main__":
    main()
