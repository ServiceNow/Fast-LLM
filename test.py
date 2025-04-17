import torch

from pathlib import Path
import shutil
import cloudpickle

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig
from fast_llm.models.gpt.config import LlamaGPTHuggingfaceCheckpointFormat

import torch


def generate(model, input_ids, attention_mask, max_new_tokens, tensors_save_path: Path | None = None):

    if tensors_save_path is not None:
        if tensors_save_path.is_dir():
            shutil.rmtree(tensors_save_path, ignore_errors=True)
        logits_save_path = tensors_save_path / "logits"
        hs_save_path = tensors_save_path / "hidden_states"
        logits_save_path.mkdir(exist_ok=True, parents=True)
        hs_save_path.mkdir(exist_ok=True, parents=True)

    # assume attention mask is left padded with zeroes if any
    mask_step = torch.ones((attention_mask.shape[0], 1), dtype=torch.int64).to(attention_mask.device)
    for i in range(max_new_tokens):
        output: CausalLMOutputWithPast = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        current_ids = output.logits[:, -1, :].argmax(dim=1, keepdim=True)
        input_ids = torch.cat([input_ids, current_ids], dim=1)
        attention_mask = torch.cat([attention_mask, mask_step], dim=1)

        if tensors_save_path is not None:
            logits_file = logits_save_path / f"tensor{i}.pt"
            torch.save(output.logits, logits_file)

            hidden_states_file = hs_save_path / f"data{i}.pickle"
            with hidden_states_file.open("wb") as f:
                cloudpickle.dump(output.hidden_states, f)

    return input_ids


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


def run_test_fast_llm(
    attn_implementation,
    torch_dtype,
    is_batch_size2,
    reverse_samples,
    tensors_save_path,
    num_new_tokens,
):
    checkpoint = "/mnt/checkpoints/pretrained_models/SmolLM2-135M-Instruct"

    device = "cuda"  # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    messages = [
        # {"role": "user", "content": "What is gravity?"},
        {"role": "user", "content": "What is gravity?"},
        {"role": "user", "content": "Who is the president of EU?"},
        # {"role": "user", "content": "Who is the president of EU?"},
    ]
    if reverse_samples:
        messages = list(reversed(messages))
    if not is_batch_size2:
        messages = messages[0:1]

    input_text = [tokenizer.apply_chat_template([el], tokenize=False) for el in messages]

    tokenizer.padding_side = "left"
    inputs = tokenizer(input_text, padding="longest", return_tensors="pt").to(device)

    fm_kwards = {}
    if attn_implementation is not None and attn_implementation == "flash_attention_2":
        fm_kwards["attn_implementation"] = "flash_attention_2"
    else:
        fm_kwards["attn_implementation"] = "fuse"
    if torch_dtype is not None and torch_dtype == torch.bfloat16:
        fm_kwards["torch_dtype"] = "bf16"

    print("fm_kwards", fm_kwards)

    model_fm = HuggingfaceGPTModelForCausalLM.from_pretrained(
        CheckpointLoadConfig(
            path=checkpoint,
            format=LlamaGPTHuggingfaceCheckpointFormat,
        ),
        **fm_kwards,
    )

    # outputs_fm = model_fm.generate(**inputs, max_new_tokens=50, use_cache=False)
    outputs_fm = generate(
        model_fm, **inputs, max_new_tokens=num_new_tokens, tensors_save_path=tensors_save_path / "fast_llm"
    )

    print(tokenizer.decode(outputs_fm[0][inputs["input_ids"].shape[1] :]))
    if len(outputs_fm) > 1:
        print("--------------------------------------------------------------")
        print(tokenizer.decode(outputs_fm[1][inputs["input_ids"].shape[1] :]))


def run_test(
    attn_implementation,
    torch_dtype,
    is_batch_size2,
    reverse_samples,
    tensors_save_path,
    num_new_tokens,
):
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
    model_hf = AutoModelForCausalLM.from_pretrained(checkpoint, **hf_kwards).to(device)

    messages = [
        # {"role": "user", "content": "What is gravity?"},
        {"role": "user", "content": "Who is the president of EU?"},
        {"role": "user", "content": "Who is the president of EU?"},
    ]
    if reverse_samples:
        messages = list(reversed(messages))
    if not is_batch_size2:
        messages = messages[0:1]

    input_text = [tokenizer.apply_chat_template([el], tokenize=False) for el in messages]

    tokenizer.padding_side = "left"
    inputs = tokenizer(input_text, padding="longest", return_tensors="pt").to(device)

    # outputs_hf = model_hf.generate(**inputs, max_new_tokens=50, use_cache=False)
    outputs_hf = generate(
        model_hf, **inputs, max_new_tokens=num_new_tokens, tensors_save_path=tensors_save_path / "hf"
    )
    # print(tokenizer.decode(outputs_hf[0]))

    fm_kwards = {}
    if attn_implementation is not None and attn_implementation == "flash_attention_2":
        fm_kwards["attn_implementation"] = "flash_attention_2"
    else:
        fm_kwards["attn_implementation"] = "fuse"
    if torch_dtype is not None and torch_dtype == torch.bfloat16:
        fm_kwards["torch_dtype"] = "bf16"

    print("fm_kwards", fm_kwards)

    model_fm = HuggingfaceGPTModelForCausalLM.from_pretrained(
        CheckpointLoadConfig(
            path=checkpoint,
            format=LlamaGPTHuggingfaceCheckpointFormat,
        ),
        **fm_kwards,
    )

    # outputs_fm = model_fm.generate(**inputs, max_new_tokens=50, use_cache=False)
    outputs_fm = generate(
        model_fm, **inputs, max_new_tokens=num_new_tokens, tensors_save_path=tensors_save_path / "fast_llm"
    )

    diff_flm_hf(
        tokenizer, outputs_fm[0][inputs["input_ids"].shape[1] :], outputs_hf[0][inputs["input_ids"].shape[1] :]
    )
    if len(outputs_fm) > 1:
        diff_flm_hf(
            tokenizer, outputs_fm[1][inputs["input_ids"].shape[1] :], outputs_hf[1][inputs["input_ids"].shape[1] :]
        )


def main():
    run_test_fast_llm(
        # run_test(
        attn_implementation="flash_attention_2",
        # attn_implementation=None,
        torch_dtype=torch.bfloat16,
        # torch_dtype=None,
        is_batch_size2=True,
        reverse_samples=False,
        # tensors_save_path=Path("/mnt/datasets/tests/denis/tensors_bf16_flash_attention_2_batch_size2/"),
        tensors_save_path=Path("/mnt/datasets/tests/denis/tmp/"),
        num_new_tokens=100,
    )


if __name__ == "__main__":
    main()
