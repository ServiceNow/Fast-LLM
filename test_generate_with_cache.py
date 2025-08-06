import time

import torch
from transformers import AutoTokenizer

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig
from fast_llm.models.gpt.config import LlamaGPTHuggingfaceCheckpointFormat
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM


def get_model_path(result_path=None):
    return "/mnt/checkpoints/pretrained_models/SmolLM2-135M-Instruct/"


def _trim_output(output, inputs):
    res = []
    for output_row, input_row in zip(output, inputs["input_ids"]):
        res.append(output_row[len(input_row) :])
    return res


def _compare_gen_outputs(outputs: dict[str, list[torch.Tensor]], min_matching_tokens: int | None = None):
    keys = list(outputs.keys())
    assert len(keys) == 2, f"Only 2 inputs can be compared, {len(keys)} provided."
    for hf_output, fast_llm_output in zip(outputs[keys[0]], outputs[keys[1]]):
        if min_matching_tokens is not None:
            hf_output = hf_output[:min_matching_tokens]
            fast_llm_output = fast_llm_output[:min_matching_tokens]
        assert torch.equal(hf_output, fast_llm_output)


def _prepare_rand_data(vocab_size, batch_size: int, prompt_length: int = 10, simulate_left_padding: bool = True):
    gen = torch.Generator().manual_seed(42)

    inputs = torch.randint(
        1,
        vocab_size,
        [batch_size, prompt_length],
        dtype=torch.int64,
        generator=gen,
    ).cuda()
    attention_mask = torch.ones_like(inputs)

    if batch_size > 1 and simulate_left_padding:
        # Randomly choose a single row to remain unpadded
        unpadded_row = torch.randint(0, batch_size, (1,), generator=gen).item()

        for i in range(batch_size):
            if i == unpadded_row:
                continue
            # Random left padding length between 1 and prompt_length - 1
            pad_len = torch.randint(1, prompt_length, (1,), generator=gen).item()
            inputs[i, :pad_len] = 0
            attention_mask[i, :pad_len] = 0

    return {"input_ids": inputs, "attention_mask": attention_mask}


def _get_fast_llm_model(
    model_path: str, use_flash_attention: bool, use_bf16: bool, checkpoint_format=LlamaGPTHuggingfaceCheckpointFormat
):
    updates = {}
    if use_flash_attention:
        updates[("base_model", "transformer", "use_flash_attention")] = True
        updates[("distributed", "training_dtype")] = "bf16"
    else:
        updates[("base_model", "transformer", "use_flash_attention")] = False
        if use_bf16:
            updates[("distributed", "training_dtype")] = "bf16"
    return HuggingfaceGPTModelForCausalLM.from_pretrained(
        CheckpointLoadConfig(
            path=model_path,
            format=checkpoint_format,
            model_weights=True,
        ),
        updates,
    )


def test_generate_with_cache(batch_size, max_new_tokens, use_flash_attention, speed_up_threshold):
    model_path = get_model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    fast_llm_model = _get_fast_llm_model(
        model_path=model_path,
        use_flash_attention=use_flash_attention,
        use_bf16=True,
        checkpoint_format=LlamaGPTHuggingfaceCheckpointFormat,
    )

    inputs = _prepare_rand_data(tokenizer.vocab_size, batch_size)

    t0 = time.time()
    output_no_cache = fast_llm_model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=False)
    dt_no_cache = time.time() - t0

    t0 = time.time()
    output_with_cache = fast_llm_model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    dt_with_cache = time.time() - t0

    _compare_gen_outputs(
        {
            "output_no_cache": _trim_output(output_no_cache, inputs),
            "output_with_cache": _trim_output(output_with_cache, inputs),
        }
    )

    assert dt_no_cache / dt_with_cache > speed_up_threshold


if __name__ == "__main__":
    test_generate_with_cache(1, 200, True, 1.5)
