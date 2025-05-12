import pytest
import torch

import huggingface_hub

from transformers import AutoTokenizer, AutoModelForCausalLM

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig
from fast_llm.models.gpt.config import LlamaGPTHuggingfaceCheckpointFormat
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM

from tests.common import requires_cuda, TEST_RESULTS_PATH


def _prepare_checkpoint(model: str) -> str:
    path = TEST_RESULTS_PATH.resolve() / "generate/model"
    model_path = huggingface_hub.snapshot_download(repo_id=model, local_dir=path)
    return model_path


def _prepare_data(tokenizer, use_batch_size2: bool):
    messages = [
        {"role": "user", "content": "What is gravity?"},
        {"role": "user", "content": "Who is the president of EU?"},
    ]
    if not use_batch_size2:
        messages = messages[0:1]

    input_text = [tokenizer.apply_chat_template([el], tokenize=False) for el in messages]

    tokenizer.padding_side = "left"
    inputs = tokenizer(input_text, padding="longest", return_tensors="pt").to("cuda")
    return inputs


def _get_hf_model(model_path: str, use_flash_attention: bool, use_bf16: bool):
    hf_kwargs = {}
    if use_flash_attention:
        hf_kwargs["attn_implementation"] = "flash_attention_2"
        hf_kwargs["torch_dtype"] = torch.bfloat16
    elif use_bf16:
        hf_kwargs["torch_dtype"] = torch.bfloat16
    return AutoModelForCausalLM.from_pretrained(model_path, **hf_kwargs).to("cuda")


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


def _trim_output(output, inputs):
    res = []
    for output_row, input_row in zip(output, inputs["input_ids"]):
        res.append(output_row[len(input_row) :])
    return res


def _generate_with_params(
    tokenizer,
    model_path: str,
    use_flash_attention: bool,
    use_bf16: bool,
    use_batch_size2: bool,
    max_new_tokens: int,
    fast_llm_checkpoint_format=LlamaGPTHuggingfaceCheckpointFormat,
):
    inputs = _prepare_data(tokenizer, use_batch_size2)

    hf_model = _get_hf_model(model_path, use_flash_attention, use_bf16)
    fast_llm_model = _get_fast_llm_model(model_path, use_flash_attention, use_bf16, fast_llm_checkpoint_format)

    return {
        "hf": _trim_output(hf_model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=False), inputs),
        "fast_llm": _trim_output(
            fast_llm_model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=False), inputs
        ),
    }


def _compare_gen_outputs(outputs: dict[str, list], min_matching_tokens: int | None = None):
    for hf_output, fast_llm_output in zip(outputs["hf"], outputs["fast_llm"]):
        if min_matching_tokens is not None:
            hf_output = hf_output[:min_matching_tokens]
            fast_llm_output = fast_llm_output[:min_matching_tokens]
        assert len(hf_output) == len(fast_llm_output) and all(
            hf_char == fast_llm_char for hf_char, fast_llm_char in zip(hf_output, fast_llm_output)
        )


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model = "HuggingFaceTB/SmolLM2-135M-Instruct"
    fast_llm_checkpoint_format = LlamaGPTHuggingfaceCheckpointFormat
    model_path = _prepare_checkpoint(model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model_path, tokenizer, fast_llm_checkpoint_format


@pytest.mark.slow
@requires_cuda
@pytest.mark.parametrize(
    "use_flash_attention, use_bf16, use_batch_size2, max_new_tokens, min_matching_tokens",
    [
        # No flash attention + no bf16
        (False, False, False, 10, 10),
        (False, False, True, 10, 10),
        # No flash attention + with bf16
        (False, True, False, 10, 10),
        (False, True, True, 10, 10),
        # Flash attention must be paired with bf16
        (True, True, False, 10, 10),
        (True, True, True, 10, 10),
    ],
)
def test_generate(
    model_and_tokenizer, use_flash_attention, use_bf16, use_batch_size2, max_new_tokens, min_matching_tokens
):
    model_path, tokenizer, fast_llm_checkpoint_format = model_and_tokenizer
    outputs = _generate_with_params(
        tokenizer,
        model_path,
        use_flash_attention=use_flash_attention,
        use_bf16=use_bf16,
        use_batch_size2=use_batch_size2,
        max_new_tokens=max_new_tokens,
        fast_llm_checkpoint_format=fast_llm_checkpoint_format,
    )
    _compare_gen_outputs(outputs, min_matching_tokens=min_matching_tokens)
