import huggingface_hub
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.schedule.config import ScheduleConfig
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.models.gpt.config import PretrainedGPTModelConfig
from fast_llm.models.gpt.conversion.config import LlamaCheckpointFormat
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
from tests.utils.model_configs import ModelTestingGroup
from tests.utils.utils import requires_cuda


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


def _prepare_rand_data(vocab_size, use_batch_size2: bool):
    inputs = torch.randint(
        1,
        vocab_size,
        [2 if use_batch_size2 else 1, 10],
        dtype=torch.int64,
        generator=torch.Generator().manual_seed(42),
    ).cuda()
    attention_mask = torch.ones_like(inputs)
    # simulate left padding on one of the rows
    if use_batch_size2:
        inputs[1, :5] = 0
        attention_mask[1, :5] = 0
    return {"input_ids": inputs, "attention_mask": attention_mask}


def _get_hf_model(model_path: str, use_flash_attention: bool, use_bf16: bool):
    hf_kwargs = {"trust_remote_code": True}
    if use_flash_attention:
        hf_kwargs["attn_implementation"] = "flash_attention_2"
        hf_kwargs["torch_dtype"] = torch.bfloat16
    elif use_bf16:
        hf_kwargs["torch_dtype"] = torch.bfloat16
    return AutoModelForCausalLM.from_pretrained(model_path, **hf_kwargs).to("cuda")


def _get_fast_llm_model(
    model_path: str, use_flash_attention: bool, use_bf16: bool, checkpoint_format=LlamaCheckpointFormat
):
    updates = {}
    if use_flash_attention:
        updates[("base_model", "decoder", "block", "mixer", "use_flash_attention")] = True
        updates[("distributed", "training_dtype")] = "bf16"
    else:
        updates[("base_model", "decoder", "block", "mixer", "use_flash_attention")] = False
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


def _get_fast_llm_model_from_model(
    model_path: str, use_flash_attention: bool, use_bf16: bool, checkpoint_format=LlamaCheckpointFormat
):
    updates = {
        ("pretrained", "path"): model_path,
        ("pretrained", "model_weights"): True,
        ("pretrained", "format"): checkpoint_format.name,
    }

    if use_flash_attention:
        updates[("model", "base_model", "decoder", "block", "mixer", "use_flash_attention")] = True
        updates[("model", "distributed", "training_dtype")] = "bf16"
    else:
        updates[("model", "base_model", "decoder", "block", "mixer", "use_flash_attention")] = False
        if use_bf16:
            updates[("model", "distributed", "training_dtype")] = "bf16"

    config = PretrainedGPTModelConfig.from_dict({}, updates)
    multi_stage = config.model.get_model_class()(config.model)
    schedule_config = ScheduleConfig()
    runner = ScheduleRunner(
        config=schedule_config,
        multi_stage=multi_stage,
        distributed_config=config.model.distributed,
    )
    distributed = Distributed(config.model.distributed)

    with torch.no_grad():
        multi_stage.setup(distributed)
        runner.setup(distributed)

    multi_stage.load_checkpoint(config.pretrained)

    return HuggingfaceGPTModelForCausalLM(multi_stage, runner=runner)


def _trim_output(output, inputs):
    res = []
    for output_row, input_row in zip(output, inputs["input_ids"]):
        res.append(output_row[len(input_row) :])
    return res


def _generate(
    inputs,
    hf_model,
    fast_llm_model,
    max_new_tokens: int,
):
    return {
        "hf": _trim_output(hf_model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=False), inputs),
        "fast_llm": _trim_output(
            fast_llm_model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=False), inputs
        ),
    }


def _compare_gen_outputs(outputs: dict[str, list[torch.Tensor]], min_matching_tokens: int | None = None):
    for hf_output, fast_llm_output in zip(outputs["hf"], outputs["fast_llm"]):
        if min_matching_tokens is not None:
            hf_output = hf_output[:min_matching_tokens]
            fast_llm_output = fast_llm_output[:min_matching_tokens]
        assert torch.equal(hf_output, fast_llm_output)


def _test_for_batches(
    hf_model,
    fast_llm_model,
    max_new_tokens,
    min_matching_tokens_batch_size_1,
    min_matching_tokens_batch_size_2,
    tokenizer=None,
):
    if tokenizer is not None:
        inputs = _prepare_data(tokenizer, use_batch_size2=False)
    else:
        inputs = _prepare_rand_data(fast_llm_model.config.fast_llm_config.base_model.vocab_size, use_batch_size2=False)
    outputs = _generate(
        inputs,
        hf_model,
        fast_llm_model,
        max_new_tokens=max_new_tokens,
    )
    _compare_gen_outputs(outputs, min_matching_tokens=min_matching_tokens_batch_size_1)

    if tokenizer is not None:
        inputs = _prepare_data(tokenizer, use_batch_size2=True)
    else:
        inputs = _prepare_rand_data(fast_llm_model.config.fast_llm_config.base_model.vocab_size, use_batch_size2=True)
    outputs = _generate(
        inputs,
        hf_model,
        fast_llm_model,
        max_new_tokens=max_new_tokens,
    )
    _compare_gen_outputs(outputs, min_matching_tokens=min_matching_tokens_batch_size_2)


@pytest.fixture(scope="module")
def model_path(result_path):
    return huggingface_hub.snapshot_download(
        repo_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        local_dir=result_path / "generate/model",
    )


def _test_generate(
    model_path,
    fast_llm_checkpoint_format,
    use_flash_attention,
    use_bf16,
    max_new_tokens,
    min_matching_tokens_batch_size_1,
    min_matching_tokens_batch_size_2,
    tokenizer=None,
):
    hf_model = _get_hf_model(model_path, use_flash_attention, use_bf16)
    fast_llm_model = _get_fast_llm_model(model_path, use_flash_attention, use_bf16, fast_llm_checkpoint_format)

    _test_for_batches(
        hf_model,
        fast_llm_model,
        max_new_tokens,
        min_matching_tokens_batch_size_1,
        min_matching_tokens_batch_size_2,
        tokenizer=tokenizer,
    )


@pytest.mark.extra_slow
@requires_cuda
@pytest.mark.parametrize(
    "use_flash_attention, use_bf16, max_new_tokens, min_matching_tokens_batch_size_1, min_matching_tokens_batch_size_2",
    [
        # No flash attention + no bf16
        (False, False, 10, 10, 10),
        # No flash attention + with bf16
        (False, True, 10, 10, 10),
        # Flash attention must be paired with bf16
        (True, True, 10, 10, 10),
    ],
)
def test_generate(
    model_path,
    use_flash_attention,
    use_bf16,
    max_new_tokens,
    min_matching_tokens_batch_size_1,
    min_matching_tokens_batch_size_2,
):
    _test_generate(
        model_path,
        LlamaCheckpointFormat,
        use_flash_attention,
        use_bf16,
        max_new_tokens,
        min_matching_tokens_batch_size_1,
        min_matching_tokens_batch_size_2,
        tokenizer=AutoTokenizer.from_pretrained(model_path),
    )


@pytest.mark.slow
@pytest.mark.model_testing_group(ModelTestingGroup.generate)
def test_export_for_generate(run_test_script_for_all_models, model_testing_config):
    # Not really testing, anything, but handles dependencies more easily than a fixture.
    if model_testing_config.checkpoint_format is None:
        pytest.skip(f"Conversion not supported for {model_testing_config.name}")
    run_test_script_for_all_models(
        [
            "training.train_iters=1",
            f"training.export.format={model_testing_config.checkpoint_format.name}",
            "training.export.interval=1",
        ],
    )


@pytest.mark.slow
@requires_cuda
@pytest.mark.depends_on(on=["test_export_for_generate[{model_testing_config}]"])
@pytest.mark.parametrize(
    "use_flash_attention, use_bf16, max_new_tokens, min_matching_tokens_batch_size_1, min_matching_tokens_batch_size_2",
    [
        # No flash attention + no bf16
        (False, False, 10, 10, 10),
        # No flash attention + with bf16
        (False, True, 10, 10, 10),
        # Flash attention must be paired with bf16
        (True, True, 10, 10, 10),
    ],
)
@pytest.mark.model_testing_group(ModelTestingGroup.generate)
def test_small_generate(
    model_testing_config,
    run_test_script_base_path,
    use_flash_attention,
    use_bf16,
    max_new_tokens,
    min_matching_tokens_batch_size_1,
    min_matching_tokens_batch_size_2,
):
    _test_generate(
        run_test_script_base_path / f"test_export_for_generate/export/{model_testing_config.checkpoint_format.name}/1",
        model_testing_config.checkpoint_format,
        use_flash_attention,
        use_bf16,
        max_new_tokens,
        min_matching_tokens_batch_size_1,
        min_matching_tokens_batch_size_2,
    )


def _test_generate_from_model(model_path, tokenizer, fast_llm_checkpoint_format):
    max_new_tokens = 10
    min_matching_tokens_batch_size_1 = 10
    min_matching_tokens_batch_size_2 = 10

    # Use flash attention for speed
    hf_model = _get_hf_model(model_path, True, True)
    fast_llm_model = _get_fast_llm_model_from_model(model_path, True, True, fast_llm_checkpoint_format)

    _test_for_batches(
        hf_model,
        fast_llm_model,
        max_new_tokens,
        min_matching_tokens_batch_size_1,
        min_matching_tokens_batch_size_2,
        tokenizer=tokenizer,
    )


@requires_cuda
@pytest.mark.extra_slow
def test_generate_from_model(
    model_path,
):
    _test_generate_from_model(model_path, AutoTokenizer.from_pretrained(model_path), LlamaCheckpointFormat)


@requires_cuda
@pytest.mark.slow
@pytest.mark.depends_on(on=["test_export_for_generate[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.generate)
def test_small_generate_from_model(model_testing_config, run_test_script_base_path):
    _test_generate_from_model(
        run_test_script_base_path / f"test_export_for_generate/export/{model_testing_config.checkpoint_format.name}/1",
        None,
        model_testing_config.checkpoint_format,
    )


def _test_forward_return_hidden_states(
    model_path,
    fast_llm_checkpoint_format,
    vocab_size: int | None = None,
):
    # Use flash attention for speed
    # TODO: hidden states have differences between HF and Fast-LLM despite resulting in the similar logits,
    #       decide if to leave as it.
    # hf_model = _get_hf_model(model_path, True, True)
    fast_llm_model = _get_fast_llm_model(model_path, True, True, fast_llm_checkpoint_format)

    inputs_ids = torch.randint(
        1,
        fast_llm_model.config.fast_llm_config.base_model.vocab_size if vocab_size is None else vocab_size,
        [1, 10],
        dtype=torch.int64,
        generator=torch.Generator().manual_seed(42),
    ).cuda()

    # res_hf = hf_model.forward(input_ids=inputs_ids, output_hidden_states=True, return_dict=True, use_cache=False)
    res_fast_llm = fast_llm_model.forward(
        input_ids=inputs_ids, output_hidden_states=True, return_dict=True, use_cache=False
    )

    # hidden_states include embeddings layer
    assert len(res_fast_llm.hidden_states) - 1 == len(fast_llm_model.config.fast_llm_config.base_model.decoder)


@pytest.mark.extra_slow
@requires_cuda
def test_forward_return_hidden_states(model_path):
    _test_forward_return_hidden_states(
        model_path, LlamaCheckpointFormat, AutoTokenizer.from_pretrained(model_path).vocab_size
    )


@pytest.mark.slow
@requires_cuda
@pytest.mark.model_testing_group(ModelTestingGroup.generate)
@pytest.mark.depends_on(on=["test_export_for_generate[{model_testing_config}]"])
def test_small_forward_return_hidden_states(model_testing_config, run_test_script_base_path):
    _test_forward_return_hidden_states(
        run_test_script_base_path / f"test_export_for_generate/export/{model_testing_config.checkpoint_format.name}/1",
        model_testing_config.checkpoint_format,
    )
