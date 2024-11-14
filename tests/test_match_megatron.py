import pytest

from tests.common import (
    CONFIG_GPT2_FAST_LLM,
    CONFIG_GPT2_MEGATRON,
    CONFIG_MISTRAL_FAST_LLM,
    CONFIG_MISTRAL_MEGATRON,
    CONFIG_MIXTRAL_FAST_LLM,
    CONFIG_MIXTRAL_MEGATRON,
    CONFIG_SC1_FAST_LLM,
    CONFIG_SC1_MEGATRON,
    CONFIG_SC2_FAST_LLM,
    CONFIG_SC2_MEGATRON,
    run_test_script,
)
from tests.compare_tensor_logs import CompareConfig


@pytest.mark.slow
@pytest.mark.skip(reason="Skipping mostly redundant test")
def test_sc1_meg():
    # Starcoder 1 (GPT2 with MQA) with Megatron.
    run_test_script("test_sc1_meg", CONFIG_SC1_MEGATRON + ["--micro-batch-size=8"], is_megatron=True)


@pytest.mark.depends(on=["test_sc1_meg"])
def test_sc1_match_meg():
    # Starcoder 1 (GPT2 with MQA) with Fast-llm.
    # QKV tensors are in a different format.
    run_test_script(
        "test_sc1_match_meg",
        CONFIG_SC1_FAST_LLM + ["model.base_model.use_megatron_initialization=True"],
        compare="test_sc1_meg",
        config=CompareConfig(
            ignore_tensors=[
                ".self_attn.query_key_value.",
                ".self_attn.query.",
                ".self_attn.key_value.",
                ".mlp.layer_2.weight",
            ]
        ),
    )


@pytest.mark.slow
@pytest.mark.skip(reason="Skipping mostly redundant test")
@pytest.mark.depends(on=["test_sc1_match_meg"])
def test_sc2_meg():
    # Starcoder 2 (GPT2 with MQA and RoPE) with Megatron.
    run_test_script("test_sc2_meg", CONFIG_SC2_MEGATRON + ["--micro-batch-size=8"], is_megatron=True)


@pytest.mark.depends(on=["test_sc2_meg"])
def test_sc2_match_meg():
    # Starcoder 2 (GPT2 with MQA and RoPE) with Fast-llm.
    # QKV tensors are in a different format,
    # dense not matching because of the way initialization is corrected for RoPE format.
    run_test_script(
        "test_sc2_match_meg",
        CONFIG_SC2_FAST_LLM + ["model.base_model.use_megatron_initialization=True"],
        compare="test_sc2_meg",
        config=CompareConfig(
            ignore_tensors=[
                ".self_attn.query_key_value.",
                ".self_attn.query.",
                ".self_attn.key_value.",
                ".self_attn.dense.",
                ".mlp.layer_2.weight",
            ]
        ),
    )


@pytest.mark.slow
def test_gpt2_meg():
    # GPT2 (MHA, layer norm, absolute embeddings) with Megatron.
    run_test_script("test_gpt2_meg", CONFIG_GPT2_MEGATRON + ["--micro-batch-size=8"], is_megatron=True)


@pytest.mark.depends(on=["test_gpt2_meg"])
def test_gpt2_match_meg():
    # GPT2 (MHA, layer norm, absolute embeddings) with Fast-llm.
    # QKV tensors are in a different format.
    run_test_script(
        "test_gpt2_match_meg",
        CONFIG_GPT2_FAST_LLM + ["model.base_model.use_megatron_initialization=True"],
        compare="test_gpt2_meg",
        config=CompareConfig(
            ignore_tensors=[
                ".self_attn.query_key_value.",
                ".self_attn.query.",
                ".self_attn.key_value.",
                ".mlp.layer_2.weight",
            ]
        ),
    )


@pytest.mark.slow
def test_mistral_meg():
    # Mistral with Megatron.
    # No linear bias, swiglu activation, RMSNorm
    run_test_script("test_mistral_meg", CONFIG_MISTRAL_MEGATRON + ["--micro-batch-size=8"], is_megatron=True)


@pytest.mark.depends(on=["test_mistral_meg"])
def test_mistral_match_meg():
    # Mistral with Fast-LLM.
    run_test_script(
        "test_mistral_match_meg",
        CONFIG_MISTRAL_FAST_LLM + ["model.base_model.use_megatron_initialization=True"],
        compare="test_mistral_meg",
        config=CompareConfig(
            ignore_tensors=[
                ".self_attn.query_key_value.",
                ".self_attn.query.",
                ".self_attn.key_value.",
                ".self_attn.dense.",
                ".mlp.layer_2.weight",
            ]
        ),
    )


@pytest.mark.slow
def test_mixtral_meg():
    # Mistral with Megatron.
    # No linear bias, swiglu activation, RMSNorm
    run_test_script("test_mixtral_meg", CONFIG_MIXTRAL_MEGATRON + ["--micro-batch-size=8"], is_megatron=True)


@pytest.mark.depends(on=["test_mixtral_meg"])
def test_mixtral_match_meg():
    # Mistral with Fast-LLM.
    run_test_script(
        "test_mixtral_match_meg",
        CONFIG_MIXTRAL_FAST_LLM + ["model.base_model.use_megatron_initialization=True"],
        compare="test_mixtral_meg",
        config=CompareConfig(
            ignore_tensors=[
                ".self_attn.query_key_value.",
                ".self_attn.query.",
                ".self_attn.key_value.",
                ".self_attn.dense.",
                ".mlp.layer_1.weight",
                ".mlp.layer_2.weight",
                ".mlp.experts",
                "Global layer 2 fw: Transformer layer 2 output",
            ],
            max_rel_tolerance=1.5e-1,
        ),
    )
