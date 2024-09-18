import pytest

from tests.common import CONFIG_COMMON, CONFIG_FAST_LLM, TEST_MODEL, run_test_script


@pytest.mark.depends(
    depends=[f"tests/test_match_megatron.py::test_{TEST_MODEL}_match_meg"],
)
def test_model_safe():
    # The safest possible config, identical to the one in test_match_megatron except for the initialization.
    run_test_script(
        f"test_{TEST_MODEL}_safe",
        CONFIG_FAST_LLM + ["--torch_dynamo_enable=0", "--data_overlap=0", "--dropless_moe=0"],
    )


@pytest.mark.depends(on=["test_model_safe"])
def test_model():
    # A baseline config (single-gpu, bf16, flash-attn).
    run_test_script(f"test_{TEST_MODEL}", CONFIG_COMMON, compare=f"test_{TEST_MODEL}_safe")


@pytest.mark.depends(on=["test_model"])
def test_model_dp2():
    # Simple data-parallel.
    run_test_script(f"test_{TEST_MODEL}_dp2", CONFIG_COMMON, num_gpus=2, compare=f"test_{TEST_MODEL}")


@pytest.mark.depends(on=["test_model"])
def test_model_tp2():
    # Simple tensor-parallel.
    run_test_script(
        f"test_{TEST_MODEL}_tp2", CONFIG_COMMON + ["--tensor-parallel=2"], num_gpus=2, compare=f"test_{TEST_MODEL}"
    )


@pytest.mark.depends(on=["test_model"])
def test_model_ce4():
    # Cross-entropy splits.
    run_test_script(
        f"test_{TEST_MODEL}_ce4", CONFIG_COMMON + ["--cross_entropy_splits=4"], compare=f"test_{TEST_MODEL}"
    )


@pytest.mark.depends(on=["test_model"])
def test_model_dp2_z2():
    # Data-parallel with zero stage 2.
    run_test_script(
        f"test_{TEST_MODEL}_dp2_z2", CONFIG_COMMON + ["--zero_stage=2"], num_gpus=2, compare=f"test_{TEST_MODEL}"
    )


@pytest.mark.depends(on=["test_model"])
def test_model_dp2_z3():
    # Data-parallel with zero stage 3.
    run_test_script(
        f"test_{TEST_MODEL}_dp2_z3", CONFIG_COMMON + ["--zero_stage=3"], num_gpus=2, compare=f"test_{TEST_MODEL}"
    )
