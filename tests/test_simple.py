import pytest

from tests.common import CONFIG_COMMON, CONFIG_FAST_LLM, TEST_MODEL


def test_model_safe(run_test_script):
    # The safest possible config, identical to the one in test_match_megatron except for the initialization.
    run_test_script(
        f"test_{TEST_MODEL}_safe",
        CONFIG_FAST_LLM
        + [
            "run.torch_dynamo_enable=False",
            "schedule.data_overlap=False",
            "model.base_model.transformer.dropless_moe=False",
        ],
    )


@pytest.mark.depends(on=["test_model_safe"])
def test_model(run_test_script):
    # A baseline config (single-gpu, bf16, flash-attn).
    # Also tests for multiple data loaders.
    run_test_script(
        f"test_{TEST_MODEL}", CONFIG_COMMON + ["training.num_workers=2"], compare=f"test_{TEST_MODEL}_safe"
    )


@pytest.mark.slow
@pytest.mark.depends(on=["test_model"])
def test_model_dp2(run_test_script):
    # Simple data-parallel.
    run_test_script(f"test_{TEST_MODEL}_dp2", CONFIG_COMMON, num_gpus=2, compare=f"test_{TEST_MODEL}")


@pytest.mark.slow
def test_model_dp2_timeout(run_test_script):
    # Test sampling timeout
    # TODO: Find a better way to test this
    run_test_script(
        f"test_{TEST_MODEL}_dp2_timeout",
        CONFIG_COMMON
        + [
            # Use a short timeout
            "model.distributed.timeout=4",
            # Make a dataset that would timeout under the distributed timeout
            'data.datasets.training={"type":"test_slow"}',
            "data.datasets.training.type=test_slow",
            "data.datasets.training.sleep=6",
            # Use a bigger timeout for the dataset.
            "training.timeout=10",
            # Remove testing clutter.
            f"model.multi_stage.debug_param_init=0",
            f"model.multi_stage.debug_layer_outputs=0",
            f"model.multi_stage.debug_layer_gradients=0",
            f"model.multi_stage.debug_all_param_gradients=0",
        ],
        num_gpus=2,
    )


@pytest.mark.slow
@pytest.mark.depends(on=["test_model"])
def test_model_tp2(run_test_script):
    # Simple tensor-parallel.
    run_test_script(
        f"test_{TEST_MODEL}_tp2",
        CONFIG_COMMON + ["model.distributed.tensor_parallel=2"],
        num_gpus=2,
        compare=f"test_{TEST_MODEL}",
    )


@pytest.mark.depends(on=["test_model"])
def test_model_ce4(run_test_script):
    # Cross-entropy splits.
    run_test_script(
        f"test_{TEST_MODEL}_ce4",
        CONFIG_COMMON + ["model.base_model.cross_entropy_splits=4"],
        compare=f"test_{TEST_MODEL}",
    )


@pytest.mark.slow
@pytest.mark.depends(on=["test_model"])
def test_model_dp2_z2(run_test_script):
    # Data-parallel with zero stage 2.
    run_test_script(
        f"test_{TEST_MODEL}_dp2_z2",
        CONFIG_COMMON + ["model.multi_stage.zero_stage=2"],
        num_gpus=2,
        compare=f"test_{TEST_MODEL}",
    )


@pytest.mark.slow
@pytest.mark.depends(on=["test_model"])
def test_model_dp2_z3(run_test_script):
    # Data-parallel with zero stage 3.
    run_test_script(
        f"test_{TEST_MODEL}_dp2_z3",
        CONFIG_COMMON + ["model.multi_stage.zero_stage=3"],
        num_gpus=2,
        compare=f"test_{TEST_MODEL}",
    )
