import pytest

from tests.utils.model_configs import ModelTestingGroup


@pytest.mark.model_testing_group(ModelTestingGroup.basic)
def test_model_safe(run_test_script_for_all_models):
    # The safest possible config, identical to the one in test_match_megatron except for the initialization.
    run_test_script_for_all_models(
        [
            "model.distributed.training_dtype=fp32",
            "run.torch_dynamo_enable=False",
            "schedule.data_overlap=False",
            "model.base_model.transformer.dropless_moe=False",
        ],
    )


@pytest.mark.depends_on(on=["test_model_safe[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.basic)
def test_model(run_test_script_for_all_models):
    # A baseline config (single-gpu, bf16, flash-attn).
    # Also tests for multiple data loaders.
    run_test_script_for_all_models(["training.num_workers=2"], compare="test_model_safe")


@pytest.mark.depends_on(on=["test_model[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_dp2(run_test_script_for_all_models):
    # Simple data-parallel.
    run_test_script_for_all_models([], num_gpus=2, compare="test_model")


@pytest.mark.skip(reason="Flaky")
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_dp2_timeout(run_test_script_for_all_models):
    # Test sampling timeout
    # TODO: Find a better way to test this
    run_test_script_for_all_models(
        [
            # Use a short timeout
            "model.distributed.timeout=4",
            # Make a dataset that would timeout under the distributed timeout
            'data.datasets.training={"type":"test_slow"}',
            "data.datasets.training.type=test_slow",
            "data.datasets.training.sleep=6",
            # Use a bigger timeout for the dataset.
            "training.timeout=10",
            # Remove testing clutter.
            "model.multi_stage.debug_param_init=0",
            "model.multi_stage.debug_layer_outputs=0",
            "model.multi_stage.debug_layer_gradients=0",
            "model.multi_stage.debug_all_param_gradients=0",
        ],
        num_gpus=2,
    )


@pytest.mark.depends_on(on=["test_model[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_tp2(run_test_script_for_all_models):
    # Simple tensor-parallel.
    run_test_script_for_all_models(
        ["model.distributed.tensor_parallel=2"],
        num_gpus=2,
        compare="test_model",
    )


@pytest.mark.depends_on(on=["test_model[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.basic)
def test_model_ce4(run_test_script_for_all_models):
    # Cross-entropy splits.
    run_test_script_for_all_models(
        ["model.base_model.cross_entropy_splits=4"],
        compare="test_model",
    )


@pytest.mark.depends_on(on=["test_model[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_dp2_z2(run_test_script_for_all_models):
    # Data-parallel with zero stage 2.
    run_test_script_for_all_models(
        ["model.multi_stage.zero_stage=2"],
        num_gpus=2,
        compare="test_model",
    )


@pytest.mark.depends_on(on=["test_model[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_dp2_z3(run_test_script_for_all_models):
    # Data-parallel with zero stage 3.
    run_test_script_for_all_models(
        ["model.multi_stage.zero_stage=3"],
        num_gpus=2,
        compare="test_model",
    )
