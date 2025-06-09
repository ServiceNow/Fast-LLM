import pytest

from tests.utils.model_configs import ModelTestingGroup


# TODO: Compare grads with simple
@pytest.mark.model_testing_group(ModelTestingGroup.basic)
def test_model_ms256(run_test_script_for_all_models):
    # Micro-sequence baseline
    run_test_script_for_all_models(["batch.micro_sequence_length=256"])


@pytest.mark.depends_on(on=["test_model_ms256[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_pp2s2_ms256(run_test_script_for_all_models):
    # Sequence-pipeline-parallel
    run_test_script_for_all_models(
        [
            "batch.micro_sequence_length=256",
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=2",
        ],
        num_gpus=2,
        compare="test_model_ms256",
    )


@pytest.mark.skip
@pytest.mark.depends_on(on=["test_model_ms256[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_dp2s2_stp2_pp2s2_ms256(run_test_script_for_all_models):
    # TODO: Handle this case.
    # Sequence-3d-parallel
    run_test_script_for_all_models(
        [
            "batch.micro_sequence_length=256",
            "model.distributed.pipeline_parallel=2",
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "model.distributed.sequence_data_parallel=2",
            "model.multi_stage.layers_per_stage=2",
        ],
        num_gpus=8,
        compare="test_model_ms256",
    )
