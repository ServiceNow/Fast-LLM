import pytest

from tests.utils.model_configs import ModelTestingGroup


# TODO: Compare grads with simple
@pytest.mark.model_testing_group(ModelTestingGroup.basic)
def test_model_sf(run_test_script_for_all_models):
    # Sequence-first baseline.
    run_test_script_for_all_models(["model.base_model.sequence_first=True"])


@pytest.mark.depends_on(on=["test_model_sf[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_sp2(run_test_script_for_all_models):
    # Sequence-tensor-parallel.
    run_test_script_for_all_models(
        ["model.distributed.tensor_parallel=2", "model.distributed.sequence_tensor_parallel=True"],
        num_gpus=2,
        compare="test_model_sf",
    )


@pytest.mark.depends_on(on=["test_model_sf[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_sdp2(run_test_script_for_all_models):
    # Sequence-data-parallel
    run_test_script_for_all_models(
        ["model.distributed.sequence_data_parallel=2"],
        num_gpus=2,
        compare="test_model_sf",
    )


@pytest.mark.depends_on(on=["test_model_sf[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_sp2_ce4(run_test_script_for_all_models):
    # Sequence-tensor-parallel with cross-entropy splits.
    run_test_script_for_all_models(
        [
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "model.base_model.parallel_embeddings=False",
            "model.base_model.cross_entropy_splits=4",
        ],
        num_gpus=2,
        compare="test_model_sf",
    )
