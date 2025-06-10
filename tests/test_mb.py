import pytest

from tests.utils.compare_tensor_logs import CompareConfig
from tests.utils.model_configs import ModelTestingGroup


# TODO: Compare grads with simple
@pytest.mark.model_testing_group(ModelTestingGroup.basic)
def test_model_df4(run_test_script_for_all_models):
    # Depth-first gradient accumulation baseline.
    run_test_script_for_all_models(["batch.depth_first_micro_batches=4"])


@pytest.mark.depends_on(on=["test_model_df4[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_df4_z3(run_test_script_for_all_models):
    # Gradient accumulation with ZeRO-3.
    run_test_script_for_all_models(
        ["model.multi_stage.zero_stage=3", "batch.depth_first_micro_batches=4"],
        num_gpus=2,
        compare="test_model_df4",
        config=CompareConfig(ignore_duplicates=["Global gradient"]),
    )


@pytest.mark.depends_on(on=["test_model_df4[{model_testing_config}]"], scope="session")
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_bf4(run_test_script_for_all_models):
    # Breadth-first gradient accumulation baseline.
    run_test_script_for_all_models(["batch.breadth_first_micro_batches=4"], compare="test_model_df4")


@pytest.mark.depends_on(on=["test_model_df4[{model_testing_config}]", "test_model_bf4[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_bf2_df2(run_test_script_for_all_models):
    # Mixed gradient accumulation baseline.
    run_test_script_for_all_models(
        ["batch.depth_first_micro_batches=2", "batch.breadth_first_micro_batches=2"], compare="test_model_df4"
    )


@pytest.mark.depends_on(on=["test_model_bf4[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_pp2s2_bf4(run_test_script_for_all_models):
    # Pipeline-parallel without tied weights.
    run_test_script_for_all_models(
        [
            "batch.breadth_first_micro_batches=4",
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=2",
        ],
        num_gpus=2,
        compare="test_model_df4",
    )


@pytest.mark.depends_on(on=["test_model_bf4[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_pp2s1_bf4(run_test_script_for_all_models):
    # Pipeline-parallel with tied weights.
    run_test_script_for_all_models(
        [
            "batch.breadth_first_micro_batches=4",
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=1",
        ],
        num_gpus=2,
        compare="test_model_df4",
        config=CompareConfig(
            ignore_duplicates=[
                "layers.0.word_embeddings_weight",
                "layers.0.position_embeddings_weight",
            ]
        ),
    )


@pytest.mark.depends_on(on=["test_model_bf4[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_dp2_tp2_pp2s2_bf4(run_test_script_for_all_models):
    # Simple 3d parallelism
    # TODO: Test fails
    run_test_script_for_all_models(
        [
            "batch.breadth_first_micro_batches=4",
            "model.distributed.tensor_parallel=2",
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=1",
        ],
        num_gpus=8,
        compare="test_model_df4",
    )
