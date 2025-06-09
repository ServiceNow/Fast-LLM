import pytest

from tests.utils.compare_tensor_logs import CompareConfig
from tests.utils.model_configs import ModelTestingGroup


# TODO: Compare grads with simple
@pytest.mark.model_testing_group(ModelTestingGroup.basic)
def test_model_df4_sf(run_test_script_for_all_models):
    # Sequence-first gradient accumulation baseline.
    run_test_script_for_all_models(["batch.depth_first_micro_batches=4", "model.base_model.sequence_first=True"])


@pytest.mark.depends_on(on=["test_model_df4_sf[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_dp2_sp2_df4(run_test_script_for_all_models):
    # Sequence-tensor-parallel with gradient accumulation.
    # TODO: Compiled cross-entropy broken for this config
    run_test_script_for_all_models(
        [
            "batch.breadth_first_micro_batches=4",
            "model.base_model.sequence_first=True",
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "run.torch_dynamo_enable=False",
        ],
        num_gpus=4,
        compare="test_model_df4_sf",
    )


@pytest.mark.skip(reason="Test is broken.")
@pytest.mark.depends_on(on=["test_model_df4_sf[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_model_dp2_sp2_pp2s1(run_test_script_for_all_models):
    # 3d-parallel with sequence-tensor-parallel.
    # TODO: Compiled cross-entropy broken for this config
    run_test_script_for_all_models(
        [
            "batch.breadth_first_micro_batches=4",
            "model.base_model.sequence_first=True",
            "model.distributed.tensor_parallel=2",
            "model.distributed.pipeline_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "run.torch_dynamo_enable=False",
        ],
        num_gpus=8,
        compare="test_model_df4_sf",
        config=CompareConfig(ignore_duplicates=["layers.0.word_embeddings_weight"]),
    )
