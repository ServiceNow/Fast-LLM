import pytest

from tests.common import CONFIG_COMMON, TEST_MODEL

CONFIG_MS = CONFIG_COMMON + ["batch.micro_sequence_length=256"]


# TODO: Compare grads with simple
def test_model_ms256(run_test_script):
    # Micro-sequence baseline
    run_test_script(f"test_{TEST_MODEL}_ms256", CONFIG_MS)


@pytest.mark.slow
@pytest.mark.depends(on=["test_model_ms256"])
def test_model_pp2s2_ms256(run_test_script):
    # Sequence-pipeline-parallel
    run_test_script(
        f"test_{TEST_MODEL}_pp2s2_ms256",
        CONFIG_MS + ["model.distributed.pipeline_parallel=2", "model.multi_stage.layers_per_stage=2"],
        num_gpus=2,
        compare=f"test_{TEST_MODEL}_ms256",
    )


@pytest.mark.slow
@pytest.mark.skip
@pytest.mark.depends(on=["test_model_ms256"])
def test_model_dp2s2_stp2_pp2s2_ms256(run_test_script):
    # TODO: Handle this case.
    # Sequence-3d-parallel
    run_test_script(
        f"test_{TEST_MODEL}_dp2s2_stp2_pp2s2_ms256",
        CONFIG_MS
        + [
            "model.distributed.pipeline_parallel=2",
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "model.distributed.sequence_data_parallel=2",
            "model.multi_stage.layers_per_stage=2",
        ],
        num_gpus=8,
        compare=f"test_{TEST_MODEL}_ms256",
    )
