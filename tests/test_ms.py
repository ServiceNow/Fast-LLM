import pytest

from tests.common import CONFIG_COMMON, TEST_MODEL, run_test_script

CONFIG_MS = CONFIG_COMMON + ["--micro_sequence_length=256"]


# TODO: Compare grads with simple
@pytest.mark.depends(
    on=["tests/test_seq_first.py::test_model_sf"],
)
def test_model_ms256():
    # Micro-sequence baseline
    run_test_script(f"test_{TEST_MODEL}_ms256", CONFIG_MS)


@pytest.mark.depends(on=["test_model_ms256"])
def test_model_pp2s2_ms256():
    # Sequence-pipeline-parallel
    run_test_script(
        f"test_{TEST_MODEL}_pp2s2_ms256",
        CONFIG_MS + ["--pipeline-parallel=2", "--layers_per_stage=2"],
        num_gpus=2,
        compare=f"test_{TEST_MODEL}_ms256",
    )


@pytest.mark.skip
@pytest.mark.depends(on=["test_model_ms256"])
def test_model_dp2s2_stp2_pp2s2_ms256():
    # TODO: Handle this case.
    # Sequence-3d-parallel
    run_test_script(
        f"test_{TEST_MODEL}_dp2s2_stp2_pp2s2_ms256",
        CONFIG_MS
        + [
            "--pipeline-parallel=2",
            "--layers_per_stage=2",
            "--tensor-parallel=2",
            "--sequence_tensor_parallel=1",
            "--sequence_data_parallel=2",
        ],
        num_gpus=8,
        compare=f"test_{TEST_MODEL}_ms256",
    )
