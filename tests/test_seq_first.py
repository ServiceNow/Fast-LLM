import pytest

from tests.common import CONFIG_COMMON, TEST_MODEL, run_test_script

CONFIG_SF = CONFIG_COMMON + ["--sequence_first=1"]


# TODO: Compare grads with simple
@pytest.mark.depends(on=["tests/test_simple.py::test_model"], scope="session")
def test_model_sf():
    # Sequence-first baseline.
    run_test_script(f"test_{TEST_MODEL}_sf", CONFIG_SF)


@pytest.mark.depends(on=["test_model_sf"])
def test_model_sp2():
    # Sequence-tensor-parallel.
    run_test_script(
        f"test_{TEST_MODEL}_sp2",
        CONFIG_SF + ["--tensor-parallel=2", "--sequence_tensor_parallel=1"],
        num_gpus=2,
        compare=f"test_{TEST_MODEL}_sf",
    )


@pytest.mark.depends(on=["test_model_sf"])
def test_model_sp2_ce4():
    # Sequence-tensor-parallel with cross-entropy splits.
    run_test_script(
        f"test_{TEST_MODEL}_sp2_ce4",
        CONFIG_SF
        + [
            "--tensor-parallel=2",
            "--sequence_tensor_parallel=1",
            "--parallel_embeddings=0",
            "--cross_entropy_splits=4",
        ],
        num_gpus=2,
        compare=f"test_{TEST_MODEL}_sf",
    )
