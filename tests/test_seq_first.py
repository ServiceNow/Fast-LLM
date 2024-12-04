import pytest

from tests.common import CONFIG_COMMON, TEST_MODEL, run_test_script

CONFIG_SF = CONFIG_COMMON + ["model.base_model.sequence_first=True"]


# TODO: Compare grads with simple
def test_model_sf():
    # Sequence-first baseline.
    run_test_script(f"test_{TEST_MODEL}_sf", CONFIG_SF)


@pytest.mark.slow
@pytest.mark.depends(on=["test_model_sf"])
def test_model_sp2():
    # Sequence-tensor-parallel.
    run_test_script(
        f"test_{TEST_MODEL}_sp2",
        CONFIG_SF + ["model.distributed.tensor_parallel=2", "model.distributed.sequence_tensor_parallel=True"],
        num_gpus=2,
        compare=f"test_{TEST_MODEL}_sf",
    )


@pytest.mark.slow
@pytest.mark.depends(on=["test_model_sf"])
def test_model_sdp2():
    # Sequence-data-parallel
    run_test_script(
        f"test_{TEST_MODEL}_sdp2",
        CONFIG_COMMON + ["model.distributed.sequence_data_parallel=2"],
        num_gpus=2,
        compare=f"test_{TEST_MODEL}_sf",
    )


@pytest.mark.slow
@pytest.mark.depends(on=["test_model_sf"])
def test_model_sp2_ce4():
    # Sequence-tensor-parallel with cross-entropy splits.
    run_test_script(
        f"test_{TEST_MODEL}_sp2_ce4",
        CONFIG_SF
        + [
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "model.base_model.parallel_embeddings=False",
            "model.base_model.cross_entropy_splits=4",
        ],
        num_gpus=2,
        compare=f"test_{TEST_MODEL}_sf",
    )
