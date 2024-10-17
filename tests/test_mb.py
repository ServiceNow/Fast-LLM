import pytest

from tests.common import CONFIG_COMMON, TEST_MODEL, run_test_script
from tests.compare_tensor_logs import CompareConfig

CONFIG_DF = CONFIG_COMMON + ["batch.depth_first_micro_batches=4"]
CONFIG_BF = CONFIG_COMMON + ["batch.breadth_first_micro_batches=4"]
CONFIG_BF_DF = CONFIG_COMMON + ["batch.depth_first_micro_batches=2", "batch.breadth_first_micro_batches=2"]


# TODO: Compare grads with simple
@pytest.mark.depends(on=["tests/test_simple.py::test_model"], scope="session")
def test_model_df4():
    # Depth-first gradient accumulation baseline.
    run_test_script(f"test_{TEST_MODEL}_df4", CONFIG_DF)


@pytest.mark.depends(on=["test_model_df4"])
def test_model_df4_z3():
    # Gradient accumulation with ZeRO-3.
    run_test_script(
        f"test_{TEST_MODEL}_df4_z3",
        CONFIG_DF + ["model.multi_stage.zero_stage=3"],
        num_gpus=2,
        compare=f"test_{TEST_MODEL}_df4",
        config=CompareConfig(ignore_duplicates=["Global gradient"]),
    )


@pytest.mark.depends(on=["test_model_df4"], scope="session")
def test_model_bf4():
    # Breadth-first gradient accumulation baseline.
    run_test_script(f"test_{TEST_MODEL}_bf4", CONFIG_BF, compare=f"test_{TEST_MODEL}_df4")


@pytest.mark.depends(on=["test_model_df4", "test_model_bf4"])
def test_model_bf2_df2():
    # Mixed gradient accumulation baseline.
    run_test_script(f"test_{TEST_MODEL}_bf2_df2", CONFIG_BF_DF, compare=f"test_{TEST_MODEL}_df4")


@pytest.mark.depends(on=["test_model_bf4"])
def test_model_pp2s2_bf4():
    # Pipeline-parallel without tied weights.
    run_test_script(
        f"test_{TEST_MODEL}_pp2s2_bf4",
        CONFIG_BF + ["model.distributed.pipeline_parallel=2", "model.multi_stage.layers_per_stage=2"],
        num_gpus=2,
        compare=f"test_{TEST_MODEL}_df4",
    )


@pytest.mark.depends(on=["test_model_bf4"])
def test_model_pp2s1_bf4():
    # Pipeline-parallel with tied weights.
    run_test_script(
        f"test_{TEST_MODEL}_pp2s1_bf4",
        CONFIG_BF + ["model.distributed.pipeline_parallel=2", "model.multi_stage.layers_per_stage=1"],
        num_gpus=2,
        compare=f"test_{TEST_MODEL}_df4",
        config=CompareConfig(ignore_duplicates=["layers.0.word_embeddings_weight"]),
    )


@pytest.mark.depends(on=["test_model_bf4"])
def test_model_dp2_tp2_pp2s2_bf4():
    # Simple 3d parallelism
    # TODO: Test fails
    run_test_script(
        f"test_{TEST_MODEL}_dp2_tp2_pp2s2_bf4",
        CONFIG_BF
        + [
            "model.distributed.tensor_parallel=2",
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=1",
        ],
        num_gpus=8,
        compare=f"test_{TEST_MODEL}_df4",
    )
