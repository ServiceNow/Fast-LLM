import pytest

from tests.common import CONFIG_COMMON, TEST_MODEL, run_test_script
from tests.compare_tensor_logs import CompareConfig

CONFIG_DF_SF = CONFIG_COMMON + ["batch.depth_first_micro_batches=4", "model.base_model.sequence_first=True"]
CONFIG_BF_SF = CONFIG_COMMON + ["batch.breadth_first_micro_batches=4", "model.base_model.sequence_first=True"]
CONFIG_BF_DF_SF = CONFIG_COMMON + [
    "batch.depth_first_micro_batches=2",
    "batch.breadth_first_micro_batches=2",
    "model.base_model.sequence_first=True",
]


# TODO: Compare grads with simple
@pytest.mark.depends(
    on=["tests/test_mb.py::test_model_df4", "tests/test_seq_first.py::test_model_sf"],
)
def test_model_df4_sf():
    # Sequence-first gradient accumulation baseline.
    run_test_script(f"test_{TEST_MODEL}_df4_sf", CONFIG_DF_SF)


@pytest.mark.slow
@pytest.mark.depends(on=["test_model_df4_sf"])
def test_model_dp2_sp2_df4():
    # Sequence-tensor-parallel with gradient accumulation.
    # TODO: Compiled cross-entropy broken for this config
    run_test_script(
        f"test_{TEST_MODEL}_dp2_sp2_df4",
        CONFIG_BF_SF
        + [
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "run.torch_dynamo_enable=False",
        ],
        num_gpus=4,
        compare=f"test_{TEST_MODEL}_df4_sf",
    )


@pytest.mark.slow
@pytest.mark.skip(reason="Test is broken.")
@pytest.mark.depends(on=["test_model_df4_sf"])
def test_model_dp2_sp2_pp2s1():
    # 3d-parallel with sequence-tensor-parallel.
    # TODO: Compiled cross-entropy broken for this config
    run_test_script(
        f"test_{TEST_MODEL}_dp2_sp2_pp2s1",
        CONFIG_BF_SF
        + [
            "model.distributed.tensor_parallel=2",
            "model.distributed.pipeline_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "run.torch_dynamo_enable=False",
        ],
        num_gpus=8,
        compare=f"test_{TEST_MODEL}_df4_sf",
        config=CompareConfig(ignore_duplicates=["layers.0.word_embeddings_weight"]),
    )
