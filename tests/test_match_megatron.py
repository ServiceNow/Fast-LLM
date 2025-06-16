import pytest

from tests.utils.compare_tensor_logs import CompareConfig
from tests.utils.dataset import DATASET_PREFIX
from tests.utils.model_configs import CONFIG_COMMON, CONFIG_MEGATRON, TEST_MODEL


@pytest.mark.slow
def test_megatron(run_test_script):
    run_test_script(f"test_{TEST_MODEL}_megatron", CONFIG_MEGATRON, is_megatron=True)


CONFIG_MATCH_MEGATRON = [
    "data.datasets={}",
    f"data.path={DATASET_PREFIX}",
]


@pytest.mark.slow
@pytest.mark.depends_on(on=["test_megatron"])
def test_match_megatron(run_test_script):
    if CONFIG_MEGATRON is None:
        pytest.skip(f"Megatron does not support model {TEST_MODEL}")

    ignore_tensors = [
        ".self_attn.query_key_value.",
        ".self_attn.query.",
        ".self_attn.key_value.",
        ".mlp.layer_2.weight",
        ".mlp.experts.",
    ]
    if TEST_MODEL == "mixtral":
        ignore_tensors.extend([".mlp.experts.", ".mlp.layer_1.weight"])

    run_test_script(
        f"test_{TEST_MODEL}_match_megatron",
        CONFIG_COMMON
        + [
            "model.distributed.training_dtype=fp32",
            "data.datasets={}",
            f"data.path={DATASET_PREFIX}",
            "model.base_model.use_megatron_initialization=True",
        ],
        compare=f"test_{TEST_MODEL}_megatron",
        config=CompareConfig(ignore_tensors=ignore_tensors),
    )
