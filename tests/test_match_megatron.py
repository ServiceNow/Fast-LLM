import pytest

from tests.utils.compare_tensor_logs import CompareConfig
from tests.utils.dataset import DATASET_PREFIX


@pytest.mark.slow
def test_megatron(run_test_script_for_all_models, model_testing_config):
    run_test_script_for_all_models(is_megatron=True)


@pytest.mark.slow
@pytest.mark.depends(on=["test_megatron"])
def test_match_megatron(run_test_script_for_all_models, model_testing_config):
    run_test_script_for_all_models(
        [
            "model.distributed.training_dtype=fp32",
            "data.datasets={}",
            f"data.path={DATASET_PREFIX}",
            "model.base_model.use_megatron_initialization=True",
        ],
        compare="test_megatron",
        config=CompareConfig(
            ignore_tensors=[
                ".self_attn.query_key_value.",
                ".self_attn.query.",
                ".self_attn.key_value.",
                ".mlp.layer_2.weight",
            ]
        ),
        use_performance_args=False,
    )
