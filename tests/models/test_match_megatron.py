import pytest

from tests.utils.compare_tensor_logs import CompareConfig
from tests.utils.dataset import DATASET_PREFIX
from tests.utils.model_configs import ModelTestingGroup


@pytest.mark.model_testing_group(ModelTestingGroup.megatron)
def test_megatron(run_test_script_for_all_models, model_testing_config):
    run_test_script_for_all_models([], is_megatron=True)


@pytest.mark.depends_on(on=["test_megatron[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.megatron)
def test_match_megatron(run_test_script_for_all_models, model_testing_config):
    assert model_testing_config.megatron_args is not None

    ignore_tensors = [
        ".self_attn.query_key_value.",
        ".self_attn.query.",
        ".self_attn.key_value.",
        ".mlp.layer_2.weight",
        ".mlp.experts.",
    ]
    if model_testing_config.name == "mixtral":
        ignore_tensors.extend([".mlp.experts.", ".mlp.layer_1.weight"])

    run_test_script_for_all_models(
        [
            "model.distributed.training_dtype=fp32",
            "data.datasets={}",
            f"data.path={DATASET_PREFIX}",
            "model.base_model.use_megatron_initialization=True",
        ],
        compare="test_megatron",
        config=CompareConfig(ignore_tensors=ignore_tensors),
    )
