import os

import pytest

from tests.utils.compare_tensor_logs import CompareConfig
from tests.utils.dataset import MODEL_DATASET_PREFIX, get_model_test_dataset
from tests.utils.distributed_configs import DistributedTestingConfig
from tests.utils.model_configs import ModelTestingGroup
from tests.utils.utils import requires_cuda


@requires_cuda
@pytest.mark.model_testing_group(ModelTestingGroup.megatron)
def test_megatron(run_distributed_script, model_testing_config, run_test_script_base_path):
    path = run_test_script_base_path / "megatron"
    env = os.environ.copy()
    # Prevent Megatron from complaining.
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["NVTE_FLASH_ATTN"] = "0"
    get_model_test_dataset()
    run_distributed_script(
        [
            "Megatron-LM/pretrain_gpt.py",
            *model_testing_config.megatron_args,
            f"--structured-logs-dir={path}",
            f"--data-cache-path={path}",
        ],
        num_gpus=1,
        env=env,
    )


@requires_cuda
@pytest.mark.depends_on(on=["test_megatron[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.megatron)
def test_match_megatron(run_test_script_for_all_models, model_testing_config, compare_results_for_all_models):
    assert model_testing_config.megatron_args is not None

    ignore_tensors = (
        ".self_attn.query_key_value.",
        ".self_attn.query.",
        ".self_attn.key_value.",
        ".mlp.layer_2.weight",
        ".mlp.experts.",
    )
    if model_testing_config.name == "mixtral":
        ignore_tensors += (".mlp.experts.", ".mlp.layer_1.weight")

    distributed_testing_config = DistributedTestingConfig(
        name="match_megatron",
        compare="megatron",
        config_args=[
            "model.distributed.training_dtype=fp32",
            "data.datasets={}",
            f"data.path={MODEL_DATASET_PREFIX}",
            "model.base_model.use_megatron_initialization=True",
        ],
        num_gpus=1,
        compare_config=CompareConfig(sub_configs={(None, ignore_tensors): CompareConfig(ignore_tensors=True)}),
    )

    run_test_script_for_all_models(distributed_testing_config)
    compare_results_for_all_models(distributed_testing_config)
