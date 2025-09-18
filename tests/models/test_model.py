import logging

import pytest
import torch

from tests.utils.distributed_configs import (
    DISTRIBUTED_TESTING_CONFIGS,
    SIMPLE_TESTING_CONFIG,
    SINGLE_GPU_TESTING_CONFIGS,
)
from tests.utils.model_configs import ModelTestingGroup
from tests.utils.utils import check_subtest_success, requires_cuda, set_subtest_success

logger = logging.getLogger(__name__)


@requires_cuda
@pytest.mark.model_testing_group(ModelTestingGroup.basic)
def test_model_simple(run_test_script_for_all_models, run_test_script_base_path):
    # A simple config to prevent unnecessary testing and creation of dependency group
    run_test_script_for_all_models(SIMPLE_TESTING_CONFIG)
    set_subtest_success(run_test_script_base_path / SIMPLE_TESTING_CONFIG.name)


@requires_cuda
@pytest.mark.depends_on(on=["test_model_simple[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.basic)
# Parametrize with config name so it shows in test name.
@pytest.mark.parametrize("config_name", SINGLE_GPU_TESTING_CONFIGS)
def test_and_compare_model(
    run_test_script_for_all_models,
    compare_results_for_all_models,
    config_name,
    run_test_script_base_path,
    model_testing_config,
):
    # We can expect tests to respect the ordering of `SINGLE_GPU_TESTING_CONFIGS`, so compare should have run already.
    config = SINGLE_GPU_TESTING_CONFIGS[config_name]
    if model_testing_config.should_skip(config):
        pytest.skip(f"Configuration not supported.")
    if config.compare is not None:
        check_subtest_success(run_test_script_base_path / config.compare)
    # A baseline config (single-gpu, bf16, flash-attn).
    # Also tests for multiple data loaders.
    run_test_script_for_all_models(config)
    set_subtest_success(run_test_script_base_path / config.name)

    if config.compare is not None:
        compare_results_for_all_models(config)


@requires_cuda
@pytest.mark.depends_on(on=["test_model_simple[{model_testing_config}]"])
@pytest.mark.model_testing_group(
    ModelTestingGroup.distributed,
)
def test_run_model_distributed(run_distributed_script, model_testing_config, run_test_script_base_path, request):
    import tests.models.distributed_test_model

    script = [
        "-m",
        tests.models.distributed_test_model.__name__,
        str(run_test_script_base_path),
        model_testing_config.name,
    ]
    if request.config.getoption("distributed_capture"):
        logger.warning(
            "Capturing output and forwarding to associated tests. Run with `--no-distributed-capture` to disable."
        )
    else:
        script.append("--no-distributed-capture")
    run_distributed_script(script, num_gpus=torch.cuda.device_count())


# We don't want to depend on `test_model_distributed` because we still want to run this in cas of failure.
# This should still run after `test_model_distributed`
@requires_cuda
@pytest.mark.depends_on(on=["test_model_simple[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
@pytest.mark.parametrize("config_name", list(DISTRIBUTED_TESTING_CONFIGS))
def test_model_distributed(
    run_test_script_for_all_models,
    compare_results_for_all_models,
    config_name,
    run_test_script_base_path,
    report_subtest,
    model_testing_config,
):
    config = DISTRIBUTED_TESTING_CONFIGS[config_name]
    if model_testing_config.should_skip(config):
        pytest.skip(f"Configuration not supported.")
    if torch.cuda.device_count() < config.num_gpus:
        pytest.skip(f"Not enough GPUs: {torch.cuda.device_count()} < {config.num_gpus}")
    report_subtest(run_test_script_base_path / config.name, config.num_gpus)
    if config.compare is not None:
        if not check_subtest_success(run_test_script_base_path / config.compare):
            pytest.fail(f"Test {config.compare} failed", pytrace=False)
        compare_results_for_all_models(config)
