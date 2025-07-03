import pytest
import torch

from tests.utils.distributed_configs import (
    DISTRIBUTED_TESTING_CONFIGS,
    SIMPLE_TESTING_CONFIG,
    SINGLE_GPU_TESTING_CONFIGS,
)
from tests.utils.model_configs import ModelTestingGroup
from tests.utils.run_test_script import ARTIFACT_PATH
from tests.utils.utils import report_subtest


@pytest.mark.model_testing_group(ModelTestingGroup.basic)
def test_model_simple(run_test_script_for_all_models):
    # A simple config to prevent unnecessary testing and creation of dependency group
    run_test_script_for_all_models(SIMPLE_TESTING_CONFIG)


@pytest.mark.depends_on(on=["test_model_simple[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.basic)
# Parametrize with config name so it shows in test name.
@pytest.mark.parametrize("config_name", SINGLE_GPU_TESTING_CONFIGS)
def test_and_compare_model(
    run_test_script_for_all_models, compare_results_for_all_models, config_name, run_test_script_base_path
):
    # We can expect tests to respect the ordering of `SINGLE_GPU_TESTING_CONFIGS`, so compare should have run already.
    config = SINGLE_GPU_TESTING_CONFIGS[config_name]
    if config.compare is not None:
        for artifact in ["init", "train_1"]:
            path = run_test_script_base_path / config.compare / ARTIFACT_PATH / "0" / f"tensor_logs_{artifact}.pt"
            if not path.is_file():
                # Dependency likely failed, skipping this test because it will most likely fail for the same reason.
                # We still need to fail because we can't confirm the failure.
                pytest.fail(f"Compared test {config.compare} failed or did not run ({path} not found).", pytrace=False)
    # A baseline config (single-gpu, bf16, flash-attn).
    # Also tests for multiple data loaders.
    run_test_script_for_all_models(config)

    if config.compare is not None:
        compare_results_for_all_models(config)


@pytest.mark.depends_on(on=["test_model_simple[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_run_model_distributed(run_distributed_script, model_testing_config, run_test_script_base_path):
    import tests.models.distributed_test_model

    run_distributed_script(
        [
            tests.models.distributed_test_model.__file__,
            str(run_test_script_base_path),
            model_testing_config.name,
        ],
        num_gpus=torch.cuda.device_count(),
    )


# We don't want to depend on `test_model_distributed` because we still want to run this in cas of failure.
# This should still run after `test_model_distributed`
@pytest.mark.depends_on(on=["test_model_simple[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
@pytest.mark.parametrize("config_name", list(DISTRIBUTED_TESTING_CONFIGS)[:1])
def test_model_distributed(
    run_test_script_for_all_models, compare_results_for_all_models, config_name, run_test_script_base_path
):
    config = DISTRIBUTED_TESTING_CONFIGS[config_name]
    report_subtest(run_test_script_base_path / config.name, config.num_gpus)
    if config.compare is not None:
        for artifact in ["init", "train_1"]:
            if not (
                run_test_script_base_path / config.compare / ARTIFACT_PATH / f"tensor_logs_{artifact}.pt"
            ).is_file():
                pytest.fail(f"Compared test {config.compare} failed or did not run.", pytrace=False)
        compare_results_for_all_models(config)
