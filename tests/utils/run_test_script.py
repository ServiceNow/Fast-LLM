import argparse
import functools
import os
import pathlib
import pprint
import subprocess
import sys
import typing

import pytest

from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.utils import Assert
from tests.utils.dataset import get_model_test_dataset
from tests.utils.distributed_configs import DistributedTestingConfig
from tests.utils.model_configs import MODEL_CONFIGS, ModelTestingConfig

if typing.TYPE_CHECKING:
    from tests.conftest import WorkerResources

# FIXME: figure out correct import of megatron modules without this hack
sys.path.append(os.getcwd())

ARTIFACT_PATH = "runs/0/artifacts"


def do_run_distributed_script(
    args: list[str],
    rendezvous_port: int,
    torchrun_port: int,
    num_gpus: int,
    timeout: float = 240,
    env: dict[str, str | None] = None,
):
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={num_gpus}",
        f"--rdzv-endpoint=localhost:{rendezvous_port}",
        f"--master-port={torchrun_port}",
        *args,
    ]
    print(" ".join(command))
    completed_proc = subprocess.run(command, timeout=timeout, env=env)
    if completed_proc.returncode:
        raise RuntimeError(f"Process failed with return code {completed_proc.returncode}")


@pytest.fixture(scope="session")
def run_distributed_script(
    worker_resources: "WorkerResources",
    run_test_script_base_path: pathlib.Path,
    model_testing_config: ModelTestingConfig,
):
    return functools.partial(
        do_run_distributed_script,
        rendezvous_port=worker_resources.rendezvous_port,
        torchrun_port=worker_resources.torchrun_port,
    )


@pytest.fixture(scope="session")
def run_test_script_base_path(model_testing_config, result_path, request):
    return result_path / "models" / model_testing_config.name


def do_run_test_script_for_all_models(
    distributed_testing_config: DistributedTestingConfig,
    model_testing_config: ModelTestingConfig,
    base_path: pathlib.Path,
    runnable_type: str = "train",
):
    Assert.leq(distributed_testing_config.num_gpus, DistributedConfig.default_world_size)
    get_model_test_dataset()
    args = [
        "fast-llm",
        runnable_type,
        model_testing_config.model_type,
        *model_testing_config.config_args,
        *distributed_testing_config.config_args,
        f"model.distributed.world_size={distributed_testing_config.num_gpus}",
        f"model.distributed.local_world_size={distributed_testing_config.num_gpus}",
        f"run.experiment_dir={base_path/distributed_testing_config.name}",
    ]
    print(" ".join(args))
    if runnable_type == "train":
        model_testing_config.trainer_config_class.parse_and_run(args[3:])
    elif runnable_type == "evaluate":
        model_testing_config.evaluators_config_class.parse_and_run(args[2:])
    else:
        raise ValueError(f"Unknown runnable_type {runnable_type}")


@pytest.fixture(scope="function")
def run_test_script_for_all_models(
    run_test_script_base_path: pathlib.Path,
    model_testing_config: ModelTestingConfig,
):
    return functools.partial(
        do_run_test_script_for_all_models,
        base_path=run_test_script_base_path,
        model_testing_config=model_testing_config,
    )


def parse_run_distributed_script(args: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", type=pathlib.Path)
    parser.add_argument("model_testing_config", type=str)
    parser.add_argument("--no-distributed-capture", dest="distributed_capture", action="store_false")

    parsed = parser.parse_args(args)
    return parsed.base_path, MODEL_CONFIGS[parsed.model_testing_config], parsed.distributed_capture


@pytest.fixture(scope="session")
def compare_results_for_all_models(
    worker_resources: "WorkerResources",
    run_test_script_base_path: pathlib.Path,
    model_testing_config: ModelTestingConfig,
):
    def do_compare_results_for_all_models(config: DistributedTestingConfig):
        assert config.compare is not None
        compare_config = config.compare_config.rescale(config.compare_factor * model_testing_config.compare_factor)
        pprint.pprint(compare_config)
        compare_config.compare_tensor_logs(
            run_test_script_base_path / config.compare / ARTIFACT_PATH,
            run_test_script_base_path / config.name / ARTIFACT_PATH,
        )

    return do_compare_results_for_all_models
