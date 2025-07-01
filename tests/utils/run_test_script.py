import argparse
import functools
import os
import pathlib
import subprocess
import sys
import typing

import pytest
import torch

from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.utils import Assert
from tests.utils.compare_tensor_logs import CompareConfig, compare_tensor_logs
from tests.utils.dataset import get_test_dataset
from tests.utils.model_configs import MODEL_CONFIGS, ModelTestingConfig

if typing.TYPE_CHECKING:
    from tests.conftest import WorkerResources

# FIXME: figure out correct import of megatron modules without this hack
sys.path.append(os.getcwd())

_ARTIFACT_PATH = "runs/0/artifacts"


def do_run_distributed_script(
    args: list[str],
    rendezvous_port: int,
    torchrun_port: int,
    num_gpus: int,
    timeout: float = 120,
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


def do_run_test_script(
    path: pathlib.Path,
    args: list[str],
    num_gpus: int = 1,
    *,
    model_type: str,
    is_megatron: bool = False,
    compare_path: pathlib.Path | None = None,
    config: CompareConfig | None = None,
    prepare_fn=None,
    compare_fn=None,
    do_compare: bool = True,
    rendezvous_port: int,
    torchrun_port: int,
):
    is_parallel = DistributedConfig.default_world_size > 1
    if is_parallel:
        Assert.eq(num_gpus, DistributedConfig.default_world_size)
    local_rank = DistributedConfig.default_rank

    if torch.cuda.device_count() < num_gpus:
        pytest.skip(f"Not enough GPUs to run test ({torch.cuda.device_count()}<{num_gpus})")
    env = os.environ.copy()
    if is_megatron:
        assert num_gpus == 1
        # Prevent Megatron from complaining.
        env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        env["NVTE_FLASH_ATTN"] = "0"
    else:
        env = None
    if local_rank == 0 and prepare_fn is not None:
        prepare_fn(path, None if compare_path is None else compare_path)
    if is_megatron:
        args = ["Megatron-LM/pretrain_gpt.py", *args, f"--structured-logs-dir={path}", f"--data-cache-path={path}"]
    else:
        args = ["--no-python", "fast-llm", "train", model_type, *args, f"run.experiment_dir={path}"]
    get_test_dataset()
    if (num_gpus == 1 or is_parallel) and not is_megatron:
        print(" ".join(args[1:]))
        RunnableConfig.parse_and_run(args[2:])
    else:
        do_run_distributed_script(
            args, rendezvous_port=rendezvous_port, torchrun_port=torchrun_port, num_gpus=num_gpus, env=env
        )
    if local_rank == 0 and compare_path is not None and do_compare:
        if compare_fn is not None:
            compare_fn(path, compare_path)
        compare_tensor_logs(
            compare_path / _ARTIFACT_PATH,
            path / _ARTIFACT_PATH,
            config,
        )


def do_run_test_script_for_all_models(
    extra_args: list[str],
    num_gpus: int = 1,
    *,
    is_megatron: bool = False,
    compare: str | None = None,
    config: CompareConfig | None = None,
    prepare_fn=None,
    compare_fn=None,
    do_compare: bool = True,
    rendezvous_port: int,
    torchrun_port: int,
    test_name: str,
    base_path: pathlib.Path,
    model_testing_config: ModelTestingConfig,
):
    do_run_test_script(
        base_path / test_name,
        (model_testing_config.megatron_args if is_megatron else model_testing_config.config_args) + extra_args,
        num_gpus,
        model_type=model_testing_config.model_type,
        is_megatron=is_megatron,
        compare_path=None if compare is None else base_path / compare,
        config=config,
        prepare_fn=prepare_fn,
        compare_fn=compare_fn,
        do_compare=do_compare,
        rendezvous_port=rendezvous_port,
        torchrun_port=torchrun_port,
    )


@pytest.fixture(scope="session")
def run_test_script(worker_resources: "WorkerResources"):
    return functools.partial(
        do_run_test_script,
        rendezvous_port=worker_resources.rendezvous_port,
        torchrun_port=worker_resources.torchrun_port,
    )


@pytest.fixture(scope="session")
def run_test_script_base_path(model_testing_config, result_path, request):
    return result_path / "models" / model_testing_config.name


@pytest.fixture(scope="function")
def run_test_script_for_all_models(
    worker_resources: "WorkerResources",
    run_test_script_base_path: pathlib.Path,
    model_testing_config: ModelTestingConfig,
    request: pytest.FixtureRequest,
):
    return functools.partial(
        do_run_test_script_for_all_models,
        rendezvous_port=worker_resources.rendezvous_port,
        torchrun_port=worker_resources.torchrun_port,
        test_name=request.node.originalname,
        base_path=run_test_script_base_path,
        model_testing_config=model_testing_config,
    )


def parse_run_distributed_script(args: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", type=pathlib.Path)
    parser.add_argument("model_testing_config", type=str)
    parsed = parser.parse_args(args)
    return parsed.base_path, MODEL_CONFIGS[parsed.model_testing_config]


@pytest.fixture(scope="session")
def run_distributed_script_for_all_models(
    worker_resources: "WorkerResources",
    run_test_script_base_path: pathlib.Path,
    model_testing_config: ModelTestingConfig,
    request: pytest.FixtureRequest,
):
    def do_run_distributed_script_for_all_models(args: list[str], num_gpus=2, base_path: pathlib.Path | None = None):
        do_run_distributed_script(
            args
            + [
                str(run_test_script_base_path if base_path is None else base_path),
                model_testing_config.name,
            ],
            worker_resources.rendezvous_port,
            worker_resources.torchrun_port,
            num_gpus,
        )

    return do_run_distributed_script_for_all_models
