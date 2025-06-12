import os
import pathlib
import shutil
import subprocess
import sys

import pytest
import torch

from fast_llm.engine.config_utils.runnable import RunnableConfig
from tests.utils.compare_tensor_logs import CompareConfig, compare_tensor_logs
from tests.utils.dataset import get_test_dataset

# FIXME: figure out correct import of megatron modules without this hack
sys.path.append(os.getcwd())

_ARTIFACT_PATH = "runs/0/artifacts"


@pytest.fixture(scope="session")
def run_test_script(worker_resources):
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
    ):
        if torch.cuda.device_count() < num_gpus:
            pytest.skip(f"Not enough GPUs to run test ({torch.cuda.device_count()}<{num_gpus})")
        env = os.environ.copy()
        if is_megatron:
            # Prevent Megatron from complaining.
            env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            env["NVTE_FLASH_ATTN"] = "0"
        skip = False
        if path.exists():
            assert path.is_dir()
            # TODO: Better way to check if the previous attempt succeeded.
            shutil.rmtree(path)
        if prepare_fn is not None:
            skip = prepare_fn(path, None if compare_path is None else compare_path, skip)
        if is_megatron:
            args = [*args, f"--structured-logs-dir={path}", f"--data-cache-path={path}"]
        else:
            args = ["train", model_type, *args, f"run.experiment_dir={path}"]
        header = ["Megatron-LM/pretrain_gpt.py"] if is_megatron else ["--no-python", "fast-llm", "train"]
        command = [
            "python",
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={num_gpus}",
            f"--rdzv-endpoint=localhost:{worker_resources.rendezvous_port}",
            f"--master-port={worker_resources.torchrun_port}",
            *header,
            *args,
        ]
        print(" ".join(command))
        if skip:
            print("Reusing existing run.")
        else:
            get_test_dataset()
            if num_gpus == 1 and not is_megatron:
                RunnableConfig.parse_and_run(args)
            else:
                completed_proc = subprocess.run(command, env=env, timeout=120)
                if completed_proc.returncode:
                    raise RuntimeError(f"Process failed with return code {completed_proc.returncode}")
        if compare_path is not None and do_compare:
            if compare_fn is not None:
                compare_fn(path, compare_path)
            compare_tensor_logs(
                compare_path / _ARTIFACT_PATH,
                path / _ARTIFACT_PATH,
                config,
            )

    return do_run_test_script


@pytest.fixture(scope="session")
def run_test_script_base_path(model_testing_config, result_path, request):
    return result_path / "models" / model_testing_config.name


@pytest.fixture(scope="function")
def run_test_script_for_all_models(run_test_script, run_test_script_base_path, model_testing_config, request):
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
    ):
        run_test_script(
            run_test_script_base_path / request.node.originalname,
            (model_testing_config.megatron_args if is_megatron else model_testing_config.config_args) + extra_args,
            num_gpus,
            model_type=model_testing_config.model_type,
            is_megatron=is_megatron,
            compare_path=None if compare is None else run_test_script_base_path / compare,
            config=config,
            prepare_fn=prepare_fn,
            compare_fn=compare_fn,
            do_compare=do_compare,
        )

    return do_run_test_script_for_all_models
