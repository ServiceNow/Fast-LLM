import os
import shutil
import subprocess

import pytest
import torch

from fast_llm.engine.config_utils.runnable import RunnableConfig
from tests.utils.compare_tensor_logs import CompareConfig, compare_tensor_logs
from tests.utils.dataset import get_test_dataset
from tests.utils.model_configs import TEST_MODEL_TYPE
from tests.utils.utils import TEST_RESULTS_PATH

FORCE_REUSE_RESULTS = int(os.environ.get("FORCE_REUSE_RESULTS", 0)) != 0
REUSE_RESULTS = FORCE_REUSE_RESULTS or int(os.environ.get("REUSE_RESULTS", 0)) != 0
ARTIFACT_PATH = "runs/0/artifacts"


@pytest.fixture(scope="session")
def run_test_script(worker_resources):
    def do_run_test_script(
        name: str,
        script: list[str],
        num_gpus: int = 1,
        *,
        model_type: str = TEST_MODEL_TYPE,
        is_megatron: bool = False,
        compare: str | None = None,
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
        path = TEST_RESULTS_PATH / name
        skip = False
        artifact_path = path / ARTIFACT_PATH
        if path.exists():
            assert path.is_dir()
            # TODO: Better way to check if the previous attempt succeeded.
            if (
                REUSE_RESULTS
                and artifact_path.is_dir()
                and len(list((artifact_path / "0").iterdir())) >= (1 if is_megatron else 3)
            ):
                skip = True
            elif FORCE_REUSE_RESULTS:
                raise RuntimeError(artifact_path)
            else:
                shutil.rmtree(path)
        elif FORCE_REUSE_RESULTS:
            raise RuntimeError(path)
        if prepare_fn is not None:
            skip = prepare_fn(TEST_RESULTS_PATH / name, None if compare is None else TEST_RESULTS_PATH / compare, skip)
        if is_megatron:
            script = [*script, f"--structured-logs-dir={path}", f"--data-cache-path={path}"]
        else:
            script = ["train", model_type, *script, f"run.experiment_dir={path}"]
        header = ["Megatron-LM/pretrain_gpt.py"] if is_megatron else ["--no-python", "fast-llm", "train"]
        command = [
            "python",
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={num_gpus}",
            f"--rdzv-endpoint=localhost:{worker_resources.rendezvous_port}",
            f"--master-port={worker_resources.torchrun_port}",
            *header,
            *script,
        ]
        print(" ".join(command))
        if skip:
            print("Reusing existing run.")
        else:
            get_test_dataset()
            if num_gpus == 1 and not is_megatron:
                RunnableConfig.parse_and_run(script)
            else:
                completed_proc = subprocess.run(command, env=env, timeout=60)
                if completed_proc.returncode:
                    raise RuntimeError(f"Process failed with return code {completed_proc.returncode}")
        if compare and do_compare:
            if compare_fn is not None:
                compare_fn(TEST_RESULTS_PATH / name, TEST_RESULTS_PATH / compare)
            compare_tensor_logs(
                TEST_RESULTS_PATH / compare / ARTIFACT_PATH,
                TEST_RESULTS_PATH / name / ARTIFACT_PATH,
                config,
            )

    return do_run_test_script
