import json
import logging
import math
import pathlib
import sys
import time
import traceback
import typing

import pytest
import torch

from fast_llm.core.distributed import ProcessGroup, allreduce_scalar, safe_barrier
from fast_llm.engine.base_model.base_model import BaseModel, Layer
from fast_llm.engine.config_utils.logging import configure_logging
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, StageConfig
from fast_llm.engine.multi_stage.stage import Stage
from fast_llm.utils import get_and_reset_memory_usage_mib, header

logger = logging.getLogger(__name__)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


TEST_RESULTS_PATH = pathlib.Path("/tmp/fast_llm_tests")


@pytest.fixture(scope="session")
def result_path():
    return TEST_RESULTS_PATH


def get_base_model(config: FastLLMModelConfig):
    # Create a base model (and distributed).
    # Using a full model config so we have the model type and distributed config in the same argument.
    distributed = Distributed(config.distributed)
    tensor_space = TensorSpace(config.distributed)
    config.base_model.setup_tensor_space(tensor_space)
    tensor_space.setup(distributed)
    base_model = config.get_model_class().base_model_class(config.base_model, config.distributed)
    base_model.setup(distributed)
    return base_model, distributed


def get_stage(base_model: BaseModel | list[Layer], distributed: Distributed):
    # Create a fast-llm stage which allocates and initializes meta tensors correctly.
    stage = Stage(
        config=StageConfig(),
        base_model=base_model,
        distributed_config=distributed.config,
        begin=0,
        end=1,
        index=0,
    )
    stage.setup(distributed=distributed)
    stage.initialize_weights()
    stage.restore_parameters()
    stage.reset_gradients()
    return stage


class DistributedSubtestContext:
    def __init__(
        self, base_path: pathlib.Path, name: str, group: ProcessGroup | None, num_gpus: int, enabled: bool = True
    ) -> None:
        self._path = base_path / name
        self._name = name
        self._group = group
        self._rank = 0 if group is None else group.rank()
        self._rank_enabled = self._rank < num_gpus
        self._enabled = enabled and self._rank_enabled
        self.success = False

    def __enter__(self) -> typing.Self:
        if self._enabled:
            self._sys_stdout = sys.stdout
            self._sys_stderr = sys.stderr
            self._path.mkdir(parents=True, exist_ok=True)
            sys.stdout = self._path.joinpath(f"pytest_stdout_{self._rank}").open("w")
            sys.stderr = self._path.joinpath(f"pytest_stderr_{self._rank}").open("w")
            # Logging is set to log to the old stdout, so we need to reconfigure.
            configure_logging()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._enabled:
            try:
                stdout_handle = sys.stdout
                stderr_handle = sys.stderr
                sys.stdout = self._sys_stdout
                sys.stderr = self._sys_stderr
                stdout_handle.close()
                stderr_handle.close()
            finally:
                configure_logging()

        if exc_type is None:
            self.success = True
        else:
            self._path.mkdir(parents=True, exist_ok=True)
            self._path.joinpath(f"pytest_traceback_{self._rank}").write_text(traceback.format_exc())

        if self._group is not None:
            # Barrier so `allreduce_scalar` doesn't go crazy in case of desync.
            safe_barrier(self._group, self._name)
            self.success = allreduce_scalar(self.success, dtype=torch.int64, group=self._group) == self._group.size()

        if self._rank_enabled:
            # Free resources to limit memory usage.
            report = get_and_reset_memory_usage_mib(clear_cache=True, global_stats=True, reset_global_stats=True)
            report["duration"] = time.perf_counter() - self._start

            json.dump(report, self._path.joinpath(f"pytest_report_{self._rank}").open("w"))

        logger.warning(f"{self._name} {"PASSED" if self.success else "FAILED"})")
        if self._rank == 0:
            set_subtest_success(self._path, self.success)

        return True


def set_subtest_success(path: pathlib.Path, success: bool = True):
    path.joinpath("pytest_success").write_text(str(int(success)))


def check_subtest_success(path: pathlib, fail: bool = True) -> bool:
    if not path.is_dir():
        if fail:
            pytest.fail(f"Test {path.name} did not run", pytrace=False)
        else:
            return False
    try:
        return bool(int(path.joinpath("pytest_success").read_text()))
    except OSError:
        return False


def format_resource_report(title: str, report: dict[str, float]) -> str:
    return "".join(
        [
            f"{title}:\n    ",
            f"Max Reserved: {report.get("max_reserved", math.nan):.0f} MiB",
            f"| Max Allocated: {report.get("max_allocated", math.nan):.0f} MiB".ljust(26),
            f"| End Reserved: {report.get("reserved", math.nan):.0f} MiB".ljust(25),
            f"| End Allocated: {report.get("allocated", math.nan):.0f} MiB".ljust(26),
            f"| Duration: {report.get("duration", math.nan):.2f}".ljust(18),
            f"| GPUs: {report["gpus"]:.0f}" if "gpus" in report else "",
        ]
    )


@pytest.fixture(scope="function")
def report_subtest(request: pytest.FixtureRequest):
    verbose = request.config.getoption("verbose")
    do_capture = request.config.getoption("distributed_capture")

    def do_report_subtest(path: pathlib.Path, world_size: int) -> None:
        success = check_subtest_success(path)
        if not do_capture:
            logger.warning("Distributed capture is disabled. See distributed test for run output.")
        elif verbose > 1 or not success:
            for rank in range(world_size):
                for fd, file_ in (("stdout", sys.stdout), ("stderr", sys.stdout), ("traceback", sys.stderr)):
                    print(header(f"{fd} rank {rank}", 80), file=file_)
                    file_path = path / f"pytest_{fd}_{rank}"
                    try:
                        print(file_path.read_text(), file=file_)
                    except OSError:
                        print(f"<<< not found {file_path}>>>", file=file_)
        else:
            print("Set verbose > 1 to show run output.")

        reports = {}
        for rank in range(world_size):
            try:
                reports[f"rank_{rank}"] = json.load(path.joinpath(f"pytest_report_{rank}").open("r"))
            except OSError:
                reports[rank] = {}
        keys = {key for report in reports.values() for key in report}
        report = {key: max(report[key] for report in reports.values() if key in report) for key in keys}
        report["gpus"] = world_size
        reports["global"] = report

        print(header(f"Resource usage", 80), file=sys.stderr)
        for name, report in reports.items():
            print(format_resource_report(name, report), file=sys.stderr)
        setattr(request.node, "fast_llm_resource_report", report)

        if not success:
            raise RuntimeError(f"test {path.name} failed")

    return do_report_subtest
