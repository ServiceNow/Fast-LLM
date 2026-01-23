import functools
import json
import logging
import math
import os
import pathlib
import sys
import time
import traceback
import typing

import pytest
import torch

from fast_llm.core.distributed import allreduce_scalar, safe_barrier
from fast_llm.engine.config_utils.logging import configure_logging
from fast_llm.engine.distributed.config import DistributedBackend, DistributedConfig
from fast_llm.engine.distributed.distributed import ProcessGroupPool
from fast_llm.utils import Assert, get_and_reset_memory_usage_mib, header

logger = logging.getLogger(__name__)


class DistributedTestContext:
    def __init__(
        self,
        do_capture: bool,
        timeout: float = 20.0,
        init_method: str = "env://",
        backend: DistributedBackend = DistributedBackend.nccl,
        use_cuda: bool = True,
    ) -> None:
        self._do_capture = do_capture
        self._timeout = timeout
        self._init_method = init_method
        self._backend = backend
        self._use_cuda = use_cuda

    def __enter__(self):
        if self._do_capture:
            logger.warning(
                "Capturing output and forwarding to associated tests. Run with `--no-distributed-capture` to disable."
            )

        self._pool = ProcessGroupPool(
            timeout=self._timeout, init_method=self._init_method, backend=self._backend, use_cuda=self._use_cuda
        ).__enter__()
        self._rank = self._pool.rank
        self._world_size = self._pool.world_size
        self._failures = []
        self._configure_logging()
        self._group = self._pool.get_process_group(range(self._world_size), self._rank)
        # TODO: Barriers needed?
        safe_barrier(self._group, "start", device=self._pool.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Final barrier to ensure everything is done before torchrun potentially kills workers.
        safe_barrier(self._group, "testing end", device=self._pool.device)
        # Let pytest know how things went.
        # These should already be reported above, we repeat for convenience.
        if self._failures:
            raise RuntimeError(f"The following subtests failed: {", ".join(self._failures)}")
        else:
            logger.warning("All tests passed")

    def subtest(self, base_path: pathlib.Path, name: str, num_gpus: int):
        return self.DistributedSubtestContext(self, base_path, name, num_gpus)

    def _configure_logging(self):
        configure_logging(rank=self._rank, world_size=self._world_size)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def group(self) -> torch.distributed.ProcessGroup:
        return self._group

    class DistributedSubtestContext:
        def __init__(
            self, test_context: "DistributedTestContext", base_path: pathlib.Path, name: str, num_gpus: int
        ) -> None:
            self._test_context = test_context
            self._path = base_path / name
            self._name = name
            self._num_gpus = num_gpus
            self._skip = self._test_context._world_size < self._num_gpus and self._test_context._use_cuda
            self._do_run = self._test_context._rank < num_gpus and not self._skip
            self._do_capture = self._test_context._do_capture and self._do_run
            self._success = False

        def __enter__(self) -> typing.Self:
            if self._do_capture:
                self._sys_stdout = sys.stdout
                self._sys_stderr = sys.stderr
                self._path.mkdir(parents=True, exist_ok=True)
                sys.stdout = self._path.joinpath(f"pytest_stdout_{self._test_context._rank}").open("w")
                sys.stderr = self._path.joinpath(f"pytest_stderr_{self._test_context._rank}").open("w")
                self._test_context._configure_logging()
                # Logging is set to log to the old stdout, so we need to reconfigure.
            self._start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._skip:
                # Skipped tests should exit right away.
                Assert.none(exc_val)
                logger.warning(
                    f"{self._name} {f"SKIPPED (not enough GPUs: {self._test_context._world_size} < {self._num_gpus})"})"
                )
                return

            if self._do_capture:
                try:
                    stdout_handle = sys.stdout
                    stderr_handle = sys.stderr
                    sys.stdout = self._sys_stdout
                    sys.stderr = self._sys_stderr
                    stdout_handle.close()
                    stderr_handle.close()
                finally:
                    assert DistributedConfig.default_world_size > 1
                    self._test_context._configure_logging()

            if exc_type is None:
                self._success = True
            else:
                self._path.mkdir(parents=True, exist_ok=True)
                self._path.joinpath(f"pytest_traceback_{self._test_context._rank}").write_text(traceback.format_exc())

            logger.warning(f"{self._name} done, waiting for other ranks ({"PASSED" if self._success else "FAILED"})")

            if (group := self._test_context._group) is not None:
                # Barrier so `allreduce_scalar` doesn't go crazy in case of desync.
                safe_barrier(group, self._name, device=self._test_context._pool.device)
                self._success = (
                    allreduce_scalar(
                        self._success, dtype=torch.int64, group=group, device=self._test_context._pool.device
                    )
                    == group.size()
                )

            if self._do_capture and torch.cuda.is_available():
                # Free resources to limit memory usage.
                report = get_and_reset_memory_usage_mib(clear_cache=True, global_stats=True, reset_global_stats=True)
                report["duration"] = time.perf_counter() - self._start

                json.dump(report, self._path.joinpath(f"pytest_report_{self._test_context._rank}").open("w"))

            if self._test_context._rank == 0:
                set_subtest_success(self._path, self._success)
                logger.warning(f"{self._name} {"PASSED" if self._success else "FAILED"}")
            if not self._success:
                self._test_context._failures.append(self._name)

            return True

        @property
        def do_run(self) -> bool:
            return self._do_run and not self._skip


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


def parallel_worker(
    rank: int,
    world_size: int,
    init_method: str,
    backend: DistributedBackend,
    do_capture: bool,
    use_cuda: bool,
    fn: typing.Callable,
    fn_args: typing.Sequence[typing.Any],
):
    DistributedConfig.default_rank = rank
    DistributedConfig.default_world_size = world_size
    DistributedConfig.default_local_world_size = world_size
    with DistributedTestContext(do_capture, 60, init_method, backend, use_cuda) as test_context:
        fn(test_context, *fn_args)


def do_run_parallel_script(
    fn: typing.Callable,
    fn_args: typing.Sequence[typing.Any],
    port: int,
    do_capture: bool,
    world_size: int,
    timeout: float = 240,
    backend: DistributedBackend = DistributedBackend.nccl,
    use_cuda: bool = True,  # Use CPU device in process group pool. May be used to disable device count check
):
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = "0"
    if do_capture:
        logger.warning(
            "Capturing output and forwarding to associated tests. Run with `--no-distributed-capture` to disable."
        )
    torch.multiprocessing.spawn(
        parallel_worker,
        args=(world_size, f"tcp://localhost:{port}", backend, do_capture, use_cuda, fn, fn_args),
        nprocs=world_size,
        join=False,
    ).join(timeout, grace_period=5)


@pytest.fixture(scope="session")
def run_parallel_script(worker_resources: "WorkerResources", request: pytest.FixtureRequest):
    return functools.partial(
        do_run_parallel_script,
        port=worker_resources.rendezvous_port,
        do_capture=request.config.getoption("distributed_capture"),
    )
