import atexit
import fcntl
import gc
import os
import pathlib
import random
import tempfile
import time
import typing
import warnings

import pytest
import torch.cuda
import yaml

from fast_llm.utils import Assert, get_free_port

# Make fixtures available globally without import
from tests.common import run_test_script, run_megatron_train, run_fast_llm_train, get_distributed_config  # isort: skip


def pytest_addoption(parser):
    parser.addoption("--skip-slow", action="store_true")
    parser.addoption(
        "--run-extra-slow",
        action="store_true",
        default=False,
        help="Run tests marked as extra_slow",
    )


_CUDA_CONTEXT_SIZE_GB = 0.7


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: Test is slow.")
    config.addinivalue_line(
        "markers", "extra_slow: Mark test as extra slow and skip unless --run-extra-slow is given."
    )
    config.addinivalue_line(
        "markers", "requires_cuda(num_gpus=None, gpu_memory_gb=None, ports=None, timeout=None): Use gpu lock"
    )

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    is_parallel = hasattr(config, "workerinput")
    if is_parallel:
        worker_name = config.workerinput["workerid"]
        assert worker_name.startswith("gw")
        worker_id = int(worker_name[2:])
    else:
        worker_id = 0
    if num_gpus > 0:
        gpu_memory_gb = torch.cuda.mem_get_info(0)[1] / 1e9
        if is_parallel:
            max_workers = int(gpu_memory_gb / _CUDA_CONTEXT_SIZE_GB / 2)
            num_workers = config.workerinput["workercount"]
            # Limit number of workers to avoid trouble.
            # Raise only in one worker to limit verboseness.
            # TODO: Raise in main process instead of worker?

            if num_workers > max_workers and worker_id == 0:
                raise ValueError(
                    "The cuda context of parallel test workers requires more than half of the available GPU memory."
                    f"Please reduce it to {max_workers} or less."
                )
            gpu_memory_gb -= _CUDA_CONTEXT_SIZE_GB * num_workers
    else:
        gpu_memory_gb = 0
    lock_path = pathlib.Path(tempfile.gettempdir()) / "pytest_xdist_locks.json" if is_parallel else None
    config.lock_adapter = GPULock(lock_path, num_gpus, gpu_memory_gb, worker_id)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="Skipping slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--run-extra-slow"):
        skip_extra_slow = pytest.mark.skip(reason="need --run-extra-slow option to run")
        for item in items:
            if "extra_slow" in item.keywords:
                item.add_marker(skip_extra_slow)


class GPULock:
    _MAX_PORT_RETRIES = 100

    def __init__(self, lock_path: pathlib.Path | None, num_gpus: int, gpu_memory_gb: float, worker_id: int):
        self._lock_path = lock_path
        self._file_handle = None
        self._num_gpus = num_gpus
        self._gpu_memory_gb = gpu_memory_gb
        self._owned = None
        self._worker_id = worker_id
        if self._lock_path is not None and self._worker_id == 0:
            # Create or reset the lock file.
            with self._lock_path.open("w") as f:
                yaml.dump(self._empty_lock, f)
        atexit.register(self.release)

    def _acquire_file_lock(self):
        """Acquire exclusive lock on the file."""
        if self._lock_path is None:
            # Lock is disabled, ignore.
            return
        if self._file_handle is None:
            # Raises OSError?
            self._file_handle = self._lock_path.open("r+")

        start_time = time.time()
        while True:
            try:
                fcntl.flock(self._file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._file_handle.seek(0)
                return
            except BlockingIOError:
                print("IUJEFNIEW", time.time() - start_time)
                if time.time() - start_time > 60:
                    raise
                time.sleep(0.01)

    def _release_file_lock(self) -> None:
        """Release exclusive lock on the file."""
        if self._file_handle is not None:
            try:
                fcntl.flock(self._file_handle, fcntl.LOCK_UN)
            except (AttributeError, OSError):
                pass

    @property
    def _empty_lock(self) -> dict[str, typing.Any]:

        return {"gpus": {i: 0 for i in range(self._num_gpus)}, "ports": [], "owned": {}}

    def _read_locks(self) -> dict[str, typing.Any]:
        """Read current locks states."""
        if self._lock_path is None or not self._lock_path.is_file():
            # Lock is disabled, ignore.
            return self._empty_lock
        self._file_handle.seek(0)
        content = self._file_handle.read()
        return yaml.safe_load(content) if content else {}

    def _write_locks(self, locks: dict[str, list[str]]) -> None:
        """Write locks state."""
        if self._lock_path is None:
            # Lock is disabled, ignore.
            return
        self._file_handle.seek(0)
        self._file_handle.truncate()
        yaml.dump(locks, self._file_handle)
        self._file_handle.flush()
        # Ensure physical write
        os.fsync(self._file_handle.fileno())

    def acquire(self, num_gpus: int, gpu_memory_gb: float, ports: int, timeout: float) -> dict[str, typing.Any]:
        if num_gpus > self._num_gpus or gpu_memory_gb > self._gpu_memory_gb:
            pytest.skip("Not enough GPUs.")
        if gpu_memory_gb > self._gpu_memory_gb:
            pytest.fail(
                f"Not enough GPU memory: needed = {gpu_memory_gb}, available = {self._gpu_memory_gb}."
                "If the available memory is too low, consider reducing the number of parallel tests."
            )

        end_time = time.time() + timeout

        while True:
            try:
                self._acquire_file_lock()
                locks = self._read_locks()
                Assert.none(self._owned)
                assert self._worker_id not in locks["owned"]

                owned = {} if self._owned is None else self._owned.copy()

                if num_gpus > 0:

                    # Try a random gpu first to spread the workload:
                    gpu_shift = random.randrange(self._num_gpus)
                    gpu_ids = []
                    for i in range(self._num_gpus):
                        index = (gpu_shift + i) % self._num_gpus
                        if locks["gpus"][index] + gpu_memory_gb <= self._gpu_memory_gb:
                            gpu_ids.append(i)
                            if len(gpu_ids) == num_gpus:
                                break

                    if len(gpu_ids) < num_gpus:
                        if time.time() > end_time:
                            available = {id: self._gpu_memory_gb - used for id, used in locks["gpus"]}
                            raise BlockingIOError(
                                f"Failed to acquire GPU resources."
                                f" Requested {num_gpus} GPUs with {gpu_memory_gb} GBs,"
                                f" available = {available}, used =  {locks["gpus"]}."
                            )
                        continue

                    for i in gpu_ids:
                        locks["gpus"][i] += gpu_memory_gb
                        # Enforce memory constraint.
                        # TODO: Make it work in subprocesses.
                        torch.cuda.set_per_process_memory_fraction(gpu_memory_gb / self._gpu_memory_gb, i)

                    torch.cuda.set_device(gpu_ids[0])

                    owned["gpus"] = {i: gpu_memory_gb for i in gpu_ids}

                if ports > 0:
                    # Get free ports, and lock them for other tests.

                    owned["ports"] = []

                    for _ in range(ports + self._MAX_PORT_RETRIES):
                        port = get_free_port()
                        if port not in locks["ports"]:
                            locks["ports"].append(port)
                            owned["ports"].append(port)
                            if len(owned["ports"]) == ports:
                                break
                    if len(owned["ports"]) != ports:
                        raise RuntimeError(f"Could not get {ports} ports.")

                locks["owned"][self._worker_id] = owned
                self._write_locks(locks)
                self._owned = owned
                return owned

            except OSError:
                if time.time() > end_time:
                    raise
                time.sleep(0.1)
            finally:
                self._release_file_lock()

        assert False

    def release(self) -> None:
        """Release lock with guaranteed cleanup."""
        if self._owned is not None:
            try:
                errors = []
                self._acquire_file_lock()
                locks = self._read_locks()
                if self._lock_path is not None:
                    Assert.eq(locks["owned"][self._worker_id], self._owned)
                if "gpus" in self._owned:
                    torch.cuda.set_device(0)
                    owned_gpus = set(self._owned["gpus"])
                    for i in range(self._num_gpus):
                        if i in owned_gpus:
                            locks["gpus"][i] -= self._owned["gpus"][i]
                            torch.cuda.set_per_process_memory_fraction(1.0, i)
                        elif (mem := torch.cuda.max_memory_reserved(i)) > 0:
                            errors.append(
                                f"Unexpected usage of gpu {i} ({mem / 1e9:.3f} GB reserved)."
                                " Make sure to request resources through the `get_test_resources` fixture or mark."
                            )
                    # Make sure the memory is released for other processes.
                    if not self._clear_cuda_cache(self._owned["gpus"].keys()):
                        errors.append("Failed to clear GPU cache.")
                if "ports" in self._owned:
                    for port in self._owned.get("ports", []):
                        if port in locks["ports"]:
                            locks["ports"].remove(port)
                        else:
                            warnings.warn(f"Port {port} not in locks.")
                if self._lock_path is not None:
                    del locks["owned"][self._worker_id]
                self._write_locks(locks)
                self._owned = None
                if errors:
                    print(torch.cuda.memory_summary())
                    pytest.fail("\n".join(errors))
            finally:
                self._release_file_lock()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.release()

    def prepare(self):
        # Prevent failures in cleanup and/or elsewhere caused by other tests.
        self._skip = False
        if not self._clear_cuda_cache():
            self._skip = True
            warnings.warn(msg := "Skipping test due to improper initial cuda state.")
            pytest.skip(msg)
        for i in range(self._num_gpus):
            torch.cuda.set_per_process_memory_fraction(0.0, i)

    def cleanup(self):
        self._release_file_lock()
        # Prevent unexpected gpu memory usage.
        if self._skip:
            #  We're already skipping, we don't want to fail.
            return
        for i in range(self._num_gpus):
            if (mem := torch.cuda.max_memory_reserved(i)) > 0:
                pytest.fail(
                    f"Unexpected usage of gpu {i} ({mem/1e9:.3f} GB reserved)."
                    " Make sure to request resources through the `get_test_resources` fixture or mark."
                )

    def _clear_cuda_cache(self, devices: typing.Sequence[int] | None = None, _collected: bool = False) -> bool:
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        for i in range(self._num_gpus) if devices is None else devices:
            torch.cuda.reset_peak_memory_stats(i)
            if torch.cuda.max_memory_reserved(i) > 0:
                if _collected:
                    return False
                else:
                    # Garbage collection is slow, only do if necessary.
                    gc.collect()
                    for obj in gc.get_objects():
                        if isinstance(obj, torch.Tensor):  # or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                            del obj
                    gc.collect()

                    return self._clear_cuda_cache(devices, True)

        return True

    @property
    def owned(self):
        if self._owned is None:
            raise ValueError("No resources requested")
        return self._owned


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    assert item.config.lock_adapter._owned is None
    item.config.lock_adapter.prepare()
    lock_marker = item.get_closest_marker("get_test_resources")
    if lock_marker is not None:
        Assert.leq(lock_marker.kwargs.keys(), {"num_gpus", "gpu_memory_gb", "ports", "timeout"})
        item.config.lock_adapter.acquire(
            num_gpus=int(lock_marker.kwargs.get("num_gpus", 1)),
            gpu_memory_gb=float(lock_marker.kwargs.get("gpu_memory_gb", 5)),
            ports=int(lock_marker.kwargs.get("ports", 0)),
            timeout=float(lock_marker.kwargs.get("timeout", 10)),
        )
    yield


@pytest.fixture(scope="session")
def get_test_resources(request):
    def _get_test_resources():
        return request.config.lock_adapter.owned

    return _get_test_resources


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
    yield
    item.config.lock_adapter.release()
    # Make sure all tests that use gpu request it.
    item.config.lock_adapter.cleanup()
