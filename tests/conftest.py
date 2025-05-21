import atexit
import contextlib
import fcntl
import json
import logging
import os
import pathlib
import random
import tempfile
import time
import typing

import pytest
import torch.cuda

from fast_llm.utils import get_free_port

# Make fixtures available globally without import
from tests.common import run_test_script, run_megatron_train, run_fast_llm_train  # isort: skip


def pytest_addoption(parser):
    parser.addoption("--skip-slow", action="store_true")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: Test is slow.")
    config.addinivalue_line("markers", "xdist_lock(num_gpus=None, gpu_memory_gb=None, timeout=None): Use gpu lock")

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    config.lock_adapter = GPULock(
        pathlib.Path(tempfile.gettempdir()) / "pytest_xdist_locks.json" if hasattr(config, "workerinput") else None,
        num_gpus,
        torch.cuda.mem_get_info(0)[1] / 1e9 if num_gpus > 0 else 0,
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="Skipping slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


class GPULock:
    _MAX_PORT_RETRIES = 100

    def __init__(self, lock_file: pathlib.Path | None, num_gpus: int, gpu_memory_gb: float):
        self._lock_file = lock_file
        self._file_handle = None
        self._owned_keys: set[str] = set()
        self._num_gpus = num_gpus
        self._gpu_memory_gb = gpu_memory_gb
        if self._lock_file is not None:
            # Reset the lock if set in previous run.
            try:
                assert self._acquire_file_lock()
                self._write_locks({})
            finally:
                self._release_file_lock()
        atexit.register(self._release_all_locks)

    def _acquire_file_lock(self) -> bool:
        """Acquire exclusive lock on the file."""
        if self._lock_file is None:
            # Lock is disabled, ignore.
            return True
        if self._file_handle is None:
            try:
                self._file_handle = open(self._lock_file, "r+")
            except OSError:
                time.sleep(0.1)
                return False

        start_time = time.time()
        while time.time() - start_time < 5:
            try:
                fcntl.flock(self._file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._file_handle.seek(0)
                return True
            except BlockingIOError:
                time.sleep(0.1)
        return False

    def _release_file_lock(self) -> None:
        """Release exclusive lock on the file."""
        if self._file_handle:
            try:
                fcntl.flock(self._file_handle, fcntl.LOCK_UN)
            except (AttributeError, OSError):
                pass

    def _read_locks(self) -> dict[str, typing.Any]:
        """Read current locks states."""
        if self._lock_file is None or not self._lock_file.is_file():
            # Lock is disabled, ignore.
            return {}
        self._file_handle.seek(0)
        content = self._file_handle.read()
        return json.loads(content) if content else {}

    def _write_locks(self, locks: dict[str, list[str]]) -> None:
        """Write locks state."""
        if self._lock_file is None:
            # Lock is disabled, ignore.
            return
        self._file_handle.seek(0)
        self._file_handle.truncate()
        json.dump(
            {lock_key: list(locked_resources) for lock_key, locked_resources in locks.items()}, self._file_handle
        )
        self._file_handle.flush()
        os.fsync(self._file_handle.fileno())  # Ensure physical write

    def acquire(
        self, test_id: str, num_gpus: int, gpu_memory_gb: float, ports: int = 0, timeout: float = 60
    ) -> dict[str, typing.Any]:
        if num_gpus > self._num_gpus or gpu_memory_gb > self._gpu_memory_gb:
            pytest.skip("GPU resources unavailable.")
        end_time = time.time() + timeout

        while time.time() < end_time:
            if not self._acquire_file_lock():
                time.sleep(0.1)
                continue

            locks = self._read_locks()

            owned = {}

            if num_gpus > 0:
                if "gpus" not in locks:
                    locks["gpus"] = {i: 0 for i in range(self._num_gpus)}

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
                    self._release_file_lock()
                    time.sleep(1)
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
                if "ports" not in locks:
                    locks["ports"] = []

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

            if "owners" not in locks:
                locks["owners"] = {}
            assert test_id not in locks["owners"]

            self._write_locks(locks)
            self._owned_keys.add(test_id)
            self._release_file_lock()
            return owned
        self._release_file_lock()
        raise RuntimeError("Failed to acquire resource lock.")

    def _release(self, locks: dict[str, typing.Any], test_id: str) -> None:
        if "owners" in locks and test_id in locks["owners"]:
            owned = locks["owners"][test_id]
            if "gpus" in owned:
                for i, memory in owned["gpus"].items():
                    locks["gpu"][i] -= memory
                    torch.cuda.set_per_process_memory_fraction(1.0, i)
                torch.cuda.set_device(0)
                # Make sure the memory is released for other processes.
                torch.cuda.empty_cache()
            if "ports" in owned:
                for port in locks["owners"][test_id].get("ports", []):
                    locks["ports"].remove(port)
            del locks["owners"][test_id]
            self._write_locks(locks)

    def release(self, test_id: str) -> None:
        """Release lock with guaranteed cleanup."""
        if not self._acquire_file_lock():
            raise BlockingIOError(f"Unable to flock file: {self._lock_file}")
        try:
            locks = self._read_locks()
            self._release(locks, test_id)
            self._owned_keys.discard(test_id)
        finally:
            self._release_file_lock()

    def _release_all_locks(self) -> None:
        """Clean up file handles and locks."""
        if getattr(self, "_is_cleaned", False) or not self._owned_keys:
            return

        try:
            if not self._acquire_file_lock():
                logging.warning("Cleanup failed - could not acquire lock")
                return
            try:
                locks = self._read_locks()
                for test_id in list(self._owned_keys):
                    self._release(locks, test_id)
                self._write_locks(locks)
                self._owned_keys.clear()
                self._is_cleaned = True
            finally:
                self._release_file_lock()
        finally:
            if self._file_handle:
                try:
                    self._file_handle.close()
                except Exception:
                    pass
                self._file_handle = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self._release_all_locks()


def _normalize_test_id(item, name: str) -> str:
    if "::" in name:
        return name.replace(".py", "").replace("/", ".")
    module_name = getattr(item, "module", None)
    if module_name is None:
        return f"module::{name}"
    return f"{module_name.__name__}::{name}"


@pytest.fixture(scope="session")
def get_test_resources(request):
    """Context manager for acquiring distributed locks in pytest-xdist."""

    @contextlib.contextmanager
    def create_lock(num_gpus: int = 1, gpu_memory_gb: float = 5, ports: int = 0, timeout: float = 60):
        # 5 GiBs should be more than enough for nearly all tests.
        test_id = _normalize_test_id(request.node, request.node.nodeid)
        owned = request.config.lock_adapter.acquire(
            test_id=test_id,
            num_gpus=num_gpus,
            gpu_memory_gb=gpu_memory_gb,
            ports=ports,
            timeout=timeout,
        )
        try:
            # TODO: Actually use these gpus.
            yield owned
        finally:
            request.config.lock_adapter.release(test_id)

    return create_lock
