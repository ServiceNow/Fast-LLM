import dataclasses
import math
import os

import networkx
import pytest
import pytest_depends
import pytest_depends.main
import torch
from xdist.scheduler import LoadGroupScheduling

# Make fixtures available globally without import
from tests.common import run_test_script  # isort: skip


def pytest_addoption(parser):
    parser.addoption("--skip-slow", action="store_true")
    parser.addoption(
        "--run-extra-slow",
        action="store_true",
        default=False,
        help="Run tests marked as extra_slow",
    )


@dataclasses.dataclass
class WorkerResources:
    worker_id: int
    gpu_id: int | None
    num_gpus: int
    torchrun_port: int
    rendezvous_port: int


MAX_TEST_MEMORY = 5e9
CUDA_CONTEXT_SIZE = 7e8
TORCHRUN_DEFAULT_PORT = 25900


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: Test is slow.")
    config.addinivalue_line(
        "markers", "extra_slow: Mark test as extra slow and skip unless --run-extra-slow is given."
    )
    # TODO: Spawned processes (multi-gpu, Megatron) ignore resource allocation.
    is_parallel = hasattr(config, "workerinput")
    if is_parallel:
        worker_name = config.workerinput["workerid"]
        assert worker_name.startswith("gw")
        worker_id = int(worker_name[2:])
    else:
        worker_id = 0

    num_gpus = torch.cuda.device_count()
    if num_gpus > 0 and is_parallel:
        # We spread workers across GPUs.
        gpu_id = worker_id % num_gpus
        # We set the device through "CUDA_VISIBLE_DEVICES", and this needs to happen before cuda initialization.
        # The `device_count` call above doesn't initialize, but `mem_get_info` below does.
        assert not torch.cuda.is_initialized()
        # TODO: Support this?
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str((gpu_id + i) % num_gpus) for i in range(num_gpus))
    elif num_gpus > 0:
        gpu_id = 0
    else:
        gpu_id = None

    gpu_memory = torch.cuda.mem_get_info(0)[1] if num_gpus > 0 else 0
    if num_gpus > 0:
        torch.cuda.set_per_process_memory_fraction(MAX_TEST_MEMORY / gpu_memory, 0)

    num_workers = config.workerinput["workercount"] if is_parallel else 1
    if num_gpus > 0:
        memory_needed = (MAX_TEST_MEMORY + CUDA_CONTEXT_SIZE) * math.ceil(num_workers / num_gpus)
        if memory_needed > gpu_memory:
            raise ValueError(
                f"Not enough GPU memory to support this many parallel workers {num_workers}."
                f"Please reduce the number of workers to {int(gpu_memory/(MAX_TEST_MEMORY + CUDA_CONTEXT_SIZE))*num_gpus} or less."
            )

    config.worker_resources = WorkerResources(
        worker_id=worker_id,
        gpu_id=gpu_id,
        num_gpus=num_gpus,
        # Each worker needs its own set of ports for safe distributed run. Hopefully these are free.
        torchrun_port=TORCHRUN_DEFAULT_PORT + 2 * worker_id,
        rendezvous_port=TORCHRUN_DEFAULT_PORT + 2 * worker_id + 1,
    )


@pytest.hookimpl(trylast=True)
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

    manager: pytest_depends.DependencyManager = pytest_depends.managers[-1]
    # Build the undirected graph as in `DependencyManager.sorted_items`.
    dag = networkx.DiGraph()
    for item in manager.items:
        node_id = pytest_depends.clean_nodeid(item.nodeid)
        dag.add_node(node_id)
        for dependency in manager.dependencies[node_id].dependencies:
            dag.add_edge(dependency, node_id)
    # Mark dependency groups for xdist.
    manager.groups = {}
    for i, node_ids in enumerate(sorted(networkx.weakly_connected_components(dag), key=len, reverse=True)):
        if len(node_ids) > 1:
            for node_id in node_ids:
                manager.nodeid_to_item[node_id]._nodeid = (
                    f"{manager.nodeid_to_item[node_id]._nodeid}@dependency_group_{i}"
                )

    old_clean_nodeid = pytest_depends.main.clean_nodeid
    # Hack into `clean_nodeid` so pytest_depends recognizes the renamed nodes.
    pytest_depends.main.clean_nodeid = lambda nodeid: old_clean_nodeid(nodeid.split("@dependency_group_")[0])


@pytest.fixture(scope="session")
def worker_resources(request) -> WorkerResources:
    return request.config.worker_resources


@pytest.mark.trylast
def pytest_xdist_make_scheduler(config, log):
    # Always use grouped load balancing to handle dependencies, and make it work with `-n`.
    assert config.getvalue("dist") == "load"
    return LoadGroupScheduling(config, log)
