import dataclasses
import json
import logging
import math
import os
import shutil

import pytest
import xdist.scheduler

from fast_llm.utils import get_and_reset_memory_usage_mib
from tests.utils.depends import DependencyManager
from tests.utils.global_variables import TEST_RESULTS_PATH, set_testing_global_variables

# TODO: Is this early enough?
set_testing_global_variables()  # isort: skip

import torch  # isort: skip

from tests.utils.save_load_configs import (  # isort: skip
    distributed_save_load_config,
    distributed_save_load_config_non_pp,
    get_convert_path,
)

# Make fixtures available globally without import
from tests.utils.run_test_script import (  # isort: skip
    compare_results_for_all_models,
    run_distributed_script,
    run_test_script_base_path,
    run_test_script_for_all_models,
)

from tests.utils.model_configs import model_testing_config, ModelTestingConfig, testing_group_enabled  # isort: skip
from tests.utils.utils import result_path, format_resource_report, report_subtest  # isort: skip

logger = logging.getLogger(__name__)

manager: DependencyManager | None = None


def pytest_addoption(parser):
    group = parser.getgroup("fast_llm")
    group.addoption("--skip-slow", action="store_true")
    group.addoption("--show-skipped", action="store_true")
    group.addoption("--show-gpu-memory", type=int, default=10)
    group.addoption("--no-distributed-capture", dest="distributed_capture", action="store_false")
    group.addoption("--models", nargs="*")
    group.addoption(
        "--run-extra-slow",
        action="store_true",
        default=False,
        help="Run tests marked as extra_slow",
    )
    group.addoption(
        "--show-dependencies",
        action="store_true",
        default=False,
        help="List all dependencies of all tests as a list of nodeids + the names that could not be resolved.",
    )


@dataclasses.dataclass
class WorkerResources:
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
    config.addinivalue_line("markers", "depends_on(name='name', on=['other_name']): marks dependencies between tests.")
    config.addinivalue_line("markers", "model_testing_group(group='group'): marks model testing group.")
    # TODO: Spawned processes (multi-gpu, Megatron) ignore resource allocation.
    is_parallel = hasattr(config, "workerinput")
    if is_parallel:
        worker_name = config.workerinput["workerid"]
        assert worker_name.startswith("gw")
        worker_id = int(worker_name[2:])
    else:
        worker_id = 0

    if TEST_RESULTS_PATH.exists():
        shutil.rmtree(TEST_RESULTS_PATH)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 0 and is_parallel:
        # We spread workers across GPUs.
        logger.warning(f"[Worker {worker_id}] Using GPUs {os.environ["CUDA_VISIBLE_DEVICES"]}")
    elif num_gpus > 0:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

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
        # Each worker needs its own set of ports for safe distributed run. Hopefully these are free.
        torchrun_port=TORCHRUN_DEFAULT_PORT + 2 * worker_id,
        rendezvous_port=TORCHRUN_DEFAULT_PORT + 2 * worker_id + 1,
    )

    # Skip slow autotune for tests. The default config has the highest block size, so this shouldn't hide any bug.
    os.environ["FAST_LLM_SKIP_TRITON_AUTOTUNE"] = "TRUE"


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items: list[pytest.Function]):
    global manager
    skip_slow = config.getoption("--skip-slow")
    skip_extra_slow = not config.getoption("--run-extra-slow")
    show_skipped = config.getoption("--show-skipped")

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

    new_items = []
    for item in items:
        if skip_slow and "slow" in item.keywords:
            if show_skipped:
                item.add_marker(pytest.mark.skip(reason="Skipping slow tests"))
            else:
                continue
        elif skip_extra_slow and "extra_slow" in item.keywords:
            if show_skipped:
                item.add_marker(pytest.mark.skip(reason="Skipping extra-slow tests"))
            else:
                continue
        elif not testing_group_enabled(item, skip_slow, skip_extra_slow, show_skipped):
            continue
        new_items.append(item)

    manager = DependencyManager(new_items)

    # Show the extra information if requested
    if config.getoption("show_dependencies"):
        manager.print_name_map(config.getoption("verbose") > 1)
        manager.print_processed_dependencies(config.getoption("color"))

    # Reorder the items so that tests run after their dependencies
    items[:] = manager.items

    # If pytest-depends is installed, it will complain about renamed nodes whether it's used or not.
    try:
        import pytest_depends.main
    except ImportError:
        pass
    else:
        old_clean_nodeid = pytest_depends.main.clean_nodeid
        # Hack into `clean_nodeid` so pytest_depends recognizes the renamed nodes.
        pytest_depends.main.clean_nodeid = lambda nodeid: old_clean_nodeid(nodeid.split("@dependency_group_")[0])


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Function, call: pytest.CallInfo):
    outcome = yield
    result = outcome.get_result()
    manager.register_result(item, result)

    # Measure GPU memory usage. (TODO: This excludes child processes)
    if call.when == "call" and torch.cuda.is_available():
        report = get_and_reset_memory_usage_mib(clear_cache=True, global_stats=True, reset_global_stats=True)
        report["duration"] = call.duration
        if hasattr(item, "fast_llm_resource_report"):
            report_ = getattr(item, "fast_llm_resource_report")
            report = {
                key: max(report[key] for report in (report, report_) if key in report)
                for key in set(report_) | set(report)
            }

        item.add_report_section(
            call.when,
            "resource usage",
            json.dumps(report),
        )


@pytest.hookimpl
def pytest_terminal_summary(terminalreporter):
    resource_reports = {}
    for reports in terminalreporter.stats.values():
        for report in reports:
            if isinstance(report, pytest.TestReport):
                for _, section in report.get_sections("Captured resource usage"):
                    if report.nodeid in resource_reports:
                        logging.error(f"Duplicate resource report for {report.nodeid}")
                    resource_reports[report.nodeid] = json.loads(section)

    if not resource_reports:
        return

    terminalreporter.write_sep("=", "Highest gpu memory usage", bold=True)
    sorted_nodeids = sorted(
        resource_reports.keys(),
        key=lambda nodeid: (
            resource_reports[nodeid]["max_reserved"] if "max_reserved" in resource_reports[nodeid] else 0
        ),
        reverse=True,
    )
    for nodeid in sorted_nodeids[: terminalreporter.config.getoption("--show-gpu-memory")]:
        terminalreporter.write_line(format_resource_report(nodeid, resource_reports[nodeid]))


def pytest_runtest_call(item: pytest.Function):
    if torch.cuda.is_available():
        # Empty cache to check is cuda is still working (TODO: Is there a better way? Can we kill the worker?)
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pytest.skip("Cuda runtime unavailable due to an error in an earlier test.")
    manager.handle_missing(item)


def pytest_unconfigure():
    global manager
    manager = None


@pytest.fixture(scope="session")
def worker_resources(request) -> WorkerResources:
    return request.config.worker_resources


@pytest.mark.trylast
def pytest_xdist_make_scheduler(config, log):
    # Always use grouped load balancing to handle dependencies, and make it work with `-n`.
    assert config.getvalue("dist") == "load"
    return xdist.scheduler.LoadGroupScheduling(config, log)
