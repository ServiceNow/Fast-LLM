import contextlib
import logging
import sys
import traceback

from fast_llm.config import ValidationError
from fast_llm.engine.config_utils.logging import configure_logging
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.engine.config_utils.runnable import RunnableConfig

# Import these submodules to ensure classes are added to the dynamic class registry.
import fast_llm.data.auto  # isort: skip
import fast_llm.engine.checkpoint.convert  # isort: skip
import fast_llm.models.auto  # isort: skip

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def fast_llm_main_wrapper():
    # (Pre-)configure logging
    configure_logging()
    try:
        yield
    except Exception as e:
        if sys.gettrace():
            raise
        if isinstance(e, ValidationError):
            log_main_rank(traceback.format_exc(), log_fn=logger.error)
        else:
            logger.critical(traceback.format_exc())
        sys.exit(1)


def fast_llm_main(args: list[str] | None = None):
    # TODO: Add hook to register model classes? (environment variable?)
    import sys

    if args is None:
        args = sys.argv[1:]
    if args and args[0] == "create-dataset":
        # Call the custom dataset creation script from the installed package
        import runpy

        runpy.run_module("fast_llm.diffullama.create_dataset_parallel", run_name="__main__")
        return
    if args and args[0] == "train-diffullama":
        # Launch train_updated.py using accelerate with recommended arguments
        import os
        import subprocess

        # Remove the first argument ("train-update") and pass the rest to train_updated.py
        train_args = args[1:]
        # You can customize config path, master addr/port, etc. here or via environment variables
        accelerate_cmd = [
            "accelerate",
            "launch",
            "--config_file",
            "/app/fast_llm/diffullama/accelerate_configs/single_node.yaml",
            # "fast_llm/diffullama/accelerate_configs/single_node.yaml",
            "--main_process_ip",
            os.environ.get("MASTER_ADDR", "localhost"),
            "--main_process_port",
            os.environ.get("MASTER_PORT", "20000"),
            "--machine_rank",
            os.environ.get("NODEID", "0"),
            "--num_processes",
            os.environ.get("WORLD_SIZE", "8"),
            "--num_machines",
            os.environ.get("NNODES", "1"),
            # "fast_llm/diffullama/train_updated.py",
            "/app/fast_llm/diffullama/train_updated.py",
        ] + train_args
        subprocess.run(accelerate_cmd, check=True)
        return
    with fast_llm_main_wrapper():
        RunnableConfig.parse_and_run(args)


if __name__ == "__main__":
    fast_llm_main()
