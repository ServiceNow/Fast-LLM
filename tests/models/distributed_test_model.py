import logging

from fast_llm.cli import fast_llm_main_wrapper
from fast_llm.core.distributed import safe_barrier
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import ProcessGroupPool
from tests.utils.distributed_configs import DISTRIBUTED_TESTING_CONFIGS
from tests.utils.run_test_script import do_run_test_script_for_all_models, parse_run_distributed_script
from tests.utils.utils import DistributedSubtestContext

logger = logging.getLogger(__name__)


def main(args: list[str] | None = None) -> None:
    base_path, model_testing_config, do_capture = parse_run_distributed_script(args)

    if do_capture:
        logger.warning(
            "Capturing output and forwarding to associated tests. Run with `--no-distributed-capture` to disable."
        )

    # TODO: Why are barriers needed?
    with ProcessGroupPool(timeout=60) as pool:
        failures = []
        world_size = DistributedConfig.default_world_size
        rank = DistributedConfig.default_rank
        group = pool.get_process_group(range(world_size), rank)

        for name, config in DISTRIBUTED_TESTING_CONFIGS.items():
            if world_size < config.num_gpus:
                logger.warning(f"{name} {f"SKIPPED (not enough GPUs: {world_size} < {config.num_gpus})"})")
                continue
            with DistributedSubtestContext(base_path, name, group, config.num_gpus, enabled=do_capture) as subtest:
                if rank < config.num_gpus:
                    do_run_test_script_for_all_models(config, model_testing_config, base_path)
            if not subtest.success:
                failures.append(name)

        # Final barrier to ensure everything is done before torchrun potentially kills workers.
        safe_barrier(group, "testing end")
        # Let pytest know how things went.
        # These should already be reported above, we repeat for convenience.
        if failures:
            raise RuntimeError(f"The following subtests failed: {", ".join(failures)}")
        else:
            logger.warning("All tests passed")


if __name__ == "__main__":
    with fast_llm_main_wrapper():
        main()
