import logging

import torch

from fast_llm.cli import fast_llm_main_wrapper
from fast_llm.core.distributed import allreduce_scalar, safe_barrier
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import ProcessGroupPool
from tests.utils.distributed_configs import DISTRIBUTED_TESTING_CONFIGS
from tests.utils.run_test_script import do_run_test_script_for_all_models, parse_run_distributed_script
from tests.utils.utils import DistributedSubtestContext

logger = logging.getLogger(__name__)


def main(args: list[str] | None = None) -> None:
    base_path, model_testing_config = parse_run_distributed_script(args)

    with ProcessGroupPool(timeout=20) as pool:
        failures = []
        world_size = DistributedConfig.default_world_size
        rank = DistributedConfig.default_rank
        group = pool.get_process_group(range(world_size), rank)

        for name, config in DISTRIBUTED_TESTING_CONFIGS.items():
            if config.num_gpus > world_size:
                logger.warning(f"{name} {f"SKIPPED (not enough GPUs: {config.num_gpus} > {world_size})"})")
            if DistributedConfig.default_rank < config.num_gpus:
                logger.info(f"Running {name}")
                with DistributedSubtestContext(base_path / name, rank) as subtest:
                    do_run_test_script_for_all_models(config, model_testing_config, base_path)
                assert subtest._capture_manager._global_capturing is None
                success = subtest.success
            else:
                # Worker is not needed for this one, skip.
                success = True

            # Barrier so `allreduce_scalar` doesn't go crazy in case of desync.
            safe_barrier(group, name)
            success = (
                success if group is None else allreduce_scalar(success, dtype=torch.int64, group=group) == world_size
            )
            logger.warning(f"{name} {"PASSED" if success else "FAILED"})")
            if not success:
                failures.append(name)
            if rank == 0:
                (base_path / name / "pytest_success").write_text(str(int(success)))

        # Final barrier to ensure everything is done before torchrun potentially kills workers.
        safe_barrier(group, "testing end")
        # Let pytest know how things went.
        # These should already be reported above, we repeat for convenience.
        if failures:
            raise RuntimeError(f"The following subtests failed: {", ".join(failures)}")


if __name__ == "__main__":
    with fast_llm_main_wrapper():
        main()
