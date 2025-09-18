import gc
import logging

import torch

from fast_llm.cli import fast_llm_main_wrapper
from fast_llm.config import NoAutoValidate
from fast_llm.core.distributed import safe_barrier
from fast_llm.engine.checkpoint.config import (
    CheckpointLoadConfig,
    CheckpointSaveConfig,
    DistributedCheckpointFormat,
    FastLLMCheckpointFormat,
)
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import ProcessGroupPool
from fast_llm.engine.multi_stage.config import StageMode
from fast_llm.utils import Assert, header
from tests.utils.model_configs import ModelTestingConfig
from tests.utils.run_test_script import parse_run_distributed_script
from tests.utils.save_load_configs import DISTRIBUTED_SAVE_LOAD_CONFIGS, DistributedSaveLoadConfig
from tests.utils.utils import DistributedSubtestContext

logger = logging.getLogger(__name__)


def _test_load_and_save_parallel(
    model_testing_config: ModelTestingConfig,
    config: DistributedSaveLoadConfig,
):
    logger.info(header(config.name))
    logger.info(f"Loading {config.load_format} checkpoint from {config.load_path}")
    with NoAutoValidate():
        load_config = CheckpointLoadConfig(path=config.load_path, format=config.load_format)
    load_config.setup(model_testing_config.model_config_class)
    load_config.validate()
    model = model_testing_config.model_class.from_pretrained(
        load_config,
        # The world size and rank are already set through environment variable.
        {"distributed": config.distributed},
        mode=StageMode.inference,
    )
    for save_format in (DistributedCheckpointFormat, FastLLMCheckpointFormat):
        logger.info(f"Loading {save_format.name} checkpoint to {config.save_path / save_format.name}")
        model.save_checkpoint(CheckpointSaveConfig(path=config.save_path / save_format.name, format=save_format))
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main(args: list[str] | None = None) -> None:
    base_path, model_testing_config, do_capture = parse_run_distributed_script(args)

    if do_capture:
        logger.warning(
            "Capturing output and forwarding to associated tests. Run with `--no-distributed-capture` to disable."
        )

    with ProcessGroupPool(timeout=20) as pool:
        failures = []
        world_size = DistributedConfig.default_world_size
        rank = DistributedConfig.default_rank
        group = pool.get_process_group(range(world_size), rank)

        for config in DISTRIBUTED_SAVE_LOAD_CONFIGS.values():
            if config.load_format == "{checkpoint_format}" and model_testing_config.checkpoint_format is None:
                continue
            config = config.resolve(base_path, model_testing_config)
            Assert.eq(world_size, config.num_gpus)
            with DistributedSubtestContext(base_path, config.name, group, world_size, enabled=do_capture) as subtest:
                _test_load_and_save_parallel(
                    model_testing_config=model_testing_config,
                    config=config,
                )
            if not subtest.success:
                failures.append(config.name)

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
