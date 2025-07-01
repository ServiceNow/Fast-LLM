import gc
import logging
import pathlib
import typing

import torch

from fast_llm.cli import fast_llm_main_wrapper
from fast_llm.engine.checkpoint.config import (
    CheckpointFormat,
    CheckpointLoadConfig,
    CheckpointSaveConfig,
    DistributedCheckpointFormat,
    FastLLMCheckpointFormat,
)
from fast_llm.engine.distributed.distributed import ProcessGroupPool
from fast_llm.engine.multi_stage.config import StageMode
from fast_llm.utils import header
from tests.models.test_checkpoint import do_get_convert_path
from tests.utils.model_configs import ModelTestingConfig
from tests.utils.run_test_script import parse_run_distributed_script

logger = logging.getLogger(__name__)


def _test_load_and_save_parallel(
    model_testing_config: ModelTestingConfig,
    pretrained_path: pathlib.Path,
    pretrained_format: type[CheckpointFormat],
    distributed_config: dict[str, typing.Any],
    save_path: pathlib.Path,
):
    logger.info(header(save_path.name))
    logger.info(f"Loading {pretrained_format.name} checkpoint from {pretrained_path}")
    model = model_testing_config.model_class.from_pretrained(
        CheckpointLoadConfig(path=pretrained_path, format=pretrained_format),
        # The world size and rank are already set through environment variable.
        {"distributed": distributed_config},
        mode=StageMode.inference,
    )
    for save_format in (DistributedCheckpointFormat, FastLLMCheckpointFormat):
        logger.info(f"Loading {save_format.name} checkpoint to {save_path / save_format.name}")
        model.save_checkpoint(CheckpointSaveConfig(path=save_path / save_format.name, format=save_format))
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main(args: list[str] | None = None) -> None:
    base_path, model_testing_config = parse_run_distributed_script(args)

    with ProcessGroupPool(timeout=20):
        for pretrained_format, pretrained_path in (
            (
                DistributedCheckpointFormat,
                do_get_convert_path(
                    DistributedCheckpointFormat, model_testing_config.checkpoint_format, base_path=base_path.parent
                ),
            ),
            (
                FastLLMCheckpointFormat,
                do_get_convert_path(
                    FastLLMCheckpointFormat, model_testing_config.checkpoint_format, base_path=base_path.parent
                ),
            ),
            (
                model_testing_config.checkpoint_format,
                do_get_convert_path(
                    model_testing_config.checkpoint_format, DistributedCheckpointFormat, base_path=base_path.parent
                ),
            ),
        ):
            _test_load_and_save_parallel(
                model_testing_config=model_testing_config,
                pretrained_path=pretrained_path,
                pretrained_format=pretrained_format,
                distributed_config={},
                save_path=base_path / f"load_pretrained_{pretrained_format.name}_in_dp2",
            )
            _test_load_and_save_parallel(
                model_testing_config=model_testing_config,
                pretrained_path=pretrained_path,
                pretrained_format=pretrained_format,
                distributed_config={"tensor_parallel": 2},
                save_path=base_path / f"load_pretrained_{pretrained_format.name}_in_tp2",
            )
            _test_load_and_save_parallel(
                model_testing_config=model_testing_config,
                pretrained_path=pretrained_path,
                pretrained_format=pretrained_format,
                distributed_config={"tensor_parallel": 2, "sequence_tensor_parallel": True},
                save_path=base_path / f"load_pretrained_{pretrained_format.name}_in_stp2",
            )
            _test_load_and_save_parallel(
                model_testing_config=model_testing_config,
                pretrained_path=pretrained_path,
                pretrained_format=pretrained_format,
                distributed_config={"pipeline_parallel": 2},
                save_path=base_path / f"load_pretrained_{pretrained_format.name}_in_pp2",
            )

        dist = DistributedCheckpointFormat.name
        _test_load_and_save_parallel(
            model_testing_config=model_testing_config,
            pretrained_path=base_path / f"load_pretrained_{dist}_in_dp2" / dist,
            pretrained_format=DistributedCheckpointFormat,
            distributed_config={"tensor_parallel": 2, "sequence_tensor_parallel": True},
            save_path=base_path / "load_pretrained_dp2_in_stp2",
        )
        _test_load_and_save_parallel(
            model_testing_config=model_testing_config,
            pretrained_path=base_path / f"load_pretrained_{dist}_in_stp2" / dist,
            pretrained_format=DistributedCheckpointFormat,
            distributed_config={},
            save_path=base_path / "load_pretrained_stp2_in_dp2",
        )
        _test_load_and_save_parallel(
            model_testing_config=model_testing_config,
            pretrained_path=base_path / f"load_pretrained_{dist}_in_tp2" / dist,
            pretrained_format=DistributedCheckpointFormat,
            distributed_config={"tensor_parallel": 2, "sequence_tensor_parallel": True},
            save_path=base_path / "load_pretrained_tp2_in_pp2",
        )
        _test_load_and_save_parallel(
            model_testing_config=model_testing_config,
            pretrained_path=base_path / f"load_pretrained_{dist}_in_pp2" / dist,
            pretrained_format=DistributedCheckpointFormat,
            distributed_config={"tensor_parallel": 2},
            save_path=base_path / "load_pretrained_pp2_in_tp2",
        )


if __name__ == "__main__":
    with fast_llm_main_wrapper():
        main()
