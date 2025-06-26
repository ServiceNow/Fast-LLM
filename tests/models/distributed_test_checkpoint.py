import gc
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
from tests.models.test_checkpoint import do_get_convert_path
from tests.utils.model_configs import ModelTestingConfig
from tests.utils.run_test_script import parse_run_distributed_script


def _test_load_and_save_parallel(
    model_testing_config: ModelTestingConfig,
    pretrained_path: pathlib.Path,
    pretrained_format: CheckpointFormat,
    distributed_config: dict[str, typing.Any],
    save_path: pathlib.Path,
):
    model = model_testing_config.model_class.from_pretrained(
        CheckpointLoadConfig(path=pretrained_path, format=pretrained_format),
        # The world size and rank are already set through environment variable.
        {"distributed": distributed_config},
        mode=StageMode.inference,
    )
    for save_format in (DistributedCheckpointFormat, FastLLMCheckpointFormat):
        model.save_checkpoint(CheckpointSaveConfig(path=save_path / save_format.name, format=save_format))
    del model
    gc.collect()
    torch.cuda.empty_cache()


# def _test_load_and_save_parallel(fixture_args, test_name, distributed_args, pretrained_path, pretrained_format):
#    # TODO: Just save and load the model instead, no need for an actual run.
#    do_run_test_script_for_all_models(
#        [
#            # First we load a checkpoint.
#            f"pretrained.path={pretrained_path}",
#            f"pretrained.format={pretrained_format}",
#            # We run for one mock iteration.
#            "training.train_iters=1",
#            "schedule.skip_step=True",
#            # Then we save a checkpoint (distributed format) and an export (fast_llm format).
#            "training.checkpoint.interval=1",
#            "training.export.interval=1",
#            "training.export.format=fast_llm",
#        ]
#        + distributed_args,
#        test_name=test_name,
#        **fixture_args,
#    )


def main(args: list[str] | None = None) -> None:
    base_path, model_testing_config = parse_run_distributed_script(args)

    # fixture_args = {
    #    "rendezvous_port": rendezvous_port,
    #    "torchrun_port": torchrun_port,
    #    "base_path": base_path,
    #    "model_testing_config": model_testing_config,
    #    "num_gpus": 2,
    # }

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
            # _test_load_and_save_parallel(
            #    fixture_args,
            #    test_name=f"test_load_pretrained_{pretrained_format}_in_dp2",
            #    distributed_args=[],
            #    pretrained_path=pretrained_path,
            #    pretrained_format=pretrained_format,
            # )
            # _test_load_and_save_parallel(
            #    fixture_args,
            #    test_name=f"test_load_pretrained_{pretrained_format}_in_tp2",
            #    distributed_args=["model.distributed.tensor_parallel=2"],
            #    pretrained_path=pretrained_path,
            #    pretrained_format=pretrained_format,
            # )
            # _test_load_and_save_parallel(
            #    fixture_args,
            #    test_name=f"test_load_pretrained_{pretrained_format}_in_stp2",
            #    distributed_args=["model.distributed.tensor_parallel=2", "model.distributed.sequence_tensor_parallel=true"],
            #    pretrained_path=pretrained_path,
            #    pretrained_format=pretrained_format,
            # )

        # _test_load_and_save_parallel(
        #    fixture_args,
        #    test_name=f"test_load_pretrained_dp2_in_tp2",
        #    distributed_args=["model.distributed.tensor_parallel=2", "model.distributed.sequence_tensor_parallel=true"],
        #    pretrained_path=base_path / "test_load_pretrained_distributed_in_dp2" / "checkpoint" / "1",
        #    pretrained_format=DistributedCheckpointFormat.name,
        # )
        # _test_load_and_save_parallel(
        #    fixture_args,
        #    test_name=f"test_load_pretrained_stp2_in_dp2",
        #    distributed_args=[],
        #    pretrained_path=base_path / "test_load_pretrained_distributed_in_stp2" / "checkpoint" / "1",
        #    pretrained_format=DistributedCheckpointFormat.name,
        # )
        # _test_load_and_save_parallel(
        #    fixture_args,
        #    test_name=f"test_load_pretrained_tp2_in_stp2",
        #    distributed_args=["model.distributed.tensor_parallel=2", "model.distributed.sequence_tensor_parallel=true"],
        #    pretrained_path=base_path / "test_load_pretrained_distributed_in_stp2" / "checkpoint" / "1",
        #    pretrained_format=DistributedCheckpointFormat.name,
        # )
        # _test_load_and_save_parallel(
        #    fixture_args,
        #    test_name=f"test_load_pretrained_stp2_in_tp2",
        #    distributed_args=["model.distributed.tensor_parallel=2"],
        #    pretrained_path=base_path / "test_load_pretrained_distributed_in_tp2" / "checkpoint" / "1",
        #    pretrained_format=DistributedCheckpointFormat.name,
        # )


if __name__ == "__main__":
    with fast_llm_main_wrapper():
        main()
