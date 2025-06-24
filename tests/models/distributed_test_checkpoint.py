import argparse
import pathlib

from fast_llm.engine.checkpoint.config import DistributedCheckpointFormat, FastLLMCheckpointFormat
from fast_llm.engine.distributed.distributed import ProcessGroupPool
from tests.models.test_checkpoint import get_convert_paths
from tests.utils.model_configs import MODEL_CONFIGS
from tests.utils.run_test_script import do_run_test_script_for_all_models


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("rendezvous_port", type=int)
    parser.add_argument("torchrun_port", type=int)
    parser.add_argument("base_path", type=pathlib.Path)
    parser.add_argument("model_testing_config", type=str)
    parsed = parser.parse_args(args)
    return parsed.rendezvous_port, parsed.torchrun_port, parsed.base_path, MODEL_CONFIGS[parsed.model_testing_config]


def _test_load_and_save_parallel(fixture_args, test_name, distributed_args, pretrained_path, pretrained_format):
    # TODO: Just save and load the model instead, no need for an actual run.
    do_run_test_script_for_all_models(
        [
            # First we load a checkpoint.
            f"pretrained.path={pretrained_path}",
            f"pretrained.format={pretrained_format}",
            # We run for one mock iteration.
            "training.train_iters=1",
            "schedule.skip_step=True",
            # Then we save a checkpoint (distributed format) and an export (fast_llm format).
            "training.checkpoint.interval=1",
            "training.export.interval=1",
            "training.export.format=fast_llm",
        ]
        + distributed_args,
        test_name=test_name,
        **fixture_args,
    )


def main(args: list[str] | None = None) -> None:
    rendezvous_port, torchrun_port, base_path, model_testing_config = parse_args(args)
    convert_paths = get_convert_paths(base_path)

    fixture_args = {
        "rendezvous_port": rendezvous_port,
        "torchrun_port": torchrun_port,
        "base_path": base_path,
        "model_testing_config": model_testing_config,
        "num_gpus": 2,
    }

    with ProcessGroupPool(timeout=20):
        for pretrained_format, pretrained_path in (
            (DistributedCheckpointFormat.name, convert_paths["distributed_0"]),
            (FastLLMCheckpointFormat.name, convert_paths["fast_llm_0"]),
            (model_testing_config.checkpoint_format.name, convert_paths["huggingface_0"]),
        ):
            _test_load_and_save_parallel(
                fixture_args,
                test_name=f"test_load_pretrained_{pretrained_format}_in_dp2",
                distributed_args=[],
                pretrained_path=pretrained_path,
                pretrained_format=pretrained_format,
            )
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
    main()
