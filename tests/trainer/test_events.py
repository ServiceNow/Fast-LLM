import contextlib
import copy
import os
import pathlib
import subprocess
import time

import pytest
import safetensors
import torch
import yaml

from tests.data.test_streaming import redis_batch_producer
from tests.utils.model_configs import MODEL_CONFIGS
from tests.utils.utils import requires_cuda


@contextlib.contextmanager
def run_fake_events_consumers(
    model_config: dict,
    test_result_path: pathlib.Path,
    broadcast_world_size: int,
    fake_consumers_broadcast_ranks: list[int],
    assigned_gpus: list[str],
    timeout: float = 30.0,  # seconds
):
    """
    Context manager to run fake event consumer subprocesses for testing.

    Each subprocess gets a separate config and CUDA_VISIBLE_DEVICES.

    After exiting the context, all subprocesses are ensured to terminate.
    Raises RuntimeError if any subprocess exits with non-zero code.
    """
    import tests.trainer.events_fake_consumer

    assert len(assigned_gpus) > 0
    assert len(assigned_gpus) == len(fake_consumers_broadcast_ranks)

    processes = []

    try:
        for i, gpu in enumerate(assigned_gpus):
            consumer_path = test_result_path / str(i)
            consumer_path.mkdir(parents=True, exist_ok=True)

            # Deep copy config and update per consumer
            this_config = copy.deepcopy(model_config)
            this_config["consumer"] = {
                "idx": i,
                "results_path": consumer_path / "results",
                "world_size": broadcast_world_size,
                "rank": fake_consumers_broadcast_ranks[i],
            }
            this_config_path = consumer_path / "config.yaml"

            # Save config as YAML
            with open(this_config_path, "w") as f:
                yaml.safe_dump(convert_paths(this_config), f)

            # Build subprocess command
            script = [
                "python",
                "-m",
                tests.trainer.events_fake_consumer.__name__,
                str(this_config_path),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            # Start subprocess
            proc = subprocess.Popen(script, env=env)
            processes.append(proc)

        # Yield control to the caller while subprocesses run
        yield

    finally:
        # Wait for processes to exit or kill after timeout
        start_time = time.time()
        for proc in processes:
            try:
                remaining = max(0, timeout - (time.time() - start_time))
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()

        # Check exit codes
        errors = [(i, p.returncode) for i, p in enumerate(processes) if p.returncode != 0]
        if errors:
            raise RuntimeError(f"Some fake consumer subprocesses failed: {errors}")


def run_fast_llm_training(model_config, run_distributed_script, assigned_gpus):
    import fast_llm.cli

    config_path = model_config["run"]["experiment_dir"] / "load_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("wt") as f:
        yaml.safe_dump(convert_paths(model_config), f)

    script = [
        "-m",
        fast_llm.cli.__name__,
        "train",
        "gpt",
        "--config",
        str(config_path),
    ]

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in assigned_gpus)
    run_distributed_script(script, num_gpus=len(assigned_gpus), env=env)


def compare_test_tensors_to_checkpoint(test_safetensor_path: str, checkpoint_dir: str):
    """
    Compare a test-saved safetensor file (a dict of tensors)
    to all safetensors in a checkpoint directory.

    Checks:
        - tensor names must match
        - shapes must match
        - dtypes must match
        - values must match (exact)
    """

    # -------------------------
    # Load test tensor file
    # -------------------------
    test_tensors = {}
    with safetensors.safe_open(test_safetensor_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            test_tensors[key] = f.get_tensor(key)

    assert len(test_tensors) > 0, f"No tensors found in {test_safetensor_path}."

    # -------------------------
    # Load checkpoint tensors
    # -------------------------
    checkpoint_tensors = {}

    for file in os.listdir(checkpoint_dir):
        if file.endswith(".safetensors"):
            path = os.path.join(checkpoint_dir, file)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in checkpoint_tensors:
                        raise AssertionError(
                            f"Duplicate tensor name '{key}' across checkpoint {checkpoint_dir} files."
                        )
                    checkpoint_tensors[key] = f.get_tensor(key)

    assert len(checkpoint_tensors) > 0, f"No safetensors found in checkpoint directory: {checkpoint_dir}"

    # -------------------------
    # Compare tensor sets
    # -------------------------
    test_names = set(test_tensors.keys())
    ckpt_names = set(checkpoint_tensors.keys())

    unexpected_in_test = test_names - ckpt_names
    missing_in_test = ckpt_names - test_names

    assert not missing_in_test, "Tensors missing in {test_safetensor_path}:\n" + "\n".join(sorted(missing_in_test))
    assert not unexpected_in_test, "Unexpected tensors in {test_safetensor_path}:\n" + "\n".join(
        sorted(unexpected_in_test)
    )

    # -------------------------
    # Compare individual tensors
    # -------------------------
    for name in sorted(test_names):
        t_test = test_tensors[name]
        t_ckpt = checkpoint_tensors[name]

        # dtype
        assert t_test.dtype == t_ckpt.dtype, f"Mismatch in dtype for '{name}': " f"{t_test.dtype} != {t_ckpt.dtype}"

        # shape
        assert t_test.shape == t_ckpt.shape, (
            f"Mismatch in shape for '{name}': " f"{tuple(t_test.shape)} != {tuple(t_ckpt.shape)}"
        )

        # values
        if not torch.equal(t_test, t_ckpt):
            diff = (t_test - t_ckpt).abs()
            max_diff = diff.max().item()
            idx = (diff > 0).nonzero(as_tuple=False)
            example = idx[0].tolist() if idx.numel() > 0 else "unknown"

            raise AssertionError(
                f"Tensor content mismatch for '{name}'.\n"
                f"Max difference: {max_diff}\n"
                f"Example differing index: {example}"
            )

    # If we reached here → all is good
    return True


def check_events_results(
    test_results_path_fast_llm,
    test_results_path_consumers,
    consumer_count,
    training_steps,
    model_checkpoint_format,
):
    for consumer_idx in range(consumer_count):
        consumer_test_results_path = test_results_path_consumers / str(consumer_idx) / "results"
        assert (consumer_test_results_path / "training_finished").is_file()
        for training_step in range(1, training_steps + 1):
            compare_test_tensors_to_checkpoint(
                consumer_test_results_path / f"{training_step}.safetensors",
                test_results_path_fast_llm / "export" / model_checkpoint_format / str(training_step),
            )


def convert_paths(obj):
    if isinstance(obj, dict):
        return {k: convert_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_paths(v) for v in obj)
    elif isinstance(obj, pathlib.Path):
        return str(obj)
    else:
        return obj


@pytest.mark.slow
@requires_cuda
def test_trainer_events_with_streaming_2gpus(fake_redis_server, run_distributed_script_lean, result_path):
    if torch.cuda.device_count() < 2:
        pytest.skip(f"Skipping, not enough gpus, {torch.cuda.device_count()} of 2 needed")

    stream_config, fake_redis_client, fake_redis_server_killer = fake_redis_server
    test_result_path = result_path / "training_events_with_streaming_2gpus"
    test_result_path_fast_llm = test_result_path / "fast_llm"
    test_result_path_consumers = test_result_path / "consumers"

    broadcast_world_size = 2
    fake_consumers_broadcast_ranks = [0]
    fake_consumers_assigned_gpus = [0]
    fast_llm_broadcast_rank = 1
    fast_llm_assigned_gpus = [1]
    train_iters = 2

    model_config = copy.deepcopy(MODEL_CONFIGS["mistral"].config_dict)
    model_config["data"]["datasets"] = {"training": stream_config.to_dict()}
    model_config["data"]["sampling"] = {"shuffle": "disabled"}
    model_config["training"]["train_iters"] = train_iters
    model_config["training"]["export"] = {"interval": 1, "format": MODEL_CONFIGS["mistral"].checkpoint_format.name}
    model_config["batch"]["micro_batch_size"] = 1
    model_config["batch"]["truncate_documents"] = False
    model_config["run"]["experiment_dir"] = test_result_path_fast_llm
    # We use same stream for messages in the test. Also make all fields explicit,
    #  so fake consumers can read them as well from this dict config
    model_config["events"] = {
        "weights_broadcast": {
            "enabled": True,
            "host": stream_config.redis.host,
            "port": stream_config.redis.port,
            "stream_key": "fast_llm_events",
            "message_key": "event",
            "message": "weights_ready",
            "rdvz_master_address": "127.0.0.1",
            "rdvz_master_port": 19999,
            "world_size": broadcast_world_size,
            "rank": fast_llm_broadcast_rank,
        },
        "training_finished": {
            "enabled": True,
            "host": stream_config.redis.host,
            "port": stream_config.redis.port,
            "stream_key": "fast_llm_events",
            "message_key": "event",
            "message": "training_finished",
        },
    }

    batch_size = model_config["batch"]["batch_size"]
    sequence_length = model_config["batch"]["sequence_length"]
    with redis_batch_producer(
        redis_client=fake_redis_client,
        fake_redis_server_killer=fake_redis_server_killer,
        stream_config=stream_config,
        batch_size=batch_size,
        sequence_length=sequence_length,
    ):
        with run_fake_events_consumers(
            model_config=model_config,
            test_result_path=test_result_path_consumers,
            broadcast_world_size=broadcast_world_size,
            fake_consumers_broadcast_ranks=fake_consumers_broadcast_ranks,
            assigned_gpus=fake_consumers_assigned_gpus,
        ):
            run_fast_llm_training(
                model_config=model_config,
                run_distributed_script=run_distributed_script_lean,
                assigned_gpus=fast_llm_assigned_gpus,
            )
    check_events_results(
        test_results_path_fast_llm=test_result_path_fast_llm,
        test_results_path_consumers=test_result_path_consumers,
        consumer_count=len(fake_consumers_assigned_gpus),
        training_steps=train_iters,
        model_checkpoint_format=MODEL_CONFIGS["mistral"].checkpoint_format.name,
    )
