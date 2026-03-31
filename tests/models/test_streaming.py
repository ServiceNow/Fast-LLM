import contextlib
import dataclasses
import functools
import json
import logging
import pathlib

import pytest
import safetensors
import torch

from fast_llm.core.distributed import broadcast as _broadcast
from fast_llm.core.distributed import broadcast_object as _broadcast_object
from fast_llm.engine.distributed.config import DistributedBackend
from fast_llm.engine.distributed.distributed import ProcessGroupPool
from fast_llm.engine.training.config import StreamingTrainerCallbackConfig
from fast_llm.engine.training.streaming import REDIS_TRAINING_FIELD, REDIS_TRAINING_STREAM
from fast_llm.utils import Assert
from tests.conftest import WorkerResources
from tests.models.test_checkpoint import compare_safetensor_files
from tests.utils.distributed_configs import DistributedTestingConfig
from tests.utils.model_configs import ModelTestingConfig, ModelTestingGroup, update_and_add_testing_config
from tests.utils.redis import redis_batch_producer
from tests.utils.run_test_script import do_run_test_script_for_all_models
from tests.utils.subtest import DistributedTestContext
from tests.utils.utils import requires_cuda


@dataclasses.dataclass(kw_only=True)
class StreamingDistributedTestingConfig(DistributedTestingConfig):
    consumer_count: int = (1,)

    @functools.cached_property
    def total_gpus(self) -> int:
        return self.num_gpus + self.consumer_count


_DISTRIBUTED_STREAMING_CONFIGS = [
    StreamingDistributedTestingConfig(name="streaming_simple", config_args=[], num_gpus=1, consumer_count=1),
    StreamingDistributedTestingConfig(name="streaming_dp2", config_args=[], num_gpus=2, consumer_count=1),
    StreamingDistributedTestingConfig(
        name="streaming_sdp2_c2",
        config_args=["model.distributed.sequence_data_parallel=2"],
        num_gpus=2,
        consumer_count=2,
    ),
    StreamingDistributedTestingConfig(
        name="streaming_tp2", config_args=["model.distributed.tensor_parallel=2"], num_gpus=2, consumer_count=2
    ),
    StreamingDistributedTestingConfig(
        name="streaming_stp2_c2",
        config_args=[
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=true",
            "callbacks.streaming.broadcast.external_world_size=2",
        ],
        num_gpus=2,
        consumer_count=2,
    ),
]


def _run_event_consumer(
    streaming_config: StreamingTrainerCallbackConfig, consumer_index: int, base_path: pathlib.Path
) -> None:
    client = streaming_config.get_client()
    init_method = f"tcp://{streaming_config.broadcast.host}:{streaming_config.broadcast.port}"
    logging.info(f"Waiting for weights broadcast rendezvous at {init_method} ...")
    path = base_path / "streaming"
    path.mkdir(parents=True, exist_ok=True)
    field = REDIS_TRAINING_FIELD.encode()
    # TODO: Create a custom process group instead.
    pool = None
    try:
        world_size = streaming_config.broadcast.external_world_size + 1
        consumer_rank = consumer_index + 1
        backend = DistributedBackend.nccl if torch.cuda.is_available() else DistributedBackend.gloo
        pool = ProcessGroupPool(
            rank=consumer_rank,
            world_size=world_size,
            timeout=streaming_config.timeout,
            init_method=init_method,
            backend=backend,
            device=(
                torch.device("cuda", torch.cuda.current_device())
                if backend == DistributedBackend.nccl
                else torch.device("cpu")
            ),
        )
        process_group = pool.get_process_group(range(world_size), consumer_rank)
        timeout_ms = int(streaming_config.timeout * 1000)
        last_id = "0-0"
        while True:
            result = client.xread(
                streams={REDIS_TRAINING_STREAM: last_id},
                count=1,
                block=timeout_ms,
            )
            if not result:
                raise TimeoutError(f"No message received after {timeout_ms} ms...")

            ((stream, events),) = result
            Assert.eq(stream.decode(), REDIS_TRAINING_STREAM)
            Assert.eq(len(events), 1)
            for last_id, message in events:
                Assert.eq(message.keys(), {field})
                message = json.loads(message[field].decode())
                logging.info(f"Received: {message}")
                if message["type"] == "training_finished":
                    return
                elif message["type"] == "weights_ready":
                    weights = {}
                    while True:
                        meta = _broadcast_object(None, process_group, src=0)
                        if meta is None:
                            print(f"Weight broadcast finished")
                            break
                        logging.info(f"receiving {meta}")
                        shard_name, layer_name, tensor_size, tensor_type = meta
                        tensor = torch.zeros(tuple(tensor_size), dtype=tensor_type, device="cuda")
                        _broadcast(tensor, 0, process_group)
                        if shard_name == "weights":
                            weights[layer_name] = tensor
                    safetensors.torch.save_file(
                        weights, path / f"rank_{consumer_index}_step_{message["step"]}.safetensors"
                    )

    finally:
        if pool is not None:
            pool.shutdown()


def _run_model_streaming_configs(
    test_context: DistributedTestContext, base_path: pathlib.Path, model_testing_config: ModelTestingConfig, port: int
) -> None:
    # Import all dynamic classes.
    import fast_llm.cli  # noqa

    for config_index, config in enumerate(_DISTRIBUTED_STREAMING_CONFIGS):
        config_port = port + config_index
        model_testing_config = update_and_add_testing_config(
            model_testing_config,
            None,
            updates={
                ("data", "datasets"): {"training": {"port": config_port, "timeout": 1.0}},
                ("training", "export"): {"format": model_testing_config.checkpoint_format.name, "interval": 1},
                "callbacks": {
                    "streaming": {
                        "type": "streaming",
                        "port": config_port,
                        "broadcast": {
                            "port": config_port + 1000,
                            "external_world_size": config.consumer_count,
                        },
                        "export": {"format": model_testing_config.checkpoint_format.name},
                        "timeout": 120,
                    }
                },
                # Disable tensor logging.
                ("run", "tensor_logs"): {},
                ("model", "multi_stage"): {},
            },
            groups={},
        )
        with test_context.subtest(base_path, config.name, config.total_gpus) as subtest:
            if subtest.do_run:
                if test_context.rank < config.num_gpus:
                    do_run_test_script_for_all_models(config, model_testing_config, base_path)
                elif test_context.rank < config.total_gpus:
                    training_config = model_testing_config.trainer_config_class.from_dict(
                        model_testing_config.config_dict
                    )
                    with (
                        redis_batch_producer(
                            training_config.callbacks["streaming"], training_config.data.micro_batch_size
                        )
                        if test_context.rank == config.num_gpus
                        else contextlib.nullcontext()
                    ):
                        _run_event_consumer(
                            training_config.callbacks["streaming"],
                            test_context.rank - config.num_gpus,
                            base_path / config.name,
                        )


@requires_cuda
@pytest.mark.slow
@pytest.mark.model_testing_group(ModelTestingGroup.streaming, ModelTestingGroup.distributed)
def test_model_streaming(run_parallel_script, model_testing_config, run_test_script_base_path, worker_resources):
    # `test_run_model_distributed_streaming` and `test_model_distributed_streaming need a common dependency
    # so they are placed in the same testing group and run in the same distributed process.
    pass


@requires_cuda
@pytest.mark.slow
@pytest.mark.depends_on(on=["test_model_streaming[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.streaming, ModelTestingGroup.distributed)
def test_run_model_distributed_streaming(
    run_parallel_script, model_testing_config, run_test_script_base_path, worker_resources
):
    if torch.cuda.device_count() < 2:
        pytest.skip(f"Not enough GPUs")
    model_testing_config.get_dataset()
    # Use a fixed shift to avoid port conflicts with other distributed tests.
    port = worker_resources.torchrun_port + 4321
    run_parallel_script(
        _run_model_streaming_configs,
        (run_test_script_base_path, model_testing_config, port),
        world_size=torch.cuda.device_count(),
        backend=model_testing_config.distributed_backend,
    )


@pytest.mark.slow
@requires_cuda
@pytest.mark.depends_on(on=["test_model_streaming[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.streaming, ModelTestingGroup.distributed)
@pytest.mark.parametrize("config", _DISTRIBUTED_STREAMING_CONFIGS)
def test_model_distributed_streaming(
    config: StreamingDistributedTestingConfig,
    run_distributed_script,
    model_testing_config,
    run_test_script_base_path,
    worker_resources: WorkerResources,
    report_subtest,
):
    report_subtest(path := run_test_script_base_path / config.name, config.total_gpus)
    compare_safetensor_files(
        path / "export" / model_testing_config.checkpoint_format.name / f"1/model_0.safetensors",
        *(
            path / "streaming" / f"rank_{consumer_index}_step_1.safetensors"
            for consumer_index in range(config.consumer_count)
        ),
    )
