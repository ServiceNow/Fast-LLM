import contextlib
import logging
import pathlib
import typing

import fakeredis
import pytest
import redis
import torch

from fast_llm.config import NoAutoValidate
from fast_llm.core.distributed import safe_barrier
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.config import REDIS_DATA_STREAM, RedisConfig, SamplingParameters, StreamingDatasetConfig
from fast_llm.data.dataset.streaming import RedisDocument, RedisStreamingDataset
from fast_llm.data.preprocessing.language_model import LanguageModelPreprocessingConfig
from fast_llm.data.sample.language_model import LanguageModelSample
from fast_llm.engine.distributed.config import DistributedBackend, DistributedConfig, DistributedDimNames
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.models.gpt.config import GPTBatchConfig
from fast_llm.utils import Assert
from tests.conftest import WorkerResources
from tests.data.common import get_sampling_data
from tests.utils.redis import make_sampling, redis_batch_producer
from tests.utils.subtest import DistributedTestContext

logger = logging.getLogger(__name__)


@pytest.fixture
def fake_redis(monkeypatch):
    """Monkeypatch redis.Redis globally."""
    fake_redis = fakeredis.FakeRedis()
    monkeypatch.setattr(redis, "Redis", lambda *args, **kwargs: fake_redis)
    try:
        yield fake_redis
    finally:
        fake_redis.close()


@pytest.mark.parametrize(
    ("documents", "preprocessing"),
    [
        ((range(3),), {}),
        ((range(3), range(3, 6)), {}),
        ((range(3), range(5), [9, 4]), {}),
        (({"tokens": list(range(5)), "loss_masking_spans": [(0, 1), (2, 3)]},), {"use_loss_masking_spans": True}),
        (
            ({"tokens": list(range(8)), "chosen_span": (0, 2), "rejected_span": (3, 5)},),
            {"use_preference_spans": True},
        ),
        (
            (
                {"tokens": list(range(3)), "advantage": 0.33, "old_log_probabilities": [0.25, -0.52, 0.99]},
                {"tokens": list(range(4)), "advantage": 0.7, "old_log_probabilities": [1, 2, 3, 4]},
            ),
            {"use_grpo_data": True},
        ),
    ],
)
def test_streaming_dataset(
    fake_redis: fakeredis.FakeRedis,
    documents: tuple[list[int] | dict[str, typing.Any], ...],
    preprocessing: dict,
    worker_resources: WorkerResources,
):
    """StreamingDataset should read a message and convert it into LanguageModelSample."""
    stream_config = StreamingDatasetConfig(port=worker_resources.torchrun_port)
    dataset_iterator = RedisStreamingDataset(stream_config, DistributedConfig()).iterate(
        get_sampling_data(len(documents), preprocessing=LanguageModelPreprocessingConfig.from_dict(preprocessing))
    )
    documents = [document if isinstance(document, dict) else {"tokens": list(document)} for document in documents]
    for document in documents:
        fake_redis.xadd(REDIS_DATA_STREAM, RedisDocument.from_dict(document).to_message())
    for document in documents:
        sample = next(dataset_iterator)
        assert isinstance(sample, LanguageModelSample)
        Assert.eq(sample.tokens.tokens.tolist(), document["tokens"])
        Assert.eq(sample.tokens.lengths, [len(document["tokens"])])

        if "loss_masking_spans" in document:
            Assert.eq(sample.loss_masking_spans.ranges, document["loss_masking_spans"])
        else:
            assert sample.loss_masking_spans is None

        if "chosen_span" in document:
            Assert.eq(sample.chosen_spans.ranges, [document["chosen_span"]])
        else:
            assert sample.chosen_spans is None

        if "rejected_span" in document:
            Assert.eq(sample.rejected_spans.ranges, [document["rejected_span"]])
        else:
            assert sample.rejected_spans is None

        assert sample.image_patches is None

        if "advantage" in document:
            Assert.rms_close(
                sample.advantages.data, torch.full([len(document["tokens"])], document["advantage"]), 1e-8
            )
        else:
            assert sample.advantages is None

        if "old_log_probabilities" in document:
            Assert.rms_close(sample.old_log_probabilities.data, torch.tensor(document["old_log_probabilities"]), 1e-8)
        else:
            assert sample.old_log_probabilities is None


@pytest.mark.parametrize(
    ("messages", "expected_samples", "expected_lengths"),
    [
        ((range(5),), (range(5),), ([5],)),  # Single message, exact fit.
        ((range(3), [3, 4]), (range(5),), ([3, 2],)),  # Two messages, exact fit.
        ((range(6), range(5)), (range(5),), ([5],)),  # Two messages, one dropped.
        (
            (range(3), range(5)),
            (
                [0, 1, 2, -100, -100],
                range(5),
            ),
            (
                [3, 2],
                [5],
            ),
        ),  # Two messages, one padded.
    ],
)
def test_streaming_sampled_dataset(
    fake_redis: fakeredis.FakeRedis,
    messages: tuple[list[int], ...],
    expected_samples: tuple[list[int], ...],
    expected_lengths: tuple[int, ...],
    worker_resources: WorkerResources,
):
    """StreamingDataset should read a message and convert it into LanguageModelSample."""
    stream_config = StreamingDatasetConfig(port=worker_resources.torchrun_port)
    distributed = Distributed(DistributedConfig(use_cuda=False))
    dataset_iterator = iter(
        RedisStreamingDataset(stream_config, distributed.config).sample(make_sampling(5, 1, distributed))
    )
    for message in messages:
        fake_redis.xadd(REDIS_DATA_STREAM, RedisDocument.from_dict({"tokens": list(message)}).to_message())
    for expected_sample, expected_lengths_ in zip(expected_samples, expected_lengths, strict=True):
        sample = next(dataset_iterator)
        assert isinstance(sample, LanguageModelSample)
        Assert.eq(sample.tokens.tokens.tolist(), list(expected_sample))
        Assert.eq(sample.tokens.lengths, expected_lengths_)
        assert sample.loss_masking_spans is None
        assert sample.chosen_spans is None
        assert sample.rejected_spans is None


_NUM_BATCHES = 1


def _get_distributed_and_batch_config(
    distributed_config_dict: dict[str, typing.Any], world_size: int = 1
) -> tuple[DistributedConfig, GPTBatchConfig]:
    distributed_config = DistributedConfig.from_dict(
        distributed_config_dict,
        {
            "world_size": world_size,
            "local_world_size": world_size,
            "use_cuda": False,
            "backend": DistributedBackend.gloo,
        },
    )
    with NoAutoValidate():
        batch_config = GPTBatchConfig(micro_batch_size=2, sequence_length=10)
        batch_config.setup(distributed_config=distributed_config)
    batch_config.validate()
    return distributed_config, batch_config


def _run_test_data_streaming(
    path: pathlib.Path, distributed_config: DistributedConfig, batch_config: GPTBatchConfig, port: int
):
    redis_config = RedisConfig(port=port + 100)

    data = GPTData(GPTDataConfig(datasets={"train": {"type": "streaming", "port": port + 100}}), distributed_config)
    distributed = Distributed(distributed_config)
    with (
        redis_batch_producer(redis_config, batch_config) if distributed_config.rank == 0 else contextlib.nullcontext()
    ):
        data.setup(
            distributed=distributed,
            sampling_parameters={
                "train": SamplingParameters(
                    sequence_length=batch_config.sequence_length,
                    extra_tokens=0,
                    num_samples=batch_config.batch_size * _NUM_BATCHES,
                    truncate_documents=False,
                )
            },
            preprocessing=LanguageModelPreprocessingConfig(),
            cache_directory=path / "cache",
            timeout=5,
        )

        data_iter = data.get_iterator(batch_config, "train", consumed_samples=0, num_workers=0, prefetch_factor=None)
        batches = [next(data_iter) for _ in range(_NUM_BATCHES)]
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            torch.stack([batch.tokens.tokens[:, 0] for batch in batches]),
            path / f"rank_{distributed_config.batch_data_rank}_"
            f"{distributed_config.get_distributed_dim(DistributedDimNames.model_and_sequence_data).rank}.pt",
        )
        # Wait for other processes to finish before shutting down the server.
        safe_barrier(distributed.world_group, "streaming test end")


def check_data_streaming_results(
    path: pathlib.Path,
    distributed_config: DistributedConfig,
    batch_config: GPTBatchConfig,
):
    sample_indexes = set()
    for batch_data_rank in range(distributed_config.batch_data_parallel):
        batches_tokens = torch.load(path / f"rank_{batch_data_rank}_0.pt")
        Assert.eq(batches_tokens.shape, (_NUM_BATCHES, batch_config.micro_batch_size))
        for model_and_sequence_data_rank in range(
            1, distributed_config.get_distributed_dim(DistributedDimNames.model_and_sequence_data).size
        ):
            Assert.all_equal(
                torch.load(path / f"rank_{batch_data_rank}_{model_and_sequence_data_rank}.pt"), batches_tokens
            )
        sample_indexes.update(batches_tokens.flatten().tolist())
    Assert.eq(len(sample_indexes), _NUM_BATCHES * batch_config.batch_size)


def _run_test_data_streaming_distributed(
    test_context: DistributedTestContext, base_path: pathlib.Path, port: int
) -> None:
    # Import all dynamic classes. TODO: needed?
    import fast_llm.cli  # noqa

    print(_DISTRIBUTED_TESTING_CONFIGS)
    for name, num_gpus, distributed_config_dict in _DISTRIBUTED_TESTING_CONFIGS:
        with test_context.subtest(base_path, name, num_gpus) as subtest:
            print(name, subtest.do_run)
            if subtest.do_run:
                distributed_config, batch_config = _get_distributed_and_batch_config(distributed_config_dict, num_gpus)
                _run_test_data_streaming(base_path / name, distributed_config, batch_config, port)


def test_data_streaming(result_path, worker_resources):
    distributed_config, batch_config = _get_distributed_and_batch_config({})
    path = result_path / "data_streaming/single_gpu"
    _run_test_data_streaming(path, distributed_config, batch_config, worker_resources.torchrun_port)
    check_data_streaming_results(path, distributed_config, batch_config)


_DISTRIBUTED_TESTING_CONFIGS = [
    ("dp2", 2, {}),
    ("sdp2", 2, {"sequence_data_parallel": 2}),
    ("tp2", 2, {"tensor_parallel": 2}),
    ("pp2", 2, {"pipeline_parallel": 2}),
    ("dp2_sdp2", 4, {"sequence_data_parallel": 2}),
    ("dp2_tp2", 4, {"tensor_parallel": 2}),
    ("dp2_pp2", 4, {"pipeline_parallel": 2}),
    ("sdp2_tp2", 4, {"sequence_data_parallel": 2, "tensor_parallel": 2}),
    ("sdp2_pp2", 4, {"sequence_data_parallel": 2, "pipeline_parallel": 2}),
    ("tp2_pp2", 4, {"tensor_parallel": 2, "pipeline_parallel": 2}),
]


@pytest.mark.slow
@pytest.mark.depends_on(on=["test_data_streaming"])
def test_run_data_streaming_distributed(run_parallel_script, result_path, worker_resources):
    run_parallel_script(
        _run_test_data_streaming_distributed,
        (result_path / "data_streaming", worker_resources.torchrun_port),
        world_size=4,
        backend=DistributedBackend.gloo,
        use_cuda=False,  # Disable device count check.
    )


@pytest.mark.slow
@pytest.mark.depends_on(on=["test_data_streaming"])
@pytest.mark.parametrize(("name", "num_gpus", "distributed_config_dict"), _DISTRIBUTED_TESTING_CONFIGS)
def test_data_streaming_distributed(result_path, name, num_gpus, distributed_config_dict, report_subtest):
    report_subtest(path := result_path / f"data_streaming/{name}", num_gpus, use_cuda=False)
    distributed_config, batch_config = _get_distributed_and_batch_config(distributed_config_dict, num_gpus)
    check_data_streaming_results(path, distributed_config, batch_config)
