import contextlib
import logging
import pathlib
import typing

import fakeredis
import pytest
import redis
import torch

from fast_llm.core.distributed import safe_barrier
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.config import REDIS_DATA_STREAM, RedisConfig, SamplingConfig, StreamingDatasetConfig
from fast_llm.data.dataset.streaming import RedisStreamingDataset, RedisStreamingDocumentData
from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.language_model import LanguageModelBatch, LanguageModelDocument
from fast_llm.engine.distributed.config import DistributedBackend, DistributedConfig, DistributedDimNames
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.utils import Assert
from tests.conftest import WorkerResources
from tests.utils.redis import redis_batch_producer
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
    "documents",
    [
        (range(3),),
        (range(3), range(3, 6)),
        (range(3), range(5), [9, 4]),
        ({"tokens": list(range(5)), "loss_masking_spans": [(0, 1), (2, 3)]},),
        ({"tokens": list(range(8)), "chosen_span": (0, 2), "rejected_span": (3, 5)},),
        (
            {"tokens": list(range(3)), "advantage": 0.33, "old_log_probabilities": [0.25, -0.52, 0.99]},
            {"tokens": list(range(4)), "advantage": 0.7, "old_log_probabilities": [1, 2, 3, 4]},
        ),
    ],
)
def test_streaming_dataset(
    fake_redis: fakeredis.FakeRedis,
    documents: tuple[list[int] | dict[str, typing.Any], ...],
    worker_resources: WorkerResources,
):
    """StreamingDataset should read a message and convert it into LanguageModelSample."""
    stream_config = StreamingDatasetConfig(port=worker_resources.torchrun_port, timeout=1)
    dataset_iterator = RedisStreamingDataset(stream_config).iterate(SamplingConfig(), len(documents), 0)
    documents = [document if isinstance(document, dict) else {"tokens": list(document)} for document in documents]
    for document in documents:
        fake_redis.xadd(REDIS_DATA_STREAM, RedisStreamingDocumentData.from_dict(document).to_message())
    for document in documents:
        sampled_document: LanguageModelDocument = next(dataset_iterator)
        assert isinstance(sampled_document, LanguageModelDocument)
        Assert.eq(sampled_document.tokens.tolist(), document["tokens"])

        if "loss_masking_spans" in document:
            Assert.eq(sampled_document.loss_masking_spans.ranges, document["loss_masking_spans"])
        else:
            assert sampled_document.loss_masking_spans is None

        if "chosen_span" in document:
            Assert.eq(sampled_document.chosen_spans.ranges, [document["chosen_span"]])
        else:
            assert sampled_document.chosen_spans is None

        if "rejected_span" in document:
            Assert.eq(sampled_document.rejected_spans.ranges, [document["rejected_span"]])
        else:
            assert sampled_document.rejected_spans is None

        assert sampled_document.image_patches is None

        if "advantage" in document:
            Assert.rms_close(
                sampled_document.advantages.data, torch.full([len(document["tokens"])], document["advantage"]), 1e-8
            )
        else:
            assert sampled_document.advantages is None

        if "old_log_probabilities" in document:
            Assert.rms_close(
                sampled_document.old_log_probabilities.data, torch.tensor(document["old_log_probabilities"]), 1e-8
            )
        else:
            assert sampled_document.old_log_probabilities is None


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
    dataset_iterator = iter(
        StreamingDatasetConfig(port=worker_resources.torchrun_port, timeout=1).build_and_sample(
            SamplingConfig(truncate_documents=False, micro_batch_size=5, predicted_tokens=0), 1, 0
        )
    )
    for message in messages:
        fake_redis.xadd(
            REDIS_DATA_STREAM, RedisStreamingDocumentData.from_dict({"tokens": list(message)}).to_message()
        )
    for expected_sample, expected_lengths_ in zip(expected_samples, expected_lengths, strict=True):
        documents = next(dataset_iterator)
        batch = LanguageModelBatch.from_documents(documents, pad_to_size=5)
        Assert.eq(batch.tokens.tolist(), list(expected_sample))
        Assert.eq(batch.lengths, expected_lengths_)
        assert batch.loss_masking_spans is None
        assert batch.advantages is None
        assert batch.old_log_probabilities is None


_NUM_BATCHES = 2
_SEQUENCE_LENGTH = 10


def _get_distributed_config(distributed_config_dict: dict[str, typing.Any], world_size: int = 1) -> DistributedConfig:
    return DistributedConfig.from_dict(
        distributed_config_dict,
        {
            "world_size": world_size,
            "local_world_size": world_size,
            "use_cuda": False,
            "backend": DistributedBackend.gloo,
        },
    )


def _run_test_data_streaming(path: pathlib.Path, distributed_config: DistributedConfig, port: int):
    redis_config = RedisConfig(port=port + 100, timeout=1)

    data = GPTData(
        GPTDataConfig(
            datasets={"train": {"type": "streaming", "port": port + 100}},
            micro_batch_size=_SEQUENCE_LENGTH,
            truncate_documents=False,
        ),
        distributed_config,
    )
    data.setup(path / "cache")

    distributed = Distributed(distributed_config)
    with (
        redis_batch_producer(redis_config, _SEQUENCE_LENGTH)
        if distributed_config.rank == 0
        else contextlib.nullcontext()
    ):
        safe_barrier(distributed.world_group, "streaming test begin")
        data.sample_dataset(
            "train",
            LanguageModelBatchPreprocessingConfig(predicted_tokens=0),
            distributed_config.batch_data_parallel * _NUM_BATCHES,
        )
        data_iter = data.get_iterator(
            "train", consumed_samples=0, num_workers=0, prefetch_factor=None, timeout=5, preprocess=False
        )
        batches = [next(data_iter) for _ in range(_NUM_BATCHES)]
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            torch.stack([batch.tokens for batch in batches]),
            path / f"rank_{distributed_config.batch_data_rank}_"
            f"{distributed_config.get_distributed_dim(DistributedDimNames.model_and_sequence_data).rank}.pt",
        )
        # Wait for other processes to finish before shutting down the server.
        safe_barrier(distributed.world_group, "streaming test end")


def check_data_streaming_results(path: pathlib.Path, distributed_config: DistributedConfig):
    sample_indexes = set()
    for batch_data_rank in range(distributed_config.batch_data_parallel):
        batches_tokens = torch.load(path / f"rank_{batch_data_rank}_0.pt")
        Assert.eq(batches_tokens.shape, (_NUM_BATCHES, _SEQUENCE_LENGTH))
        for model_and_sequence_data_rank in range(
            1, distributed_config.get_distributed_dim(DistributedDimNames.model_and_sequence_data).size
        ):
            Assert.all_equal(
                torch.load(path / f"rank_{batch_data_rank}_{model_and_sequence_data_rank}.pt"), batches_tokens
            )
        sample_indexes.update(batches_tokens.flatten().tolist())
    Assert.eq(len(sample_indexes), distributed_config.batch_data_parallel * _NUM_BATCHES)


def _run_test_data_streaming_distributed(
    test_context: DistributedTestContext, base_path: pathlib.Path, port: int
) -> None:
    # Import all dynamic classes. TODO: needed?
    import fast_llm.cli  # noqa

    for name, num_gpus, distributed_config_dict in _DISTRIBUTED_TESTING_CONFIGS:
        with test_context.subtest(base_path, name, num_gpus) as subtest:
            logger.info(name, subtest.do_run)
            if subtest.do_run:
                distributed_config = _get_distributed_config(distributed_config_dict, num_gpus)
                _run_test_data_streaming(base_path / name, distributed_config, port)


def test_data_streaming(result_path, worker_resources):
    distributed_config = _get_distributed_config({})
    path = result_path / "data_streaming/single_gpu"
    _run_test_data_streaming(path, distributed_config, worker_resources.torchrun_port)
    check_data_streaming_results(path, distributed_config)


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
    distributed_config = _get_distributed_config(distributed_config_dict, num_gpus)
    check_data_streaming_results(path, distributed_config)
