import fakeredis
import orjson
import pytest
import torch

from fast_llm.config import NoAutoValidate
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.config import (
    RedisConfig,
    SamplingConfig,
    SamplingData,
    SamplingParameters,
    ShufflingType,
    StreamingDatasetConfig,
)
from fast_llm.data.dataset.streaming import StreamingDataset
from fast_llm.data.sample.pipeline_rl import PipelineRLSample
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.models.gpt.config import GPTBatchConfig

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def fake_redis():
    """Return a FakeRedis instance."""
    return fakeredis.FakeRedis()


@pytest.fixture
def monkeypatched_redis(monkeypatch, fake_redis):
    """Monkeypatch redis.Redis globally (works even for imports inside functions)."""
    import redis

    monkeypatch.setattr(redis, "Redis", lambda *args, **kwargs: fake_redis)
    return fake_redis


@pytest.fixture
def stream_config():
    return StreamingDatasetConfig(
        redis=RedisConfig(
            host="localhost",
            port=6379,
            stream_key="test_stream",
            group_name="test_group",
            consumer_name_prefix="consumer",
        ),
        data_key="data",
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def push_msg(redis_client, config, tokens=None, is_eof=False):
    """Push a message into FakeRedis stream."""
    if is_eof:
        msg = {"eof": True}
    else:
        msg = {
            "tokens": tokens,
            "tokens_dtype": "int64",
        }
    redis_client.xadd(config.redis.stream_key, {config.data_key: orjson.dumps(msg)})


def make_sampling(sequence_length, extra_tokens, num_samples, distributed):
    return SamplingData(
        parameters=SamplingParameters(
            sequence_length=sequence_length,
            extra_tokens=extra_tokens,
            num_samples=num_samples,
            truncate_documents=False,
        ),
        config=SamplingConfig(shuffle=ShufflingType.disabled),
        distributed=distributed,
        dataset_name="test",
        cache_directory="/tmp",
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_streaming_dataset_reads_single_message(monkeypatched_redis, stream_config):
    """StreamingDataset should read a message and convert it into PipelineRLSample."""
    fake_redis = monkeypatched_redis

    distributed = Distributed(DistributedConfig(), use_cpu=True)
    dataset = StreamingDataset(stream_config, distributed)

    # Insert a message
    push_msg(fake_redis, stream_config, [1, 2, 3])

    it = iter(dataset)
    sample = next(it)

    assert isinstance(sample, PipelineRLSample)
    assert torch.equal(sample.tokens.tokens, torch.tensor([1, 2, 3], dtype=torch.int64))
    assert sample.tokens.lengths == [3]
    assert sample.loss_masking_spans is None
    assert sample.chosen_spans is None
    assert sample.rejected_spans is None


def test_streaming_dataset_reads_multiple_messages(monkeypatched_redis, stream_config):
    """StreamingDataset should read a message and convert it into PipelineRLSample."""
    fake_redis = monkeypatched_redis

    distributed = Distributed(DistributedConfig(), use_cpu=True)
    dataset = StreamingDataset(stream_config, distributed)

    # Insert a message
    push_msg(fake_redis, stream_config, [1, 2, 3])
    push_msg(fake_redis, stream_config, [1, 2, 3])
    push_msg(fake_redis, stream_config, [1, 2, 3])

    it = iter(dataset)
    for i in range(3):
        sample = next(it)

        assert isinstance(sample, PipelineRLSample)
        assert torch.equal(sample.tokens.tokens, torch.tensor([1, 2, 3], dtype=torch.int64))
        assert sample.tokens.lengths == [3]
        assert sample.loss_masking_spans is None
        assert sample.chosen_spans is None
        assert sample.rejected_spans is None


def test_sampling_1_doc_exact_fit(monkeypatched_redis, stream_config):
    """Docs exactly fill one sample."""
    fake_redis = monkeypatched_redis

    # Two rollouts: lengths 4 and 6 -> exactly 10
    push_msg(fake_redis, stream_config, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    distributed = Distributed(DistributedConfig(), use_cpu=True)
    sampler = StreamingDataset(stream_config, distributed).sample(make_sampling(10, 0, 1, distributed))

    out = list(sampler)

    assert len(out) == 1
    s = out[0]
    assert isinstance(s, PipelineRLSample)
    assert len(s) == 10
    assert s.tokens.tokens.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_sampling_2_docs_exact_fit(monkeypatched_redis, stream_config):
    """Docs exactly fill one sample."""
    fake_redis = monkeypatched_redis

    # Two rollouts: lengths 4 and 6 -> exactly 10
    push_msg(fake_redis, stream_config, [1, 2, 3, 4])
    push_msg(fake_redis, stream_config, [5, 6, 7, 8, 9, 10])

    distributed = Distributed(DistributedConfig(), use_cpu=True)
    sampler = StreamingDataset(stream_config, distributed).sample(make_sampling(10, 0, 1, distributed))

    out = list(sampler)

    assert len(out) == 1
    s = out[0]
    assert isinstance(s, PipelineRLSample)
    assert len(s) == 10
    assert s.tokens.tokens.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_sampling_skips_too_long_doc_and_padding_final(monkeypatched_redis, stream_config):
    """Rollout longer than sample_length must be dropped."""
    fake_redis = monkeypatched_redis

    push_msg(fake_redis, stream_config, list(range(20)))  # skip: too long
    push_msg(fake_redis, stream_config, list(range(8)))  # usable
    push_msg(fake_redis, stream_config, is_eof=True)

    distributed = Distributed(DistributedConfig(), use_cpu=True)
    sampler = StreamingDataset(stream_config, distributed).sample(make_sampling(10, 0, 1, distributed))

    out = list(sampler)

    assert len(out) == 1
    s = out[0]
    assert len(s) == 10
    assert s.tokens.tokens.tolist() == list(range(8)) + [-100, -100]


def test_sampling_overflow_creates_two_and_padding_final(monkeypatched_redis, stream_config):
    """A document overflowing the boundary triggers padding + next sample."""
    fake_redis = monkeypatched_redis

    push_msg(fake_redis, stream_config, list(range(6)))
    push_msg(fake_redis, stream_config, list(range(6)))
    push_msg(fake_redis, stream_config, is_eof=True)

    distributed = Distributed(DistributedConfig(), use_cpu=True)
    sampler = StreamingDataset(stream_config, distributed).sample(make_sampling(10, 0, 2, distributed))

    out = list(sampler)

    assert len(out) == 2

    # sample 1: 0..5 + pad(4)
    assert out[0].tokens.tokens.tolist() == list(range(6)) + [-100, -100, -100, -100]

    # sample 2: 0..5 + pad(4)
    assert out[1].tokens.tokens.tolist() == list(range(6)) + [-100, -100, -100, -100]


def test_data_single_consumer(monkeypatched_redis, stream_config):
    fake_redis = monkeypatched_redis

    sequence_length = 10
    samples_count = 2

    push_msg(fake_redis, stream_config, list(range(sequence_length)))
    push_msg(fake_redis, stream_config, list(range(sequence_length)))

    distributed = Distributed(DistributedConfig(), use_cpu=True)
    sampling_data = make_sampling(sequence_length, 0, samples_count, distributed)

    data_config = {"datasets": {"streaming1": stream_config.to_dict()}, "sampling": {"shuffle": "disabled"}}
    data_config = GPTDataConfig.from_dict(data_config)

    data = GPTData(data_config, distributed.config)

    data.setup(distributed, {"streaming1": sampling_data.parameters}, "/tmp")

    with NoAutoValidate():
        batch_config = GPTBatchConfig(
            micro_batch_size=samples_count, batch_size=samples_count, sequence_length=sequence_length
        )
        batch_config.setup(distributed_config=distributed.config)
        batch_config.validate()

    # TODO: check why is not working with num_workers == 1
    data_iter = data.get_iterator(batch_config, "streaming1", consumed_samples=0, num_workers=0, prefetch_factor=None)

    batch = next(data_iter)
    assert batch.tokens.tokens.shape == (2, 10)
