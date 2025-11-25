import threading

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


@pytest.fixture
def fake_redis_server(stream_config):
    server_address = (stream_config.redis.host, stream_config.redis.port)
    server = fakeredis.TcpFakeServer(server_address, server_type="redis")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Create a redis-py client pointing at the fake server
    import redis

    client = redis.Redis(host=server_address[0], port=server_address[1])

    yield stream_config, client

    # Everything after yield = teardown
    server.shutdown()
    server.server_close()
    thread.join()


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


def generate_parallelism_variants(total_gpus: int):
    """
    Generate all valid variants of (data_groups, tensor_parallel, pipeline_parallel, sequence_parallel)
    for a  number of GPUs up to the total_gpus.
    If total_gpus is odd and > 1, fallback to nearest lower even number for decomposable parallelism.
    """
    if total_gpus > 1 and total_gpus % 2 == 1:
        total_gpus = total_gpus - 1

    if total_gpus == 0:
        return [
            {
                "data_groups": 1,
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "sequence_data_parallel": 1,
                "total_gpus": 0,
            }
        ]

    variants = [
        {
            "data_groups": 1,
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "sequence_data_parallel": 1,
            "total_gpus": 1,
        }
    ]

    for gpus in range(2, total_gpus + 1, 2):
        # try all possible numbers of data groups (1..total_gpus)
        for data_groups in range(1, gpus + 1):
            if gpus % data_groups != 0:
                continue  # cannot evenly split

            gpus_per_group = gpus // data_groups

            # now find all decompositions of gpus_per_group into tp*pp*sp
            for tp in range(1, gpus_per_group + 1):
                if gpus_per_group % tp != 0:
                    continue
                rem_after_tp = gpus_per_group // tp
                for pp in range(1, rem_after_tp + 1):
                    if rem_after_tp % pp != 0:
                        continue
                    sp = rem_after_tp // pp
                    try:
                        # instead of repeating all safeguards here just try to instantiate distributed config  to check if combination is valid
                        DistributedConfig(
                            tensor_parallel=tp,
                            pipeline_parallel=pp,
                            sequence_data_parallel=sp,
                            world_size=gpus,
                            rank=0,
                        )
                    except Exception:
                        continue
                    variants.append(
                        {
                            "data_groups": data_groups,
                            "tensor_parallel": tp,
                            "pipeline_parallel": pp,
                            "sequence_data_parallel": sp,
                            "total_gpus": gpus,
                        }
                    )
    return variants


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


def distributed_gptdata_streaming_test(
    stream_config,
    sequence_length,
    micro_batch_size,
    batch_size,
    tensor_parallel,
    pipeline_parallel,
    sequence_data_parallel,
    total_gpus,
):
    distributed = Distributed(
        DistributedConfig(
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            sequence_data_parallel=sequence_data_parallel,
        ),
        use_cpu=total_gpus > 0,
    )
    sampling_data = make_sampling(sequence_length, 0, micro_batch_size, distributed)

    data_config = {"datasets": {"streaming1": stream_config.to_dict()}, "sampling": {"shuffle": "disabled"}}
    data_config = GPTDataConfig.from_dict(data_config)

    data = GPTData(data_config, distributed.config)

    data.setup(distributed, {"streaming1": sampling_data.parameters}, "/tmp")

    with NoAutoValidate():
        batch_config = GPTBatchConfig(
            micro_batch_size=micro_batch_size, batch_size=batch_size, sequence_length=sequence_length
        )
        batch_config.setup(distributed_config=distributed.config)
        batch_config.validate()

    data_iter = data.get_iterator(batch_config, "streaming1", consumed_samples=0, num_workers=1, prefetch_factor=1)

    batch = next(data_iter)
    assert batch.tokens.tokens.shape == (micro_batch_size, sequence_length)


variants = generate_parallelism_variants(torch.cuda.device_count())


@pytest.mark.parametrize(
    "variant",
    variants,
    ids=[
        f"dg{v['data_groups']}_tp{v['tensor_parallel']}_pp{v['pipeline_parallel']}_sp{v['sequence_data_parallel']}_gpu{v['total_gpus']}"
        for v in variants
    ],
)
def test_gptdata_streaming(fake_redis_server, variant):
    if variant["total_gpus"] > 1:
        pytest.skip(f"Skipping, not implemented for gpu count {variant["total_gpus"]}")

    stream_config, fake_redis = fake_redis_server

    sequence_length = 10
    micro_batch_size = 2
    batch_size = micro_batch_size * variant["data_groups"] // variant["sequence_data_parallel"]

    for _ in range(batch_size):
        push_msg(fake_redis, stream_config, list(range(sequence_length)))

    # TODO: call with torchrun.distributed for more than 1 gpu
    distributed_gptdata_streaming_test(
        stream_config,
        sequence_length,
        micro_batch_size,
        batch_size,
        variant["tensor_parallel"],
        variant["pipeline_parallel"],
        variant["sequence_data_parallel"],
        variant["total_gpus"],
    )
