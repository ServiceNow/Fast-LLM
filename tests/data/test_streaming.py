import logging
import os
import pickle

import fakeredis
import pytest
import torch

from fast_llm.data.dataset.streaming import RedisStreamingDataset
from fast_llm.data.sample.language_model import LanguageModelSample
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from tests.utils.redis import make_sampling, push_msg, redis_batch_producer
from tests.utils.utils import requires_cuda

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def generate_parallelism_variants(total_gpus: int):
    """
    Generate all valid variants of (data_groups, tensor_parallel, pipeline_parallel, sequence_parallel)
    for a  number of GPUs up to the total_gpus.
    If total_gpus is odd and > 1, fallback to nearest lower even number for decomposable parallelism.
    """
    if total_gpus > 1 and total_gpus % 2 == 1:
        total_gpus = total_gpus - 1

    if total_gpus < 2:
        # No gpu and one gpu tests are the same,
        # so no need of creation of variant for a single gpu
        return []

    variants = []

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
                # TODO: currently streaming dataset does not support pipeline parallel setup
                # for pp in range(1, rem_after_tp + 1):
                for pp in range(1, 2):
                    if rem_after_tp % pp != 0:
                        continue
                    sp = rem_after_tp // pp
                    try:
                        # instead of repeating all safeguards here just try to
                        # instantiate distributed config  to check if combination is valid
                        dist_config = DistributedConfig(
                            tensor_parallel=tp,
                            pipeline_parallel=pp,
                            sequence_data_parallel=sp,
                            world_size=gpus,
                            # TODO: works only on one node
                            local_world_size=gpus,
                            rank=0,
                        )
                    except Exception:
                        continue

                    variants.append(
                        {
                            "data_groups": data_groups,
                            "batch_data_parallel": dist_config.batch_data_parallel,
                            "tensor_parallel": tp,
                            "pipeline_parallel": pp,
                            "sequence_data_parallel": sp,
                            "total_gpus": gpus,
                        }
                    )
    return variants


def run_distributed_gptdata_streaming_test(
    fake_redis_server,
    variant,
    run_distributed_script,
    result_path,
    request,
):
    import tests.data.gptdata_streaming_test

    stream_config, fake_redis, fake_redis_server_killer = fake_redis_server

    sequence_length = 10
    micro_batch_size = 2
    batch_size = micro_batch_size * variant["batch_data_parallel"]
    tensor_parallel = variant["tensor_parallel"]
    pipeline_parallel = variant["pipeline_parallel"]
    sequence_data_parallel = variant["sequence_data_parallel"]
    total_gpus = variant["total_gpus"]
    redis_port = stream_config.port

    result_path = result_path / "distributed_gptdata_streaming_test" / request.node.name

    with redis_batch_producer(
        redis_client=fake_redis,
        fake_redis_server_killer=fake_redis_server_killer,
        batch_size=batch_size,
        sequence_length=10,
    ):
        if total_gpus > 0:
            script = [
                "-m",
                tests.data.gptdata_streaming_test.__name__,
                "--sequence-length",
                str(sequence_length),
                "--micro-batch-size",
                str(micro_batch_size),
                "--batch-size",
                str(batch_size),
                "--tensor-parallel",
                str(tensor_parallel),
                "--pipeline-parallel",
                str(pipeline_parallel),
                "--sequence-data-parallel",
                str(sequence_data_parallel),
                "--total-gpus",
                str(total_gpus),
                "--result-path",
                str(result_path),
                "--redis-port",
                str(redis_port),
            ]
            # TODO: distributed_capture is ignored now inside the script
            if request.config.getoption("distributed_capture"):
                logger.warning(
                    "Capturing output and forwarding to associated tests. Run with `--no-distributed-capture` to disable."
                )
            else:
                script.append("--no-distributed-capture")

            env = os.environ.copy()
            env["PYTHONHASHSEED"] = "42"
            run_distributed_script(script, num_gpus=total_gpus, env=env)
        else:
            tests.data.gptdata_streaming_test.distributed_gptdata_streaming_test(
                sequence_length=sequence_length,
                micro_batch_size=micro_batch_size,
                batch_size=batch_size,
                tensor_parallel=tensor_parallel,
                pipeline_parallel=pipeline_parallel,
                sequence_data_parallel=sequence_data_parallel,
                total_gpus=total_gpus,
                redis_port=redis_port,
                result_path=result_path,
            )

    check_distributed_gptdata_streaming_test_results(
        result_path=result_path,
        micro_batch_size=micro_batch_size,
        batch_data_parallel=variant["batch_data_parallel"],
        total_gpu=variant["total_gpus"],
    )


def check_distributed_gptdata_streaming_test_results(
    result_path,
    micro_batch_size,
    batch_data_parallel,
    total_gpu,
):
    batch_data_parallel_size = total_gpu // batch_data_parallel if total_gpu > 0 else 1
    sample_idx = set()
    for i in range(batch_data_parallel):
        ref_batch = None
        for j in range(batch_data_parallel_size):
            with (result_path / f"{i}_{j}" / "batch.pkl").open("rb") as f:
                batch = pickle.load(f)
            if ref_batch is None:
                ref_batch = batch
            else:
                # batches for same batch_data_parallel_group must be equal on all ranks
                assert torch.equal(batch.tokens.tokens, ref_batch.tokens.tokens)
        for j in range(micro_batch_size):
            val = ref_batch.tokens.tokens[j, 0].item()
            # all samples in batches between groups and in the batch must be unique
            assert val not in sample_idx
            sample_idx.add(val)
    # unique sample count must be the same as global batch size
    assert len(sample_idx) == micro_batch_size * batch_data_parallel


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_streaming_dataset_reads_single_message(monkeypatched_redis, stream_config):
    """StreamingDataset should read a message and convert it into LanguageModelSample."""
    fake_redis = monkeypatched_redis

    dataset = RedisStreamingDataset(stream_config, DistributedConfig())

    # Insert a message
    push_msg(fake_redis, [1, 2, 3])

    it = iter(dataset)
    sample = next(it)

    assert isinstance(sample, LanguageModelSample)
    assert torch.equal(sample.tokens.tokens, torch.tensor([1, 2, 3], dtype=torch.int64))
    assert sample.tokens.lengths == [3]
    assert sample.loss_masking_spans is None
    assert sample.chosen_spans is None
    assert sample.rejected_spans is None


def test_streaming_dataset_reads_multiple_messages(monkeypatched_redis, stream_config):
    """StreamingDataset should read a message and convert it into LanguageModelSample."""
    fake_redis = monkeypatched_redis

    dataset = RedisStreamingDataset(stream_config, DistributedConfig())

    # Insert a message
    push_msg(fake_redis, [1, 2, 3])
    push_msg(fake_redis, [1, 2, 3])
    push_msg(fake_redis, [1, 2, 3])

    it = iter(dataset)
    for i in range(3):
        sample = next(it)

        assert isinstance(sample, LanguageModelSample)
        assert torch.equal(sample.tokens.tokens, torch.tensor([1, 2, 3], dtype=torch.int64))
        assert sample.tokens.lengths == [3]
        assert sample.loss_masking_spans is None
        assert sample.chosen_spans is None
        assert sample.rejected_spans is None


def test_sampling_1_doc_exact_fit(monkeypatched_redis, stream_config):
    """Docs exactly fill one sample."""
    fake_redis = monkeypatched_redis

    push_msg(fake_redis, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    distributed = Distributed(DistributedConfig(), use_cpu=True)
    sampler = RedisStreamingDataset(stream_config, distributed.config).sample(make_sampling(10, 0, 1, distributed))

    out = next(iter(sampler))

    assert isinstance(out, LanguageModelSample)
    assert len(out) == 10
    assert out.tokens.tokens.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_sampling_2_docs_exact_fit(monkeypatched_redis, stream_config):
    """Docs exactly fill one sample."""
    fake_redis = monkeypatched_redis

    # Two rollouts: lengths 4 and 6 -> exactly 10
    push_msg(fake_redis, [1, 2, 3, 4])
    push_msg(fake_redis, [5, 6, 7, 8, 9, 10])

    distributed = Distributed(DistributedConfig(), use_cpu=True)
    sampler = RedisStreamingDataset(stream_config, distributed.config).sample(make_sampling(10, 0, 1, distributed))

    out = next(iter(sampler))

    assert isinstance(out, LanguageModelSample)
    assert len(out) == 10
    assert out.tokens.tokens.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_sampling_skips_too_long_doc_and_padding_final(monkeypatched_redis, stream_config):
    """Rollout longer than sample_length must be dropped."""
    fake_redis = monkeypatched_redis

    push_msg(fake_redis, list(range(20)))  # skip: too long
    push_msg(fake_redis, list(range(10)))  # usable

    distributed = Distributed(DistributedConfig(), use_cpu=True)
    sampler = RedisStreamingDataset(stream_config, distributed.config).sample(make_sampling(10, 0, 1, distributed))

    out = next(iter(sampler))

    # too big message is skipped and next message is returned instead
    assert len(out) == 10
    assert out.tokens.tokens.tolist() == list(range(10))


def test_sampling_overflow_creates_two(monkeypatched_redis, stream_config):
    """A document overflowing the boundary triggers padding + next sample."""
    fake_redis = monkeypatched_redis

    push_msg(fake_redis, list(range(6)))
    push_msg(fake_redis, list(range(10)))

    distributed = Distributed(DistributedConfig(), use_cpu=True)
    sampler = RedisStreamingDataset(stream_config, distributed.config).sample(make_sampling(10, 0, 2, distributed))

    sampler_iter = iter(sampler)
    out = [next(sampler_iter)]
    out.append(next(sampler_iter))

    # sample 1: 0..5 + pad(4)
    assert out[0].tokens.tokens.tolist() == list(range(6)) + [-100, -100, -100, -100]

    # sample 2: 0..5 + pad(4)
    assert out[1].tokens.tokens.tolist() == list(range(10))


def test_gptdata_streaming_single_consumer(fake_redis_server, run_distributed_script_lean, result_path, request):

    run_distributed_gptdata_streaming_test(
        fake_redis_server=fake_redis_server,
        variant={
            "data_groups": 1,
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "sequence_data_parallel": 1,
            "total_gpus": 0,
            "batch_data_parallel": 1,
        },
        run_distributed_script=run_distributed_script_lean,
        result_path=result_path,
        request=request,
    )


variants = generate_parallelism_variants(torch.cuda.device_count())


@pytest.mark.slow
@requires_cuda
@pytest.mark.parametrize(
    "variant",
    variants,
    ids=[
        f"dg{v['data_groups']}_tp{v['tensor_parallel']}_pp{v['pipeline_parallel']}_sp{v['sequence_data_parallel']}_gpu{v['total_gpus']}"
        for v in variants
    ],
)
def test_gptdata_streamin_gpus(fake_redis_server, variant, run_distributed_script_lean, result_path, request):
    # TODO: make tests on the same number of gpu as subtests
    #  similar to how it is done in the test_model for speed
    run_distributed_gptdata_streaming_test(
        fake_redis_server=fake_redis_server,
        variant=variant,
        run_distributed_script=run_distributed_script_lean,
        result_path=result_path,
        request=request,
    )
