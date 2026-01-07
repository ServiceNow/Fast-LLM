import contextlib
import itertools
import json
import pathlib
import socket
import threading
import time

import fakeredis

from fast_llm.data.dataset.config import (
    REDIS_DATA_STREAM,
    REDIS_FIELD,
    REDIS_GROUP_NAME,
    RedisConfig,
    SamplingConfig,
    SamplingData,
    SamplingParameters,
    StreamingDatasetConfig,
)
from fast_llm.data.preprocessing.language_model import LanguageModelPreprocessingConfig
from fast_llm.models.gpt.config import GPTBatchConfig


def find_free_port():
    """Find a free TCP port and return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def push_msg(redis_client, tokens):
    """Push a message into FakeRedis stream."""
    redis_client.xadd(REDIS_DATA_STREAM, {REDIS_FIELD: json.dumps({"tokens": tokens, "tokens_dtype": "int64"})})


def wait_until_stream_empty(
    redis_client,
    stream_key,
    consumer_group,
    stop_event,
):
    """
    Wait until lag == 0, meaning all messages have been delivered AND acknowledged.
    Absence of group mean test has not started yet, so we wait
    """
    consumer_group = consumer_group.encode()
    while not stop_event.is_set():
        groups = redis_client.xinfo_groups(stream_key)

        g = next((g for g in groups if g["name"] == consumer_group), None)
        if g is not None:
            lag = g.get("lag", 0)
            if lag == 0:
                return

        time.sleep(0.05)


def get_consumer_count(redis_client, stop_event, config: StreamingDatasetConfig):
    while not stop_event.is_set():
        res = redis_client.hget(f"{REDIS_DATA_STREAM}:consumer_count", "0")
        if res is None:
            time.sleep(0.05)
            continue
        return int(res)


@contextlib.contextmanager
def redis_batch_producer(config: RedisConfig, batch_config: GPTBatchConfig):
    with fake_redis_server(config):
        stop_event = threading.Event()
        client = config.get_client()

        def producer_loop():
            for sample_index in itertools.count():
                if stop_event.is_set():
                    break
                push_msg(client, [sample_index] * batch_config.sequence_length)
                if sample_index % 5 == 0:
                    wait_until_stream_empty(client, REDIS_DATA_STREAM, REDIS_GROUP_NAME, stop_event)

        thread = threading.Thread(target=producer_loop, daemon=True)
        thread.start()

        try:
            yield
        finally:
            stop_event.set()
            thread.join(timeout=1)
            client.close()


def make_sampling(sequence_length, num_samples, distributed):
    return SamplingData(
        parameters=SamplingParameters(
            sequence_length=sequence_length,
            extra_tokens=0,
            num_samples=num_samples,
            truncate_documents=False,
        ),
        config=SamplingConfig(),
        distributed=distributed,
        dataset_name="test",
        cache_directory=pathlib.Path("/tmp"),
        preprocessing=LanguageModelPreprocessingConfig(),
    )


@contextlib.contextmanager
def fake_redis_server(config: RedisConfig):
    # We search for free port as port from previous test can still be not free even after server shutdown

    # ----- Monkey-patch handler to suppress broken pipes -----
    orig_handle = fakeredis._tcp_server.TCPFakeRequestHandler.handle

    def safe_handle(self):
        try:
            orig_handle(self)
        except (ConnectionResetError, BrokenPipeError):
            # Client disconnected abruptly (e.g., when a PyTorch DataLoader iterator is deleted).
            # These errors occur only with fake Redis and can be safely ignored.
            pass
        except Exception as e:
            print(f"Unexpected exception in fake Redis handler: {e}")

    fakeredis._tcp_server.TCPFakeRequestHandler.handle = safe_handle

    server = fakeredis.TcpFakeServer((config.host, config.port), server_type="redis")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield
    finally:
        # ----- Teardown -----
        server.shutdown()
        server.server_close()
        thread.join()
