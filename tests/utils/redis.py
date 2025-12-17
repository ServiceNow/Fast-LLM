import contextlib
import pathlib
import socket
import threading
import time

import fakeredis
import orjson
import pytest

from fast_llm.data.dataset.config import (
    SamplingConfig,
    SamplingData,
    SamplingParameters,
    ShufflingType,
    StreamingDatasetConfig,
)
from fast_llm.data.dataset.streaming import REDIS_DATA_KEY, REDIS_GROUP_NAME
from fast_llm.data.preprocessing.language_model import LanguageModelPreprocessingConfig


def find_free_port():
    """Find a free TCP port and return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def push_msg(redis_client, tokens=None, stream_key_suffix=None, payload_key="data", stream_key=REDIS_DATA_KEY):
    """Push a message into FakeRedis stream."""
    msg = {
        "tokens": tokens,
        "tokens_dtype": "int64",
    }
    if stream_key_suffix is not None:
        stream_key += stream_key_suffix
    redis_client.xadd(stream_key, {payload_key: orjson.dumps(msg)})


class FakeRedisServerKiller:
    def __init__(self, server):
        self._server = server
        self._is_killed = False
        self._lock = threading.Lock()

    def kill(self):
        with self._lock:
            if not self._is_killed:
                try:
                    self._server.shutdown()
                    self._server.server_close()
                finally:
                    self._is_killed = True


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
        res = redis_client.hget(f"{REDIS_DATA_KEY}:consumer_count", "0")
        if res is None:
            time.sleep(0.05)
            continue
        return int(res)


@contextlib.contextmanager
def redis_batch_producer(redis_client, fake_redis_server_killer, batch_size, sequence_length, num_batches=None):
    stop_event = threading.Event()
    thread_exc = []

    def producer_loop():
        try:
            batch_idx = 0
            while not stop_event.is_set():
                if num_batches is not None and batch_idx >= num_batches:
                    break
                for i in range(batch_size):
                    if stop_event.is_set():
                        return
                    push_msg(
                        redis_client,
                        [batch_idx * batch_size + i] * sequence_length,
                    )
                wait_until_stream_empty(
                    redis_client,
                    REDIS_DATA_KEY,
                    REDIS_GROUP_NAME,
                    stop_event,
                )
                batch_idx += 1
        except Exception as e:
            # if failed to push messages kill redis server so waiting side in the test would unlock
            fake_redis_server_killer.kill()
            thread_exc.append(e)
            raise

    thread = threading.Thread(target=producer_loop, daemon=True)
    thread.start()

    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=10)
        if thread_exc:
            raise thread_exc[0]


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
        cache_directory=pathlib.Path("/tmp"),
        preprocessing=LanguageModelPreprocessingConfig(),
    )


@pytest.fixture
def stream_config():
    # TODO: ======= Not safe for parallel tests? =======
    return StreamingDatasetConfig.from_dict({"redis": {"port": find_free_port()}})


@pytest.fixture
def fake_redis_server(stream_config):
    # We search for free port as port from previous test can still be not free even after server shutdown
    server_address = (stream_config.redis.host, stream_config.redis.port)

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

    server = fakeredis.TcpFakeServer(server_address, server_type="redis")
    server_killer = FakeRedisServerKiller(server)

    # ----- Start server thread -----
    def serve():
        try:
            server.serve_forever()
        except Exception:
            # Extra safety: catch anything from serve_forever
            pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    # ----- reate a redis-py client pointing at the fake serve -----
    import redis

    client = redis.Redis(host=server_address[0], port=server_address[1])

    yield stream_config, client, server_killer

    # ----- Teardown -----
    server_killer.kill()
    thread.join()
