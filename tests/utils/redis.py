import contextlib
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
    StreamingDatasetRedisConfig,
)


def get_stream_config():
    return StreamingDatasetConfig(
        redis=StreamingDatasetRedisConfig(
            host="localhost",
            port=6379,
            stream_key="test_stream",
            payload_key="data",
        ),
        group_name="test_group",
        consumer_name_prefix="consumer",
    )


def find_free_port():
    """Find a free TCP port and return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def push_msg(redis_client, config, tokens=None):
    """Push a message into FakeRedis stream."""
    msg = {
        "tokens": tokens,
        "tokens_dtype": "int64",
    }
    redis_client.xadd(config.redis.stream_key, {config.redis.payload_key: orjson.dumps(msg)})


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


def wait_until_stream_empty(r, stream_key, group, stop_event):
    """
    Wait until lag == 0, meaning all messages have been delivered AND acknowledged.
    Absence of group mean test has not started yet, so we wait
    """
    group = group.encode()
    while not stop_event.is_set():
        groups = r.xinfo_groups(stream_key)

        g = next((g for g in groups if g["name"] == group), None)
        if g is not None:
            lag = g.get("lag", 0)
            if lag == 0:
                return

        time.sleep(0.05)


@contextlib.contextmanager
def redis_batch_producer(
    redis_client, fake_redis_server_killer, stream_config, batch_size, sequence_length, num_batches=None
):
    stop_event = threading.Event()
    thread_exc = []

    def producer_loop():
        try:
            stream = stream_config.redis.stream_key
            group = stream_config.group_name
            batch_idx = 0
            while not stop_event.is_set():
                if num_batches is not None and batch_idx >= num_batches:
                    break
                for i in range(batch_size):
                    if stop_event.is_set():
                        return
                    push_msg(redis_client, stream_config, [batch_idx * batch_size + i] * sequence_length)
                wait_until_stream_empty(redis_client, stream, group, stop_event)
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
        cache_directory="/tmp",
    )


@pytest.fixture
def stream_config():
    return get_stream_config()


@pytest.fixture
def fake_redis_server(stream_config):
    # We search for free port as port from previous test can still be not free even after server shutdown
    stream_config = stream_config.from_dict(stream_config.to_dict(), {("redis", "port"): find_free_port()})

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
