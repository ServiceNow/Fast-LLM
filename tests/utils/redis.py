import contextlib
import itertools
import threading
import time

import fakeredis

from fast_llm.data.dataset.config import (
    REDIS_DATA_STREAM,
    REDIS_GROUP_NAME,
    RedisConfig,
)
from fast_llm.data.dataset.streaming import RedisStreamingDocumentData


def _wait_until_stream_empty(
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


@contextlib.contextmanager
def redis_batch_producer(config: RedisConfig, sequence_length):
    with fake_redis_server(config):
        stop_event = threading.Event()
        client = config.get_client()

        def producer_loop():
            for sample_index in itertools.count():
                if stop_event.is_set():
                    break
                client.xadd(
                    REDIS_DATA_STREAM,
                    RedisStreamingDocumentData.from_dict({"tokens": [sample_index] * sequence_length}).to_message(),
                )
                if sample_index % 5 == 0:
                    _wait_until_stream_empty(client, REDIS_DATA_STREAM, REDIS_GROUP_NAME, stop_event)

        thread = threading.Thread(target=producer_loop, daemon=True)
        thread.start()

        try:
            yield
        finally:
            stop_event.set()
            thread.join(timeout=1)
            client.close()


@contextlib.contextmanager
def fake_redis_server(config: RedisConfig):
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
    print(f"Starting fake redis server at {config.host}:{config.port}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield
    finally:
        # ----- Teardown -----
        print(f"Shutting down fake redis server server at {config.host}:{config.port}")
        server.shutdown()
        server.server_close()
        thread.join()
