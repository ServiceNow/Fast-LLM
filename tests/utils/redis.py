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

    # ----- Monkey-patch setup to use Resp2Writer instead of Resp3Writer -----
    # fakeredis 2.34+ hardcodes Resp3Writer for all connections, causing blocked
    # XREADGROUP timeouts to return RESP3 null (b'_\r\n') even on RESP2 connections
    # (i.e. clients that never sent HELLO 3). The redis-py RESP2 parser raises
    # Protocol Error: b'_' on this byte. Fix: replace with Resp2Writer at setup time.
    # The Resp2Writer class was introduced alongside the bug in 2.34, so use its
    # presence as the version guard.
    orig_setup = fakeredis._tcp_server.TCPFakeRequestHandler.setup
    if hasattr(fakeredis._tcp_server, "Resp3Writer"):
        # fakeredis 2.34+ hardcodes Resp3Writer for all connections, causing blocked
        # XREADGROUP timeouts to return RESP3 null (b'_\r\n') even on RESP2 connections
        # (i.e. clients that never sent HELLO 3). The redis-py RESP2 parser raises
        # Protocol Error: b'_' on this byte. Fix: replace with Resp2Writer at setup time.
        if not hasattr(fakeredis._tcp_server, "Resp2Writer"):
            raise RuntimeError(
                f"fakeredis {fakeredis.__version__} has Resp3Writer but not Resp2Writer — "
                "the workaround for the RESP2/RESP3 null encoding bug no longer applies. "
                "See tests/utils/redis.py for details."
            )

        def resp2_setup(self):
            orig_setup(self)
            if not isinstance(self.writer, fakeredis._tcp_server.Resp2Writer):
                self.writer = fakeredis._tcp_server.Resp2Writer(self.client_address, self.wfile, self)
                self.current_client.writer = self.writer

        fakeredis._tcp_server.TCPFakeRequestHandler.setup = resp2_setup

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
        fakeredis._tcp_server.TCPFakeRequestHandler.setup = orig_setup
        fakeredis._tcp_server.TCPFakeRequestHandler.handle = orig_handle
