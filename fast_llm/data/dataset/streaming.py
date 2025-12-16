import typing

import torch.utils.data
import xxhash

from fast_llm.data.dataset.abstract_iterable import SamplableIterableDataset, SampledIterableDataset
from fast_llm.data.dataset.config import HashType, IngestionType, SamplingData, StreamingDatasetConfig
from fast_llm.data.dataset.sampled import NaiveSampledIterableDataset
from fast_llm.data.sample.language_model import LanguageModelSample
from fast_llm.data.sample.range import RangeSample
from fast_llm.data.sample.token import TokenSample
from fast_llm.engine.config_utils.run import is_main_rank
from fast_llm.engine.distributed.distributed import Distributed

if typing.TYPE_CHECKING:
    import redis


def dtype_from_string(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError:
        raise ValueError(f"Unknown torch dtype: {name}")


class StreamingDataset[SampleType: LanguageModelSample](SamplableIterableDataset[SampleType]):
    def __init__(self, config: StreamingDatasetConfig, distributed: Distributed):
        super().__init__()
        if distributed.config.pipeline_parallel > 1:
            # NOTE: It is not yet clear whether the issue comes from the streaming dataset
            # itself or from the distributed data-loader wrappers, but currently it
            # interferes with pipeline-parallel training and causes a timeout during
            # the training step.
            raise NotImplementedError("Streaming dataset support is not implemented for pipeline-parallel training.")

        self._name = f"redis[{config.redis.host}:{config.redis.port}]({config.redis.stream_key}|{config.group_name})[{config.redis.payload_key}]"
        self._config = config
        self.batch_data_rank = distributed.config.batch_data_rank
        self.batch_data_parallel = distributed.config.batch_data_parallel
        self.is_batch_data_group_leader = (
            distributed.model_and_sequence_data_group is None or distributed.model_and_sequence_data_group.rank() == 0
        )
        self.payload_key_b = self._config.redis.payload_key.encode()
        self.hash_key_b = self._config.hash_key.encode()

        self._set_consumer_count()

    @property
    def name(self) -> str:
        return self._name

    def sample(self, config: SamplingData) -> SampledIterableDataset[LanguageModelSample]:
        # TODO: actually sample the dataset and not return docs
        return NaiveSampledIterableDataset(self, config)

    def _set_consumer_count(self):
        import redis

        if is_main_rank():
            redis_client = redis.Redis(host=self._config.redis.host, port=self._config.redis.port)
            redis_client.hset(f"{self._config.redis.stream_key}:consumer_count", "0", self.batch_data_parallel)

    def __getstate__(self) -> tuple[str, StreamingDatasetConfig, int, int, bool, bytes, bytes]:
        return (
            self._name,
            self._config,
            self.batch_data_parallel,
            self.batch_data_rank,
            self.is_batch_data_group_leader,
            self.payload_key_b,
            self.hash_key_b,
        )

    def __setstate__(self, state: tuple[str, StreamingDatasetConfig, int, bool, bytes, bytes]):
        name, config, batch_data_parallel, batch_data_rank, is_batch_data_group_leader, payload_key_b, hash_key_b = (
            state
        )
        self._name = name
        self._config = config
        self.batch_data_parallel = batch_data_parallel
        self.batch_data_rank = batch_data_rank
        self.is_batch_data_group_leader = is_batch_data_group_leader
        self.payload_key_b = payload_key_b
        self.hash_key_b = hash_key_b

    def __iter__(self) -> typing.Iterator[LanguageModelSample]:
        import orjson
        import redis

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError("StreamingDataset can work only with one instance per rank")

        if not self.is_batch_data_group_leader:
            raise RuntimeError("Must be only called on the batch data group leader")

        redis_client = redis.Redis(host=self._config.redis.host, port=self._config.redis.port)

        match self._config.ingestion_type:
            case IngestionType.CONSUMER_GROUP:
                messages_iter = self._iter_consumer_group
            case IngestionType.ONE_STREAM:
                messages_iter = self._iter_stream
            case IngestionType.N_STREAMS:
                messages_iter = self._iter_stream
            case _:
                raise ValueError(f"Unknown ingestion type {self._config.ingestion_type}")

        ack_hash = f"{self._config.redis.stream_key}:ack"
        consumer_id = str(self.batch_data_rank)
        # If one stream each groups receives all data otherwise each groups receives just its data
        if self._config.ingestion_type == IngestionType.ONE_STREAM:
            ack_period = self._config.ack_period_per_consumer * self.batch_data_parallel
        else:
            ack_period = self._config.ack_period_per_consumer

        for msg_data in messages_iter(redis_client, ack_hash=ack_hash, consumer_id=consumer_id, ack_period=ack_period):
            data = orjson.loads(msg_data[self.payload_key_b])
            yield self._sample_from_msg_data(data)

    def _iter_consumer_group(
        self, redis_client: "redis.Redis", ack_hash: str, consumer_id: str, ack_period: int
    ) -> typing.Iterator[LanguageModelSample]:
        import redis.exceptions

        # Create the consumer group at the start of the stream ("0")
        # If the stream already exists, XGROUP CREATE will fail unless we add mkstream=True
        try:
            redis_client.xgroup_create(
                name=self._config.redis.stream_key, groupname=self._config.group_name, id="0", mkstream=True
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Consumer group already exists
                pass
            else:
                raise

        processed = 0
        while True:
            # XREADGROUP reads from the consumer group
            # COUNT: max number of messages to fetch at once
            # BLOCK: wait for new messages (milliseconds)
            messages = redis_client.xreadgroup(
                groupname=self._config.group_name,
                consumername=f"{self._config.consumer_name_prefix}_{self.batch_data_rank}",
                # ">" reads only new messages that have not been delivered to any consumer
                streams={self._config.redis.stream_key: ">"},
                count=1,
                block=1000,
                # No explicit ACK: messages are processed immediately; on rank failure the job restarts,
                # so message loss is acceptable and simplifies coordination
                noack=True,
            )
            if messages:
                for stream_key, msgs in messages:
                    assert stream_key == self._config.redis.stream_key.encode()
                    for msg_id, msg_data in msgs:
                        processed += 1
                        # TODO: or do it after processing all received messaged then count > 1?
                        if processed % ack_period == 0:
                            redis_client.hset(ack_hash, consumer_id, msg_id)

                        yield msg_data

    def _iter_stream(
        self, redis_client: "redis.Redis", ack_hash: str, consumer_id: str, ack_period: int
    ) -> typing.Iterator[LanguageModelSample]:
        last_id = "0-0"
        stream_key = self._config.redis.stream_key
        if self._config.ingestion_type == IngestionType.N_STREAMS:
            stream_key += f"_{self.batch_data_rank}"
        stream_key_b = stream_key.encode()
        processed = 0
        while True:
            messages = redis_client.xread(
                streams={stream_key: last_id},
                count=1,
                block=1000,
            )
            if not messages:
                continue
            for this_stream_key_b, msgs in messages:
                assert this_stream_key_b == stream_key_b
                for msg_id, msg_data in msgs:
                    last_id = msg_id

                    processed += 1
                    # TODO: or do it after processing all received messaged then count > 1?
                    if processed % ack_period == 0:
                        redis_client.hset(ack_hash, consumer_id, last_id)

                    if self._config.ingestion_type == IngestionType.N_STREAMS or self._is_for_this_rank(
                        msg_id, msg_data, processed - 1
                    ):
                        yield msg_data

    def _is_for_this_rank(self, msg_id: bytes, msg_data: dict, msg_index: int) -> bool:
        hash_type = self._config.hash_type

        if hash_type is HashType.MESSAGE_INDEX:
            h = msg_index
        elif hash_type is HashType.MESSAGE_ID:
            h = xxhash.xxh64(msg_id).intdigest()
        elif hash_type is HashType.MESSAGE_BODY:
            h = xxhash.xxh64(msg_data[self.payload_key_b]).intdigest()
        elif hash_type is HashType.PRODUCER_PROVIDED:
            h = self._get_hash_key_value(msg_data[self.hash_key_b])
        else:
            raise ValueError(f"Unknown hash_type: {hash_type}")

        return (h % self.batch_data_parallel) == self.batch_data_rank

    def _get_hash_key_value(self, value: bytes | int | str):
        if isinstance(value, int):
            # already an integer
            msg_hash = value
        elif isinstance(value, bytes):
            try:
                # try decoding as UTF-8 string and converting to int
                msg_hash = int(value.decode("utf-8"))
            except ValueError:
                # not an integer, treat as a hash string
                import xxhash

                msg_hash = xxhash.xxh64(value).intdigest()
        elif isinstance(value, str):
            try:
                msg_hash = int(value)
            except ValueError:
                # not an integer, treat as a hash string
                import xxhash

                msg_hash = xxhash.xxh64(value.encode("utf-8")).intdigest()
        else:
            raise TypeError(f"Unexpected type for hash key: {type(value)}")
        return msg_hash

    def _sample_from_msg_data(self, data: dict) -> LanguageModelSample:
        tokens = torch.tensor(data["tokens"], dtype=dtype_from_string(data["tokens_dtype"]))
        sample_size = len(tokens)
        if "loss_masking_spans" in data:
            loss_masking_spans = [tuple(el) for el in data["loss_masking_spans"]]
        else:
            loss_masking_spans = None
        if "chosen_spans" in data:
            chosen_spans = [tuple(el) for el in data["chosen_spans"]]
        else:
            chosen_spans = None
        if "rejected_spans" in data:
            rejected_spans = [tuple(el) for el in data["rejected_spans"]]
        else:
            rejected_spans = None
        return LanguageModelSample(
            TokenSample(tokens, [sample_size]),
            RangeSample(loss_masking_spans, sample_size) if loss_masking_spans is not None else None,
            RangeSample(chosen_spans, sample_size) if chosen_spans is not None else None,
            RangeSample(rejected_spans, sample_size) if rejected_spans is not None else None,
        )
