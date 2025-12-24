import json
import typing

import redis
import torch.utils.data

from fast_llm.config import Configurable
from fast_llm.data.dataset.abstract import SamplableIterableDataset
from fast_llm.data.dataset.config import StreamingDatasetConfig
from fast_llm.data.sample.language_model import LanguageModelSample
from fast_llm.data.sample.range import RangeSample
from fast_llm.data.sample.token import TokenSample
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames


def dtype_from_string(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError:
        raise ValueError(f"Unknown torch dtype: {name}")


REDIS_DATA_KEY = "fast_llm_streaming"
REDIS_GROUP_NAME = "fast_llm_group"


class RedisStreamingDataset[ConfigType: StreamingDatasetConfig, SampleType: LanguageModelSample](
    Configurable[ConfigType], SamplableIterableDataset[SampleType]
):
    def __init__(self, config: ConfigType, distributed_config: DistributedConfig):
        super().__init__(config)
        # if distributed_config.pipeline_parallel > 1:
        # NOTE: It is not yet clear whether the issue comes from the streaming dataset
        # itself or from the distributed data-loader wrappers, but currently it
        # interferes with pipeline-parallel training and causes a timeout during
        # the training step.
        # raise NotImplementedError("Streaming dataset support is not implemented for pipeline-parallel training.")

        self._name = f"redis[{config.host}:{config.port}]({REDIS_DATA_KEY}|{REDIS_GROUP_NAME})[data]"
        self._config = config
        self._rank = distributed_config.batch_data_rank
        self.is_batch_data_group_leader = (
            distributed_config.get_distributed_dim(DistributedDimNames.model_and_sequence_data).rank == 0
        )

        # TODO: Not needed?
        # if distributed_config.rank == 0:
        #    redis_client = redis.Redis(host=self._config.host, port=self._config.port)
        #    redis_client.hset(f"{REDIS_DATA_KEY}:consumer_count", "0", str(distributed_config.batch_data_parallel))

    @property
    def requires_broadcast(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return self._name

    def __iter__(self) -> typing.Iterator[LanguageModelSample]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError("StreamingDataset can work only with one instance per rank")

        if not self.is_batch_data_group_leader:
            raise RuntimeError("Must be only called on the batch data group leader")

        client = redis.Redis(host=self._config.host, port=self._config.port)

        # Create the consumer group at the start of the stream ("0")
        # If the stream already exists, XGROUP CREATE will fail unless we add mkstream=True
        try:
            client.xgroup_create(name=REDIS_DATA_KEY, groupname=REDIS_GROUP_NAME, id="0", mkstream=True)
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
            messages = client.xreadgroup(
                groupname=REDIS_GROUP_NAME,
                consumername=f"fast_llm_consumer_{self._rank}",
                # ">" reads only new messages that have not been delivered to any consumer
                streams={REDIS_DATA_KEY: ">"},
                count=1,
                block=1000,
                # No explicit ACK: messages are processed immediately; on rank failure the job restarts,
                # so message loss is acceptable and simplifies coordination
                noack=True,
            )
            if messages:
                for stream_key, msgs in messages:
                    assert stream_key == REDIS_DATA_KEY.encode()
                    for msg_id, msg_data in msgs:
                        processed += 1
                        # TODO: or do it after processing all received messaged then count > 1?
                        if processed % self._config.acknowledge_interval == 0:
                            client.hset(f"{REDIS_DATA_KEY}:ack", str(self._rank), msg_id)

                        yield self._read_document(json.loads(msg_data[b"data"]))

    def _read_document(self, data: dict) -> LanguageModelSample:
        tokens = torch.tensor(data["tokens"], dtype=dtype_from_string(data["tokens_dtype"]))
        sample_size = len(tokens)
        if "loss_masking_spans" in data:
            loss_masking_spans = RangeSample([(begin, end) for begin, end in data["loss_masking_spans"]], sample_size)
        else:
            loss_masking_spans = None
        if "chosen_spans" in data:
            chosen_spans = RangeSample([(begin, end) for begin, end in data["chosen_spans"]], sample_size)
        else:
            chosen_spans = None
        if "rejected_spans" in data:
            rejected_spans = RangeSample([(begin, end) for begin, end in data["rejected_spans"]], sample_size)
        else:
            rejected_spans = None
        return LanguageModelSample(
            TokenSample(tokens, [sample_size]),
            loss_masking_spans,
            chosen_spans,
            rejected_spans,
        )
