import functools
import json
import typing

import redis
import torch.utils.data

from fast_llm.config import Config, Configurable, Field, config_class
from fast_llm.data.dataset.abstract import SamplableIterableDataset
from fast_llm.data.dataset.config import REDIS_DATA_STREAM, REDIS_GROUP_NAME, StreamingDatasetConfig
from fast_llm.data.sample.language_model import LanguageModelSample
from fast_llm.data.sample.range import RangeSample
from fast_llm.data.sample.token import TokenSample
from fast_llm.data.sample.token_data import TokenDataSample
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.utils import Assert


@config_class()
class RedisDocument(Config):
    """
    Schema for sending and receiving documents through redis, and the associated handling code.
    """

    tokens: torch.Tensor = Field()
    loss_masking_spans: list[tuple[int, int]] | None = Field(default=None)
    chosen_span: tuple[int, int] | None = Field(default=None)
    rejected_span: tuple[int, int] | None = Field(default=None)
    advantage: float | None = Field(default=None)
    old_log_probabilities: torch.Tensor | None = Field(default=None)

    def _validate(self):
        # Decode message
        if isinstance(self.tokens, bytes):
            self.tokens = torch.frombuffer(self.tokens, dtype=torch.int64)
        elif isinstance(self.tokens, (list, tuple)):
            self.tokens = torch.tensor(self.tokens, dtype=torch.int64)
        if isinstance(self.loss_masking_spans, str):
            self.loss_masking_spans = json.loads(self.loss_masking_spans)
        if isinstance(self.chosen_span, str):
            self.chosen_span = json.loads(self.chosen_span)
        if isinstance(self.rejected_span, str):
            self.rejected_span = json.loads(self.rejected_span)
        if isinstance(self.old_log_probabilities, bytes):
            self.old_log_probabilities = torch.frombuffer(self.old_log_probabilities, dtype=torch.float32)
        elif isinstance(self.old_log_probabilities, (list, tuple)):
            self.old_log_probabilities = torch.tensor(self.old_log_probabilities, dtype=torch.float32)
        super()._validate()
        if self.old_log_probabilities is not None:
            Assert.eq(len(self.old_log_probabilities), self.num_tokens)

    @functools.cached_property
    def num_tokens(self) -> int:
        return len(self.tokens)

    @classmethod
    def from_message(cls, message: dict[bytes, bytes]) -> typing.Self:
        # Read
        kwargs = {}
        for key, value in message.items():
            key = key.decode()
            if key == "data":
                kwargs.update(json.loads(value))
            else:
                kwargs[key] = value
        return cls.from_dict(kwargs)

    def to_message(self) -> dict[str, str | int | float | bytes]:
        # Encode message
        message: dict[str, str | int | float | bytes] = {"tokens": self.tokens.numpy().tobytes()}
        if self.old_log_probabilities is not None:
            message["old_log_probabilities"] = self.old_log_probabilities.numpy().tobytes()
        data = {}
        if self.loss_masking_spans is not None:
            data["loss_masking_spans"] = self.loss_masking_spans
        if self.chosen_span is not None:
            data["chosen_span"] = self.chosen_span
        if self.rejected_span is not None:
            data["rejected_span"] = self.rejected_span
        if self.advantage is not None:
            data["advantage"] = self.advantage
        if data:
            message["data"] = json.dumps(data)
        return message

    def to_sample(self):
        sample_size = len(self.tokens)
        return LanguageModelSample(
            tokens=TokenSample(self.tokens, [sample_size]),
            loss_masking_spans=(
                None
                if self.loss_masking_spans is None
                else RangeSample([(begin, end) for begin, end in self.loss_masking_spans], sample_size)
            ),
            chosen_spans=None if self.chosen_span is None else RangeSample([self.chosen_span], sample_size),
            rejected_spans=None if self.rejected_span is None else RangeSample([self.rejected_span], sample_size),
            advantages=(
                None
                if self.advantage is None
                else TokenDataSample(torch.full([sample_size], self.advantage, dtype=torch.float32))
            ),
            old_log_probabilities=(
                None if self.old_log_probabilities is None else TokenDataSample(self.old_log_probabilities)
            ),
        )


class RedisStreamingDataset[ConfigType: StreamingDatasetConfig, SampleType: LanguageModelSample](
    Configurable[ConfigType], SamplableIterableDataset[SampleType]
):
    def __init__(self, config: ConfigType, distributed_config: DistributedConfig):
        super().__init__(config)
        self._name = f"redis[{config.host}:{config.port}]({REDIS_DATA_STREAM}|{REDIS_GROUP_NAME})[data]"
        self._config = config
        self._rank = distributed_config.batch_data_rank
        self.is_batch_data_group_leader = (
            distributed_config.get_distributed_dim(DistributedDimNames.model_and_sequence_data).rank == 0
        )

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
            client.xgroup_create(name=REDIS_DATA_STREAM, groupname=REDIS_GROUP_NAME, id="0", mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Consumer group already exists
                pass
            else:
                raise

        while True:
            # XREADGROUP reads from the consumer group
            # COUNT: max number of messages to fetch at once
            # BLOCK: wait for new messages (milliseconds)
            messages = client.xreadgroup(
                groupname=REDIS_GROUP_NAME,
                consumername=f"fast_llm_consumer_{self._rank}",
                # ">" reads only new messages that have not been delivered to any consumer
                streams={REDIS_DATA_STREAM: ">"},
                count=1,
                block=1000,
                # No explicit ACK: messages are processed immediately; on rank failure the job restarts,
                # so message loss is acceptable and simplifies coordination
                noack=True,
            )
            if messages:
                for stream_key, messages_ in messages:
                    assert stream_key == REDIS_DATA_STREAM.encode()
                    for message_id, message in messages_:
                        print(message)
                        yield RedisDocument.from_message(message).to_sample()
