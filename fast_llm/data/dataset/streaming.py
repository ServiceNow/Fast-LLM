import functools
import json
import time
import typing

import redis
import torch.utils.data

from fast_llm.config import Config, Configurable, Field, config_class
from fast_llm.data.dataset.abstract import SamplableIterableDataset
from fast_llm.data.dataset.config import REDIS_DATA_STREAM, REDIS_GROUP_NAME, SamplingConfig, StreamingDatasetConfig
from fast_llm.data.document.language_model import LanguageModelDocument
from fast_llm.data.document.range import RangeDocument
from fast_llm.data.document.token_data import TokenDataDocument
from fast_llm.utils import Assert


@config_class()
class RedisStreamingDocumentData(Config):
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

    def to_document(self):
        sample_size = len(self.tokens)
        # TODO: Check explicitly that required data is available?
        return LanguageModelDocument(
            tokens=self.tokens,
            loss_masking_spans=(
                RangeDocument(ranges=[(begin, end) for begin, end in self.loss_masking_spans])
                if self.loss_masking_spans
                else None
            ),
            chosen_spans=RangeDocument(ranges=[self.chosen_span]) if self.chosen_span else None,
            rejected_spans=RangeDocument(ranges=[self.rejected_span]) if self.rejected_span else None,
            advantages=(
                None
                if self.advantage is None
                else TokenDataDocument(data=torch.full([sample_size], self.advantage, dtype=torch.float32))
            ),
            old_log_probabilities=(
                None if self.old_log_probabilities is None else TokenDataDocument(data=self.old_log_probabilities)
            ),
        )


class RedisStreamingDataset[ConfigType: StreamingDatasetConfig, DocumentType: LanguageModelDocument](
    Configurable[ConfigType], SamplableIterableDataset[DocumentType]
):
    def __init__(self, config: ConfigType):
        super().__init__(config)
        self._name = f"redis[{config.host}:{config.port}]({REDIS_DATA_STREAM}|{REDIS_GROUP_NAME})[data]"
        self._config = config

    @property
    def requires_broadcast(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return self._name

    def iterate(self, config: SamplingConfig, num_samples: int, seed: int) -> typing.Iterator[DocumentType]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError("StreamingDataset can work only with one instance per rank")

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

        start_time = time.time()
        while True:
            # XREADGROUP reads from the consumer group
            # COUNT: max number of messages to fetch at once
            # BLOCK: wait for new messages (milliseconds)
            messages = client.xreadgroup(
                groupname=REDIS_GROUP_NAME,
                consumername=f"fast_llm_consumer_{config.rank}",
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
                        yield RedisStreamingDocumentData.from_message(message).to_document()
                start_time = time.time()

            elif (t := time.time() - start_time) > self._config.timeout:
                raise TimeoutError(f"No document received after {t} seconds")
