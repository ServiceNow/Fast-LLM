import typing

import torch.utils.data

from fast_llm.data.dataset.abstract import SamplableIterableDataset, SampledIterableDataset
from fast_llm.data.dataset.config import SamplingData, StreamingDatasetConfig
from fast_llm.data.dataset.sampled import NaiveSampledIterableDataset
from fast_llm.data.sample.pipeline_rl import PipelineRLSample
from fast_llm.data.sample.range import RangeSample
from fast_llm.data.sample.token import TokenSample
from fast_llm.engine.distributed.distributed import Distributed


def dtype_from_string(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError:
        raise ValueError(f"Unknown torch dtype: {name}")


class StreamingDataset[SampleType: PipelineRLSample](SamplableIterableDataset[SampleType]):
    def __init__(self, config: StreamingDatasetConfig, distributed: Distributed):
        super().__init__()
        self._name = f"redis[{config.redis.host}:{config.redis.port}]({config.redis.stream_key}|{config.redis.group_name})[{config.data_key}]"
        self._config = config
        self.batch_data_rank = distributed.config.batch_data_rank
        self.is_batch_data_group_leader = (
            distributed.model_and_sequence_data_group is None or distributed.model_and_sequence_data_group.rank() == 0
        )

    @property
    def name(self) -> str:
        return self._name

    def sample(self, config: SamplingData) -> SampledIterableDataset[PipelineRLSample]:
        # TODO: actually sample the dataset and not return docs
        return NaiveSampledIterableDataset(self, config)

    def __getstate__(self) -> tuple[str, StreamingDatasetConfig, int, bool]:
        return (self._name, self._config, self.batch_data_rank, self.is_batch_data_group_leader)

    def __setstate__(self, state: tuple[str, StreamingDatasetConfig, int, bool]):
        name, config, batch_data_rank, is_batch_data_group_leader = state
        self._name = name
        self._config = config
        self.batch_data_rank = batch_data_rank
        self.is_batch_data_group_leader = is_batch_data_group_leader

    def __iter__(self) -> typing.Iterator[PipelineRLSample]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError("StreamingDataset can work only with one instance per rank")

        if not self.is_batch_data_group_leader:
            raise RuntimeError("Must be only called on the batch data group leader")

        import orjson
        import redis
        import redis.exceptions

        r = redis.Redis(host=self._config.redis.host, port=self._config.redis.port)

        # Create the consumer group at the start of the stream ("0")
        # If the stream already exists, XGROUP CREATE will fail unless we add mkstream=True
        try:
            r.xgroup_create(
                name=self._config.redis.stream_key, groupname=self._config.redis.group_name, id="0", mkstream=True
            )
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
            messages = r.xreadgroup(
                groupname=self._config.redis.group_name,
                consumername=f"{self._config.redis.consumer_name_prefix}_{self.batch_data_rank}",
                # ">" means read only new messages that were never delivered to this consumer
                streams={self._config.redis.stream_key: ">"},
                count=1,
                block=5000,  # wait up to 5 seconds
            )

            if messages:
                for stream_key, msgs in messages:
                    assert stream_key == self._config.redis.stream_key.encode()
                    for msg_id, msg_data in msgs:
                        r.xack(self._config.redis.stream_key, self._config.redis.group_name, msg_id)
                        data = orjson.loads(msg_data[self._config.data_key.encode()])
                        yield self._sample_from_dict(data)

    def _sample_from_dict(cls, data: dict) -> PipelineRLSample:
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
        return PipelineRLSample(
            TokenSample(tokens, [sample_size]),
            RangeSample(loss_masking_spans, sample_size) if loss_masking_spans is not None else None,
            RangeSample(chosen_spans, sample_size) if chosen_spans is not None else None,
            RangeSample(rejected_spans, sample_size) if rejected_spans is not None else None,
        )
