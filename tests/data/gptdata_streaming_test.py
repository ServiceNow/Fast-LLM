import argparse
import pathlib
import pickle

from fast_llm.config import NoAutoValidate
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.config import IngestionType
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.models.gpt.config import GPTBatchConfig
from tests.utils.redis import get_stream_config, make_sampling


def distributed_gptdata_streaming_test(
    sequence_length,
    micro_batch_size,
    batch_size,
    tensor_parallel,
    pipeline_parallel,
    sequence_data_parallel,
    total_gpus,
    redis_port,
    result_path,
    ingestion_type,
):
    stream_config = get_stream_config()
    stream_config = stream_config.from_dict(
        stream_config.to_dict(), {("redis", "port"): redis_port, ("ingestion_type"): ingestion_type}
    )

    distributed = Distributed(
        DistributedConfig(
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            sequence_data_parallel=sequence_data_parallel,
        ),
        use_cpu=total_gpus == 0,
    )
    sampling_data = make_sampling(sequence_length, 0, micro_batch_size, distributed)

    data_config = {"datasets": {"streaming1": stream_config.to_dict()}, "sampling": {"shuffle": "disabled"}}
    data_config = GPTDataConfig.from_dict(data_config)

    data = GPTData(data_config, distributed.config)

    data.setup(
        distributed=distributed,
        sampling_parameters={"streaming1": sampling_data.parameters},
        preprocessing={},
        cache_directory="/tmp",
    )

    with NoAutoValidate():
        batch_config = GPTBatchConfig(
            micro_batch_size=micro_batch_size, batch_size=batch_size, sequence_length=sequence_length
        )
        batch_config.setup(distributed_config=distributed.config)
        batch_config.validate()

    data_iter = data.get_iterator(batch_config, "streaming1", consumed_samples=0, num_workers=1, prefetch_factor=1)

    batch = next(data_iter)
    # TODO: save result per batch_data_group and rank
    assert batch.tokens.tokens.shape == (micro_batch_size, sequence_length)

    result_path = (
        pathlib.Path(result_path)
        / (
            f"{distributed.config.batch_data_rank}_"
            f"{distributed.model_and_sequence_data_group.rank() if distributed.model_and_sequence_data_group is not None else 0}"
        )
        / "batch.pkl"
    )
    result_path.parent.mkdir(exist_ok=True, parents=True)
    with result_path.open("wb") as f:
        pickle.dump(batch, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Run distributed GPT data streaming test.")

    parser.add_argument("--sequence-length", type=int, required=True, help="Sequence length of the model input.")
    parser.add_argument("--micro-batch-size", type=int, required=True, help="Micro batch size.")
    parser.add_argument("--batch-size", type=int, required=True, help="Global batch size.")
    parser.add_argument("--tensor-parallel", type=int, required=True, help="Tensor parallel degree.")
    parser.add_argument("--pipeline-parallel", type=int, required=True, help="Pipeline parallel degree.")
    parser.add_argument("--sequence-data-parallel", type=int, required=True, help="Sequence data parallel degree.")
    parser.add_argument("--total-gpus", type=int, required=True, help="Total number of GPUs available.")
    parser.add_argument("--redis-port", type=int, required=True, help="Redis port to connect to.")
    parser.add_argument("--result-path", type=str, required=True, help="Path to save test results.")
    parser.add_argument("--ingestion-type", type=str, required=True, help="Ingestion type used in streaming dataset.")

    return parser.parse_args()


def main():
    args = parse_args()

    distributed_gptdata_streaming_test(
        sequence_length=args.sequence_length,
        micro_batch_size=args.micro_batch_size,
        batch_size=args.batch_size,
        tensor_parallel=args.tensor_parallel,
        pipeline_parallel=args.pipeline_parallel,
        sequence_data_parallel=args.sequence_data_parallel,
        total_gpus=args.total_gpus,
        redis_port=args.redis_port,
        result_path=args.result_path,
        ingestion_type=IngestionType(args.ingestion_type),
    )


if __name__ == "__main__":
    main()
