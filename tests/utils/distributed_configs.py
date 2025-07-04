import dataclasses
import logging

from tests.utils.compare_tensor_logs import CompareConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class DistributedTestingConfig:
    name: str
    compare: str | None = None
    config_args: list[str]
    num_gpus: int = 1
    compare_config: CompareConfig | None = None


# Baseline (also tests data-parallel workers)
SIMPLE_TESTING_CONFIG = DistributedTestingConfig(
    name="simple",
    compare=None,
    config_args=["training.num_workers=2"],
    num_gpus=1,
)

_SINGLE_GPU_TESTING_CONFIGS = [
    # Sequence-first baseline
    DistributedTestingConfig(
        name="sf",
        compare=None,
        config_args=["model.base_model.sequence_first=True"],
        num_gpus=1,
    ),
    # Cross-entropy splits.
    DistributedTestingConfig(
        name="ce4",
        compare=None,
        config_args=["model.base_model.cross_entropy_splits=4"],
        num_gpus=1,
    ),
    # Micro-sequence baseline
    DistributedTestingConfig(
        name="ms",
        compare=None,
        config_args=["batch.micro_sequence_length=256"],
        num_gpus=1,
    ),
    # Gradient accumulation baseline.
    DistributedTestingConfig(
        name="df4",
        compare=None,
        config_args=["batch.depth_first_micro_batches=4"],
        num_gpus=1,
    ),
    # Breadth-first gradient accumulation.
    DistributedTestingConfig(
        name="bf4",
        compare="df4",
        config_args=["batch.breadth_first_micro_batches=4"],
        num_gpus=1,
    ),
    # Mixed gradient accumulation.
    DistributedTestingConfig(
        name="bf2_df2",
        compare="df4",
        config_args=["batch.depth_first_micro_batches=2", "batch.breadth_first_micro_batches=2"],
        num_gpus=1,
    ),
    # Sequence-first gradient accumulation baseline.
    DistributedTestingConfig(
        name="df4_sf",
        compare=None,
        config_args=["batch.depth_first_micro_batches=4", "model.base_model.sequence_first=True"],
        num_gpus=1,
    ),
]

SINGLE_GPU_TESTING_CONFIGS = {config.name: config for config in _SINGLE_GPU_TESTING_CONFIGS}


_DISTRIBUTED_TESTING_CONFIGS = [
    # ===== Data-parallel configs
    # Simple
    DistributedTestingConfig(
        name="dp2",
        compare="simple",
        config_args=[],
        num_gpus=2,
    ),
    # Zero stage 2
    DistributedTestingConfig(
        name="dp2_z2",
        compare="simple",
        config_args=["model.multi_stage.zero_stage=2"],
        num_gpus=2,
    ),
    # Zero stage 3
    DistributedTestingConfig(
        name="dp2_z3",
        compare="simple",
        config_args=["model.multi_stage.zero_stage=3"],
        num_gpus=2,
    ),
    # Depth-first micro-batches
    DistributedTestingConfig(
        name="dp2_z3_df4",
        compare="df4",
        config_args=["model.multi_stage.zero_stage=3", "batch.depth_first_micro_batches=4"],
        num_gpus=2,
        compare_config=CompareConfig(
            ignore_duplicates=[
                "Global gradient",
            ]
        ),
    ),
    # Sequence-data-parallel
    DistributedTestingConfig(
        name="sdp2",
        compare="sf",
        config_args=["model.distributed.sequence_data_parallel=2"],
        num_gpus=2,
    ),
    # ===== Tensor-parallel configs
    # Simple tensor-parallel
    DistributedTestingConfig(
        name="tp2",
        compare="simple",
        config_args=["model.distributed.tensor_parallel=2"],
        num_gpus=2,
    ),
    # Simple sequence-tensor-parallel
    DistributedTestingConfig(
        name="stp2",
        compare="sf",
        config_args=["model.distributed.tensor_parallel=2", "model.distributed.sequence_tensor_parallel=True"],
        num_gpus=2,
    ),
    # Cross-entropy splits
    DistributedTestingConfig(
        name="stp2_ce4",
        compare="sf",
        config_args=[
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "model.base_model.parallel_embeddings=False",
            "model.base_model.cross_entropy_splits=4",
        ],
        num_gpus=2,
    ),
    # ===== 2d configs (Data + Tensor)
    # Simple
    DistributedTestingConfig(
        name="dp2_stp2",
        compare="sf",
        config_args=[
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
        ],
        num_gpus=4,
    ),
    # Depth-first micro-batches, tensor-parallel
    DistributedTestingConfig(
        name="tp2_df4",
        compare="df4",
        config_args=[
            "model.distributed.tensor_parallel=2",
            "batch.depth_first_micro_batches=4",
        ],
        num_gpus=4,
    ),
    # Breadth-first micro-batches
    DistributedTestingConfig(
        name="sdp2_stp2_bf4",
        compare="df4_sf",
        config_args=[
            "model.distributed.sequence_data_parallel=2",
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "batch.breadth_first_micro_batches=4",
        ],
        num_gpus=4,
    ),
    # Sequence-data-parallel
    DistributedTestingConfig(
        name="sdp2_stp2",
        compare="sf",
        config_args=[
            "model.distributed.sequence_data_parallel=2",
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
        ],
        num_gpus=4,
    ),
    # ===== Pipeline-parallel configs
    # Simple [mb]
    DistributedTestingConfig(
        name="pp2s2_bf4",
        compare="df4",
        config_args=[
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=2",
            "batch.breadth_first_micro_batches=4",
        ],
        num_gpus=2,
    ),
    # Tied weights on different ranks
    DistributedTestingConfig(
        name="pp2s1_bf4",
        compare="df4",
        config_args=[
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=1",
            "batch.breadth_first_micro_batches=4",
        ],
        num_gpus=2,
        compare_config=CompareConfig(
            ignore_duplicates=[
                "layers.0.word_embeddings_weight",
                "layers.0.position_embeddings_weight",
            ]
        ),
    ),
    # Micro-sequence [ms]
    DistributedTestingConfig(
        name="pp2s2_ms",
        compare="ms",
        config_args=[
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=2",
            "batch.micro_sequence_length=256",
        ],
        num_gpus=2,
    ),
    # ===== 2d configs (Data + Pipeline)
    # Simple
    DistributedTestingConfig(
        name="dp2_pp2s2_bf4",
        compare="df4",
        config_args=[
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=2",
            "batch.breadth_first_micro_batches=4",
        ],
        num_gpus=4,
    ),
    # ===== 2d configs (Tensor + Pipeline)
    # Simple [sf, mb]
    DistributedTestingConfig(
        name="stp2_pp2s1_bf4",
        compare="df4_sf",
        config_args=[
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=2",
            "batch.breadth_first_micro_batches=4",
        ],
        num_gpus=4,
        compare_config=CompareConfig(
            ignore_duplicates=[
                "layers.0.word_embeddings_weight",
                "layers.0.position_embeddings_weight",
            ]
        ),
    ),
    # ===== Data + Tensor + Pipeline
    # Simple
    DistributedTestingConfig(
        name="dp2_stp2_pp2s2",
        compare="mb",
        config_args=[
            "model.distributed.tensor_parallel=2",
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=2",
            "batch.breadth_first_micro_batches=4",
        ],
        num_gpus=8,
    ),
    # Tied weights on different ranks
    DistributedTestingConfig(
        name="dp2_tp2_pp2s1_bf4",
        compare="mb",
        config_args=[
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=1",
            "batch.breadth_first_micro_batches=4",
        ],
        num_gpus=8,
        compare_config=CompareConfig(
            ignore_duplicates=[
                "layers.0.word_embeddings_weight",
                "layers.0.position_embeddings_weight",
            ]
        ),
    ),
    # Micro-sequence
    DistributedTestingConfig(
        name="sdp2_stp2_pp2s2_ms",
        compare="ms",
        config_args=[
            "model.distributed.sequence_data_parallel=2",
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=2",
            "batch.micro_sequence_length=256",
        ],
        num_gpus=8,
    ),
]

DISTRIBUTED_TESTING_CONFIGS = {config.name: config for config in _DISTRIBUTED_TESTING_CONFIGS}
