import copy
import dataclasses
import logging

import torch

from tests.utils.compare_tensor_logs import CompareConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class DistributedTestingConfig:
    name: str
    compare: str | None = None
    config_args: list[str]
    num_gpus: int = 1
    compare_config: CompareConfig | None = None
    # Scale the comparison thresholds for specific distributed configs.
    compare_factor: float = 1.0


def get_config(relative: float = 0, absolute: float = 0, **kwargs) -> CompareConfig:
    return CompareConfig(
        rms_rel_tolerance=relative,
        max_rel_tolerance=relative * 10,
        rms_abs_tolerance=absolute,
        max_abs_tolerance=absolute * 10,
        rms_eps=absolute / 10,
        **kwargs,
    )


# TODO: Ajust
_compare_layer_match = get_config(
    sub_configs={
        ("init", None): get_config(),
        (None, "fw"): get_config(1e-3, 1e-4),
        (None, "bw"): get_config(3e-3, 1e-5),
        # Biases have higher absolute error.
        (None, "bias"): get_config(3e-3, 5e-5),
        (None, "gradient"): get_config(3e-3, 3e-5),
    }
)

_compare_layer_mismatch = copy.deepcopy(_compare_layer_match)
for tensor in ("fw", "bw"):
    _compare_layer_mismatch.sub_configs[(None, tensor)].ignore_tensors = True
_pp_tied_weight_compare = copy.deepcopy(_compare_layer_mismatch)
_compare_layer_match_duplicate_gradients = copy.deepcopy(_compare_layer_match)
_compare_layer_match_duplicate_gradients.sub_configs[(None, "bias")].ignore_duplicates = True
_compare_layer_match_duplicate_gradients.sub_configs[(None, "gradient")].ignore_duplicates = True
_compare_layer_mismatch_duplicate_gradients = copy.deepcopy(_compare_layer_mismatch)
_compare_layer_mismatch_duplicate_gradients.sub_configs[(None, "bias")].ignore_duplicates = True
_compare_layer_mismatch_duplicate_gradients.sub_configs[(None, "gradient")].ignore_duplicates = True
_pp_tied_weight_compare.sub_configs[(None, "gradient")].ignore_duplicates = True
_pp_tied_weight_compare.sub_configs[("init", None)].ignore_duplicates = True
for tensor in ("fw", "bw"):
    _pp_tied_weight_compare.sub_configs[(None, tensor)].ignore_duplicates = True


_bf16_compare = get_config(
    sub_configs={
        ("init", None): get_config(),
        (None, "fw"): get_config(1.5e-2, 1.5e-3),
        (None, "bw"): get_config(1.5e-2, 1e-5),
        # TODO: Normalization gradient broken on CPU, getting inconsistent results across machines.
        **(
            {}
            if torch.cuda.is_available()
            else {
                (None, "norm"): get_config(ignore_tensors=True),
                (None, "word_embeddings_weight"): get_config(8e-2, 1e-4),
            }
        ),
        (None, "bias"): get_config(2e-2, 1e-3) if torch.cuda.is_available() else get_config(2e-2, 2e-3),
        (None, "gradient"): get_config(2e-2, 5e-5) if torch.cuda.is_available() else get_config(2e-2, 1e-4),
    }
)

_fp16_compare = get_config(
    sub_configs={
        ("init", None): get_config(),
        # Saved gradient include the gradient scaling by 2**16 (default initial value)
        (None, "fw"): get_config(1.2e-3, 3e-4),
        (None, "bw"): get_config(3e-3, 1e-5, scale=2**16),
        # TODO: Normalization gradient broken on CPU, getting inconsistent results across machines.
        **(
            {}
            if torch.cuda.is_available()
            else {
                (None, "norm"): get_config(ignore_tensors=True),
                (None, "word_embeddings_weight"): get_config(2e-2, 1e-4, scale=2**16),
            }
        ),
        (None, "bias"): (
            get_config(3e-3, 1e-4, scale=2**16) if torch.cuda.is_available() else get_config(6e-3, 2e-4, scale=2**16)
        ),
        (None, "gradient"): (
            get_config(3e-3, 5e-5, scale=2**16) if torch.cuda.is_available() else get_config(6e-3, 1e-4, scale=2**16)
        ),
    }
)


# Simple case
# TODO: ====== Backup attn takes too much memory with 4k tokens.
SIMPLE_TESTING_CONFIG = DistributedTestingConfig(
    name="simple",
    compare=None,
    config_args=["data.micro_batch_size=4096"],
    num_gpus=1,
)

_SINGLE_GPU_TESTING_CONFIGS = [
    DistributedTestingConfig(
        name="bf16",
        compare="simple",
        # Also tests parallel data loader.
        config_args=[
            "model.distributed.compute_dtype=bf16",
            "training.num_workers=1",
            "data.micro_batch_size=4096",
        ],
        num_gpus=1,
        compare_config=_bf16_compare,
    ),
    DistributedTestingConfig(
        name="fp16",
        compare="simple",
        config_args=["model.distributed.compute_dtype=fp16", "data.micro_batch_size=4096"],
        num_gpus=1,
        compare_config=_fp16_compare,
    ),
    # Cross-entropy splits.
    DistributedTestingConfig(
        name="ce4",
        compare="simple",
        config_args=["model.base_model.head.cross_entropy_splits=4", "data.micro_batch_size=4096"],
        num_gpus=1,
        compare_config=_compare_layer_mismatch,
    ),
    # Micro-sequence baseline
    DistributedTestingConfig(
        name="ms4",
        compare="simple",
        config_args=["schedule.micro_batch_splits=4", "data.micro_batch_size=4096"],
        num_gpus=1,
        compare_config=_compare_layer_mismatch,
    ),
    # Gradient accumulation baselines.
    DistributedTestingConfig(
        name="df2",
        config_args=["schedule.depth_first_micro_batches=2", "data.micro_batch_size=2048"],
        num_gpus=1,
    ),
    DistributedTestingConfig(
        name="df4",
        config_args=["schedule.depth_first_micro_batches=4", "data.micro_batch_size=1024"],
        num_gpus=1,
    ),
    DistributedTestingConfig(
        name="df8",
        config_args=["schedule.depth_first_micro_batches=8", "data.micro_batch_size=512"],
        num_gpus=1,
    ),
    # Breadth-first gradient accumulation.
    DistributedTestingConfig(
        name="bf4",
        compare="df4",
        config_args=["schedule.breadth_first_micro_batches=4", "data.micro_batch_size=1024"],
        num_gpus=1,
        compare_config=_compare_layer_match,
    ),
    # Mixed gradient accumulation.
    DistributedTestingConfig(
        name="bf2_df2",
        compare="df4",
        config_args=[
            "schedule.depth_first_micro_batches=2",
            "schedule.breadth_first_micro_batches=2",
            "data.micro_batch_size=1024",
        ],
        num_gpus=1,
        compare_config=_compare_layer_match,
    ),
]

SINGLE_GPU_TESTING_CONFIGS = {config.name: config for config in _SINGLE_GPU_TESTING_CONFIGS}


_DISTRIBUTED_TESTING_CONFIGS = [
    # ===== Data-parallel configs
    # Simple
    DistributedTestingConfig(
        name="dp2",
        compare="df2",
        config_args=["data.micro_batch_size=2048"],
        num_gpus=2,
        # TODO: layer outputs are the same but logged differently.
        compare_config=_compare_layer_mismatch,
    ),
    # Zero stage 2
    DistributedTestingConfig(
        name="dp2_z2",
        compare="dp2",
        config_args=["model.multi_stage.zero_stage=2", "data.micro_batch_size=2048"],
        num_gpus=2,
        compare_config=_compare_layer_match,
    ),
    # Zero stage 3
    DistributedTestingConfig(
        name="dp2_z3",
        compare="dp2",
        config_args=["model.multi_stage.zero_stage=3", "data.micro_batch_size=2048"],
        num_gpus=2,
        compare_config=_compare_layer_match,
    ),
    # Depth-first micro-batches
    DistributedTestingConfig(
        name="dp2_z2_df4",
        compare="df8",
        config_args=[
            "model.multi_stage.zero_stage=2",
            "schedule.depth_first_micro_batches=4",
            "data.micro_batch_size=512",
        ],
        num_gpus=2,
        compare_config=_compare_layer_mismatch_duplicate_gradients,
    ),
    # Sequence-data-parallel
    DistributedTestingConfig(
        name="sdp2",
        compare="simple",
        config_args=["model.distributed.sequence_data_parallel=2", "data.micro_batch_size=4096"],
        num_gpus=2,
        compare_config=_compare_layer_match,
    ),
    # ===== Tensor-parallel configs
    # Simple tensor-parallel
    DistributedTestingConfig(
        name="tp2",
        compare="simple",
        config_args=["model.distributed.tensor_parallel=2", "data.micro_batch_size=4096"],
        num_gpus=2,
        compare_config=_compare_layer_match,
    ),
    # Simple sequence-tensor-parallel
    DistributedTestingConfig(
        name="stp2",
        compare="simple",
        config_args=[
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "data.micro_batch_size=4096",
        ],
        num_gpus=2,
        compare_config=_compare_layer_match,
    ),
    # Depth-first micro-batches, tensor-parallel
    DistributedTestingConfig(
        name="tp2_df4",
        compare="df4",
        config_args=[
            "model.distributed.tensor_parallel=2",
            "schedule.depth_first_micro_batches=4",
            "data.micro_batch_size=1024",
        ],
        num_gpus=2,
        compare_config=_compare_layer_match,
    ),
    # Cross-entropy splits
    DistributedTestingConfig(
        name="stp2_ce4",
        compare="simple",
        config_args=[
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "model.base_model.embeddings.vocab_parallel=False",
            "model.base_model.head.cross_entropy_splits=4",
            "data.micro_batch_size=4096",
        ],
        num_gpus=2,
        compare_config=_compare_layer_match,
    ),
    # ===== 2d configs (Data + Tensor)
    # Simple
    DistributedTestingConfig(
        name="dp2_stp2",
        compare="dp2",
        config_args=[
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "data.micro_batch_size=2048",
        ],
        num_gpus=4,
        compare_config=_compare_layer_match,
    ),
    # Breadth-first micro-batches
    DistributedTestingConfig(
        name="sdp2_stp2_bf4",
        compare="df4",
        config_args=[
            "model.distributed.sequence_data_parallel=2",
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "schedule.breadth_first_micro_batches=4",
            "data.micro_batch_size=512",
        ],
        num_gpus=4,
        compare_config=_compare_layer_mismatch,
    ),
    # Sequence-data-parallel
    DistributedTestingConfig(
        name="sdp2_stp2",
        compare="simple",
        config_args=[
            "model.distributed.sequence_data_parallel=2",
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "data.micro_batch_size=4096",
        ],
        num_gpus=4,
        compare_config=_compare_layer_match,
    ),
    # ===== Pipeline-parallel configs
    # Simple [mb]
    DistributedTestingConfig(
        name="pp2s2_bf4",
        compare="df4",
        config_args=[
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=2",
            "schedule.breadth_first_micro_batches=4",
            "data.micro_batch_size=1024",
        ],
        num_gpus=2,
        compare_config=_compare_layer_match,
    ),
    # Tied weights on different ranks
    DistributedTestingConfig(
        name="pp2s1_bf4",
        compare="df4",
        config_args=[
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=1",
            "schedule.breadth_first_micro_batches=4",
            "data.micro_batch_size=1024",
        ],
        num_gpus=2,
        compare_config=_pp_tied_weight_compare,
    ),
    # Micro-sequence [ms]
    DistributedTestingConfig(
        name="pp2s2_ms4",
        compare="ms4",
        config_args=[
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=2",
            "schedule.micro_batch_splits=4",
            "data.micro_batch_size=4096",
        ],
        num_gpus=2,
        compare_config=_compare_layer_match,
    ),
    # ===== 2d configs (Data + Pipeline)
    # Simple
    DistributedTestingConfig(
        name="dp2_pp2s2_bf4",
        compare="dp2_z2_df4",
        config_args=[
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=2",
            "schedule.breadth_first_micro_batches=4",
            "data.micro_batch_size=512",
        ],
        num_gpus=4,
        compare_config=_compare_layer_match_duplicate_gradients,
    ),
    # ===== 2d configs (Tensor + Pipeline)
    # Simple [mb]
    DistributedTestingConfig(
        name="stp2_pp2s1_bf4",
        compare="df4",
        config_args=[
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=2",
            "schedule.breadth_first_micro_batches=4",
            "data.micro_batch_size=1024",
        ],
        num_gpus=4,
        compare_config=_pp_tied_weight_compare,
    ),
    # ===== Data + Tensor + Pipeline
    # Simple
    DistributedTestingConfig(
        name="dp2_stp2_pp2s2_bf4",
        compare="dp2_z2_df4",
        config_args=[
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=2",
            "schedule.breadth_first_micro_batches=4",
            "data.micro_batch_size=412",
        ],
        num_gpus=8,
        compare_config=_compare_layer_match,
    ),
    # Tied weights on different ranks
    DistributedTestingConfig(
        name="dp2_tp2_pp2s1_bf4",
        compare="dp2_z2_df4",
        config_args=[
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=1",
            "schedule.breadth_first_micro_batches=4",
            "data.micro_batch_size=512",
        ],
        num_gpus=8,
        compare_config=_pp_tied_weight_compare,
    ),
    # Micro-sequence
    DistributedTestingConfig(
        name="sdp2_stp2_pp2s2_ms4",
        compare="df2",
        config_args=[
            "model.distributed.sequence_data_parallel=2",
            "model.distributed.tensor_parallel=2",
            "model.distributed.sequence_tensor_parallel=True",
            "model.distributed.pipeline_parallel=2",
            "model.multi_stage.layers_per_stage=2",
            "schedule.micro_batch_splits=4",
            "data.micro_batch_size=2048",
        ],
        num_gpus=8,
        compare_config=_compare_layer_mismatch,
    ),
]

DISTRIBUTED_TESTING_CONFIGS = {config.name: config for config in _DISTRIBUTED_TESTING_CONFIGS}
