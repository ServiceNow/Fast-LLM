import collections
import pathlib
import subprocess

import pytest
import yaml

from fast_llm.config import NoAutoValidate
from fast_llm.data.dataset.gpt.config import GPTSamplingConfig
from fast_llm.engine.checkpoint.config import CheckpointSaveMetadataConfig, ModelConfigType
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDim, DistributedDimNames
from fast_llm.models.gpt.config import GPTModelConfig, GPTTrainerConfig, PretrainedGPTModelConfig
from fast_llm.utils import Assert, check_equal_nested


def run_without_import(cmd: str):
    # Make sure validation imports only the bare minimum.
    # Run the test in a separate process since lots of things are already imported in this one.
    repo_path = pathlib.Path(__file__).parents[1].resolve()
    command = [
        "python3",
        "-c",
        "\n".join(
            [
                # Import required third party libraries here, so they can be found later.
                "import sys, yaml, requests, packaging.version",
                # Prevent any other third party package from being imported (or at least try to)
                "sys.path=[p for p in sys.path if not any(x in p for x in ('site-packages', 'dist-packages', '.egg'))]",
                # We still want to enable imports from within Fast-llm
                f"sys.path.append('{repo_path}')",
                "from fast_llm.cli import fast_llm_main as main",
                cmd,
            ]
        ),
    ]

    completed_proc = subprocess.run(command)
    if completed_proc.returncode:
        raise RuntimeError(f"Process failed with return code {completed_proc.returncode}")


def test_validate_train_gpt_without_import():
    run_without_import("main(['train', 'gpt', '-v'])")


def test_validate_prepare_gpt_memmap_without_import():
    run_without_import(
        "main(['prepare', 'gpt_memmap', '-v', 'dataset.path=test', 'output_path=test', 'tokenizer.path=test'])"
    )


def test_validate_convert_gpt_without_import():
    run_without_import("main(['convert', 'gpt', '-v'])")


def test_validate_example_config():
    fast_llm_config_dict = yaml.safe_load(
        (pathlib.Path(__file__).parents[1] / "examples" / "mistral.yaml").read_text()
    )
    GPTTrainerConfig.from_dict(fast_llm_config_dict)


@pytest.mark.parametrize("cls", (GPTSamplingConfig, GPTModelConfig))
def test_serialize_default_config_updates(cls):
    # Config classes used as config updates should have a default that serializes to an empty dict
    #   so no value is incorrectly overridden.
    with NoAutoValidate():
        check_equal_nested(cls.from_dict({}).to_dict(), {})


@pytest.mark.parametrize("load_config", tuple(ModelConfigType))
def test_pretrained_config(load_config: ModelConfigType, result_path):
    config_path = result_path / "pretrained_config"
    pretrained_model_config = GPTModelConfig.from_dict(
        {
            "base_model": {
                "embeddings_layer": {
                    "hidden_size": 1024,  # Default
                },
                "decoder": {
                    "block": {
                        "mixer": {
                            "rotary": {"type": "default"},
                            "window_size": 32,
                            "head_groups": 4,
                        },
                        "mlp": {
                            "intermediate_size": 4096,  # Implicit default, default value
                            "activation": "silu",  # Implicit default, non-default value
                        },
                        "normalization": {"type": "rms_norm"},  # Nested
                    },
                    "num_blocks": 12,  # Default
                },
                "output_layer": {"tied_weight": False},
            },
            "multi_stage": {"zero_stage": 3},
            "distributed": {"compute_dtype": "bfloat16"},
        }
    )
    with NoAutoValidate():
        save_config = CheckpointSaveMetadataConfig.from_dict({"format": "fast_llm", "path": config_path})
    save_config.setup(GPTModelConfig)
    save_config.validate()
    pretrained_model_config.save_metadata(save_config)

    base_model_update = {
        "embeddings_layer": {"hidden_size": 512, "vocab_size": 1000},
        "decoder": {
            "block": {
                "mixer": {
                    "head_groups": 1,  # Override to default
                },
                # rotary: Don't override nested.
                "normalization": {"implementation": "triton"},  # Update non-default nested
            },
        },
        "peft": {"type": "lora", "freeze_others": False},  # Update default nested, change type
    }
    pretrained_config = PretrainedGPTModelConfig.from_dict(
        {
            "model": {
                "base_model": base_model_update,
                "distributed": {"seed": 1234, "compute_dtype": "float16"},
            },
            "pretrained": {"format": "fast_llm", "path": config_path, "load_config": load_config},
        }
    )
    serialized_config = pretrained_config.model.to_dict()
    expected_config = {"type": "gpt", "distributed": DistributedConfig().to_dict()}

    if load_config == ModelConfigType.fast_llm:
        expected_config["multi_stage"] = {"zero_stage": 3}
    expected_config["distributed"].update({"seed": 1234, "compute_dtype": "float16"})
    if load_config in (ModelConfigType.fast_llm, ModelConfigType.model):
        expected_config["base_model"] = {
            "embeddings_layer": {
                "hidden_size": 512,
                "vocab_size": 1000,
            },
            "decoder": {
                "type": "fixed",
                "block": {
                    "type": "decoder",
                    "mixer": {
                        "type": "attention",
                        "rotary": {"type": "default"},
                        "window_size": 32,
                        "head_groups": 1,
                    },
                    "mlp": {
                        "type": "mlp",
                        "intermediate_size": 4096,  # Implicit default, default value
                        "activation": "silu",  # Implicit default, non-default value
                    },
                    "normalization": {"type": "rms_norm", "implementation": "triton"},
                },
                "num_blocks": 12,
            },
            "output_layer": {"tied_weight": False, "normalization": {"type": "layer_norm"}},
            "peft": {"type": "lora", "freeze_others": False},
        }
    else:
        base_model_update["decoder"]["type"] = "fixed"
        base_model_update["decoder"]["block"]["type"] = "decoder"
        base_model_update["decoder"]["block"]["normalization"]["type"] = "layer_norm"
        base_model_update["decoder"]["block"]["mixer"]["type"] = "attention"
        base_model_update["decoder"]["block"]["mixer"]["rotary"] = {"type": "none"}
        base_model_update["decoder"]["block"]["mlp"] = {"type": "mlp"}
        base_model_update["output_layer"] = {"normalization": {"type": "layer_norm"}}
        base_model_update["peft"] = {"type": "lora", "freeze_others": False}
        expected_config["base_model"] = base_model_update

    check_equal_nested(serialized_config, expected_config)


def _check_dim(dim: DistributedDim, name: str, rank: int, size: int, global_rank: int):
    Assert.eq(dim.name, name)
    Assert.eq(dim.size, size)
    Assert.eq(dim.rank, rank)
    # Already checked in distributed config, we repeat for extra safety.
    Assert.eq(dim.global_ranks[rank], global_rank)
    Assert.eq(len(dim.global_ranks), size)


@pytest.mark.parametrize(
    ("bdp", "sdp", "tp", "pp", "pipeline_first"),
    (
        (1, 1, 1, 1, False),
        (4, 1, 1, 1, False),
        (1, 4, 1, 1, False),
        (1, 1, 4, 1, False),
        (1, 1, 1, 4, False),
        (1, 4, 1, 3, False),
        (1, 1, 3, 2, False),
        (1, 1, 3, 2, True),
        (3, 1, 1, 2, False),
        (3, 1, 1, 2, True),
        (2, 2, 2, 3, False),
    ),
)
def test_distributed_global_ranks(bdp: int, sdp: int, tp: int, pp: int, pipeline_first: bool):
    world_size = bdp * sdp * tp * pp
    dp = sdp * bdp
    config_dict = {
        "sequence_data_parallel": sdp,
        "tensor_parallel": tp,
        "pipeline_parallel": pp,
        "pipeline_first": pipeline_first,
        "world_size": world_size,
        "local_world_size": world_size,
    }

    all_global_ranks = collections.defaultdict(set)
    rank_breakdowns = set()
    for rank in range(world_size):
        # Independent computation of the group ranks.
        tp_rank = rank % tp
        rank_ = rank // tp
        if pipeline_first:
            pp_rank = rank_ % pp
            dp_rank = rank_ // pp
        else:
            dp_rank = rank_ % dp
            pp_rank = rank_ // dp

        config = DistributedConfig.from_dict(config_dict, {"rank": rank})
        # Check that each group has the right size and rank.
        _check_dim(
            world_dim := config.get_distributed_dim(DistributedDimNames.world),
            DistributedDimNames.world,
            rank,
            world_size,
            rank,
        )
        _check_dim(
            tp_dim := config.get_distributed_dim(DistributedDimNames.tensor),
            DistributedDimNames.tensor,
            tp_rank,
            tp,
            rank,
        )
        _check_dim(
            tp_sdp_dim := config.get_distributed_dim(DistributedDimNames.tensor_and_sequence_data),
            DistributedDimNames.tensor_and_sequence_data,
            dp_rank % sdp * tp + tp_rank,
            tp * sdp,
            rank,
        )
        _check_dim(
            sdp_dim := config.get_distributed_dim(DistributedDimNames.sequence_data),
            DistributedDimNames.sequence_data,
            dp_rank % sdp,
            sdp,
            rank,
        )
        _check_dim(
            bdp_dim := config.get_distributed_dim(DistributedDimNames.batch_data),
            DistributedDimNames.batch_data,
            dp_rank // sdp,
            bdp,
            rank,
        )
        _check_dim(
            dp_dim := config.get_distributed_dim(DistributedDimNames.data),
            DistributedDimNames.data,
            dp_rank,
            bdp * sdp,
            rank,
        )
        _check_dim(
            pp_dim := config.get_distributed_dim(DistributedDimNames.pipeline),
            DistributedDimNames.pipeline,
            pp_rank,
            pp,
            rank,
        )
        all_global_ranks["world"].add(tuple(world_dim.global_ranks))
        all_global_ranks["tp"].add(tuple(tp_dim.global_ranks))
        all_global_ranks["tp_sdp"].add(tuple(tp_sdp_dim.global_ranks))
        all_global_ranks["sdp"].add(tuple(sdp_dim.global_ranks))
        all_global_ranks["bdp"].add(tuple(bdp_dim.global_ranks))
        all_global_ranks["dp"].add(tuple(dp_dim.global_ranks))
        all_global_ranks["pp"].add(tuple(pp_dim.global_ranks))
        rank_breakdowns.add((tp_rank, dp_rank // sdp, dp_rank % sdp, pp_rank))

    for name, global_ranks_set in all_global_ranks.items():
        # Check that the global ranks are partitioned into disjoint groups for each distributed dimension,
        # and indirectly that `DistributedDim.global_ranks` is consistent between ranks.
        Assert.eq(sum(len(global_ranks) for global_ranks in global_ranks_set), world_size)
        Assert.eq(len({global_rank for global_ranks in global_ranks_set for global_rank in global_ranks}), world_size)

    Assert.eq(len(rank_breakdowns), world_size)
