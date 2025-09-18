import copy
import dataclasses
import enum
import functools
import os
import typing

import pytest

from fast_llm.config import set_nested_dict_value
from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.models.gpt.conversion.config import (
    AprielHybridSSMCheckpointFormat,
    DiffusionDreamCheckpointFormat,
    DiffusionLlamaCheckpointFormat,
    LlamaCheckpointFormat,
    MistralCheckpointFormat,
    MixtralCheckpointFormat,
    MTPLlamaCheckpointFormat,
    Qwen2CheckpointFormat,
)
from tests.utils.distributed_configs import DistributedTestingConfig
from tests.utils.global_variables import MODEL_DATASET_PREFIX, MODEL_TEST_VOCAB_SIZE

from fast_llm.engine.evaluation.evaluators import (  # isort:skip  # needed for dynamic type registration
    EvaluatorsConfig,
)

_LOG_LEVEL = int(os.environ.get("LOG_LEVEL", 13))


class ModelTestingGroup(enum.StrEnum):
    basic = "basic"
    checkpoint = "checkpoint"
    convert = "convert"
    generate = "generate"
    megatron = "megatron"
    distributed = "distributed"


class ModelTestingGroupAction(enum.StrEnum):
    # Critical test, will always run.
    main = "main"
    # Standard test, treated as slow
    normal = "normal"
    # Feature is not important enough for frequent testing (ex. mostly redundant), treated as extra-slow.
    unimportant = "unimportant"
    # Test is known to fail, treated as extra-slow.
    broken = "broken"
    # Tested feature is unsupported for this model, skip unconditionally.
    not_implemented = "not_implemented"


def _config_dict_to_args(config_dict: dict[str, typing.Any], keys=()):
    """
    Converts a config dict to cli arguments. Not generic but good enough for the tests.
    """
    args = []
    for key, value in config_dict.items():
        if isinstance(value, dict):
            args += _config_dict_to_args(value, (*keys, key))
        else:
            args.append(f"{'.'.join((*keys, key))}={value}")
    return args


@dataclasses.dataclass(kw_only=True, frozen=True)
class ModelTestingConfig:
    name: str = None
    model_type: str
    config_dict: dict[str, typing.Any]
    megatron_args: list[str] | None
    checkpoint_format: type[CheckpointFormat] | None
    groups: dict[ModelTestingGroup, ModelTestingGroupAction]
    # Scale the comparison thresholds for specific models.
    compare_factor: float = 1.0
    # Option to skip specific distributed configuration with name containing any of the provided strings.
    skip_tests: tuple[str] = ()

    @functools.cached_property
    def config_args(self):
        return _config_dict_to_args(self.config_dict)

    @functools.cached_property
    def trainer_config_class(self) -> type[TrainerConfig]:
        return TrainerConfig.get_subclass(self.model_type)

    @functools.cached_property
    def trainer_config(self) -> TrainerConfig:
        # See `RunnableConfig._from_parsed_args`
        return self.trainer_config_class.from_dict(self.config_dict)

    @functools.cached_property
    def evaluators_config_class(self) -> type[EvaluatorsConfig]:
        # EvaluatorsConfig is a base class that, during parse_and_run, replaces itself with the appropriate TrainingConfig subclass.
        # Therefore, the arguments passed to EvaluatorsConfig.parse_and_run must include the model type as the first element.
        return EvaluatorsConfig

    @functools.cached_property
    def evaluators_config(self) -> EvaluatorsConfig:
        # See `RunnableConfig._from_parsed_args`
        return self.evaluators_config_class.from_dict(self.config_dict)

    @functools.cached_property
    def model_config_class(self) -> type[FastLLMModelConfig]:
        # TODO: Ok to assume the model and trainer have the same name?
        return FastLLMModelConfig.get_subclass(self.model_type)

    @functools.cached_property
    def model_config(self) -> FastLLMModelConfig:
        return self.trainer_config.model

    @functools.cached_property
    def huggingface_model_for_causal_lm_class(self):
        return self.model_config_class.get_huggingface_model_for_causal_lm_class()

    @functools.cached_property
    def model_class(self):
        return self.model_config_class.get_model_class()

    @functools.cached_property
    def base_model_config_class(self):
        return self.model_config_class.get_base_model_config_class()

    def should_skip(self, distributed_config: DistributedTestingConfig) -> bool:
        return any(key in distributed_config.name for key in self.skip_tests)


def _update_and_add_testing_config(
    old_name: str,
    new_name: str,
    *,
    model_type: str | None = None,
    updates: dict[str | tuple[str, ...], typing.Any] | None = None,
    megatron_args: list[str] | None = ...,
    groups: dict[ModelTestingGroup, ModelTestingGroupAction],
    **kwargs,
) -> ModelTestingConfig:

    config = MODEL_CONFIGS[old_name]
    config_dict = copy.deepcopy(config.config_dict)
    if updates is not None:
        for keys, update in updates.items():
            set_nested_dict_value(config_dict, keys, update)
    if megatron_args is not ...:
        if megatron_args is None:
            megatron_args = None
        elif config.megatron_args is not None:
            megatron_args = config.megatron_args + megatron_args
    new_config = dataclasses.replace(
        config,
        name=new_name,
        model_type=config.model_type if model_type is None else model_type,
        groups=groups,
        config_dict=config_dict,
        megatron_args=megatron_args,
        **kwargs,
    )
    MODEL_CONFIGS[new_name] = new_config
    return new_config


MODEL_CONFIGS: dict[str, ModelTestingConfig] = {}

# We use a smaller initialization scheme than the default to lower variance in layer outputs during comparisons.
# This is as if we had a hidden size of 2048
init_1 = {"initialization": {"type": "normal", "std": 2**-5.5}}
# Needed to match Megatron (init_1 / (2 * num_layers) ** 0.5)
init_2 = {"initialization": {"type": "normal", "std": 2**-6.5}}

MODEL_CONFIGS["gpt_2"] = ModelTestingConfig(
    # Tests gpt2 features (absolute embeddings, layer norm,  relu activation, tied embeddings, MHA, linear biases).
    name="gpt_2",
    model_type="gpt",
    config_dict={
        "run": {
            "tensor_logs": {
                "save": True,
                "show": False,
            },
        },
        "training": {
            "logs": {"interval": 1},
            "train_iters": 2,
            "num_workers": 0,
            "timeout": 30,
        },
        "model": {
            "base_model": {
                "embeddings_layer": {
                    "word_embeddings": init_1,
                    "position_embeddings": {"enabled": True, **init_1},
                    "hidden_size": 256,
                    "num_position_embeddings": 512,
                    "vocab_size": MODEL_TEST_VOCAB_SIZE,
                },
                "decoder": {
                    "block": {
                        "mixer": {
                            "query_layer": {"weight": init_1},
                            "key_layer": {"weight": init_1},
                            "value_layer": {"weight": init_1},
                            "dense_layer": {"weight": init_2},
                            "heads": 8,
                            "head_groups": 8,
                            "head_size": 32,
                        },
                        "mlp": {
                            "layer_1": {"weight": init_1},
                            "layer_2": {"weight": init_2},
                            "intermediate_size": 1024,
                        },
                    },
                    "num_blocks": 2,
                },
                "output_layer": {"output_weight": init_1},
            },
            "multi_stage": {
                "debug_param_init": _LOG_LEVEL,
                "debug_layer_outputs": _LOG_LEVEL,
                "debug_layer_gradients": _LOG_LEVEL,
                "debug_all_param_gradients": _LOG_LEVEL,
                "debug_tensor_parallel": True,
            },
            "distributed": {
                "reproducible_init": True,
                "timeout": 20,
            },
        },
        "batch": {"batch_size": 8, "sequence_length": 512},
        "data": {
            "datasets": {
                "training": {
                    "dataset": {"type": "memmap", "path": MODEL_DATASET_PREFIX},
                    "type": "slice",
                    "end": 0.969,
                },
                "validation": {
                    "dataset": {"type": "memmap", "path": MODEL_DATASET_PREFIX},
                    "type": "slice",
                    "begin": 0.969,
                    "end": 0.999,
                },
                "test": {
                    "dataset": {"type": "memmap", "path": MODEL_DATASET_PREFIX},
                    "type": "slice",
                    "begin": 0.999,
                    "end": 1,
                },
            }
        },
        "optimizer": {"learning_rate": {"base": 0.0001}},
    },
    megatron_args=[
        "--num-layers=2",
        "--hidden-size=256",
        "--num-attention-heads=8",
        "--log-interval=1",
        "--train-iters=2",
        "--eval-iters=0",
        "--hidden-dropout=0",
        "--attention-dropout=0",
        f"--debug_param_init={_LOG_LEVEL}",
        f"--debug_layer_outputs={_LOG_LEVEL}",
        f"--debug_layer_gradients={_LOG_LEVEL}",
        f"--debug_all_param_gradients={_LOG_LEVEL}",
        "--debug_param_update=0",
        "--global-batch-size=8",
        "--micro-batch-size=8",
        "--max-position-embeddings=512",
        "--seq-length=512",
        f"--init-method-std={2**-5.5}",
        "--lr=0.0001",
        "--num-workers=0",
        "--valid-num-workers=0",
        "--tokenizer-type=NullTokenizer",
        # Megatron messes with the vocab size, so we have to subtract 1.
        f"--vocab-size={MODEL_TEST_VOCAB_SIZE - 1}",
        f"--data-path={MODEL_DATASET_PREFIX}",
        "--split=1,0,0",
        "--lr-decay-style=constant",
        # Initialization is set up to match MCore models (MCore inverts self-attn qkv and dense layers compared to original Megatron)
        "--use-mcore-models",
        # local implementation doesn't allow for RMS norm.
        "--transformer-impl=transformer_engine",
    ],
    checkpoint_format=None,
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.main,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.main,
        # TODO: PP checkpoint failing for tied weights.
        ModelTestingGroup.convert: ModelTestingGroupAction.broken,
        ModelTestingGroup.generate: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.megatron: ModelTestingGroupAction.normal,
        ModelTestingGroup.distributed: ModelTestingGroupAction.normal,
    },
)

_update_and_add_testing_config(
    # Tests MQA.
    "gpt_2",
    "starcoder",
    updates={
        ("model", "base_model", "decoder", "block", "mixer", "head_groups"): 1,
    },
    megatron_args=["--group-query-attention"],
    checkpoint_format=None,
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.normal,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.normal,
        ModelTestingGroup.convert: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.generate: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.megatron: ModelTestingGroupAction.unimportant,
        ModelTestingGroup.distributed: ModelTestingGroupAction.unimportant,
    },
)

_update_and_add_testing_config(
    # Tests intermediate between gpt2 and llama, closest converter to gpt2.
    "gpt_2",
    "starcoder_2",
    updates={
        ("model", "base_model", "decoder", "block", "mixer", "head_groups"): 4,
        ("model", "base_model", "decoder", "block", "mixer", "rotary", "type"): "default",
        ("model", "base_model", "embeddings_layer", "position_embeddings", "enabled"): False,
    },
    megatron_args=[
        "--group-query-attention",
        "--num-query-groups=4",
        "--use-rotary-position-embeddings",
        "--no-position-embedding",
    ],
    checkpoint_format=None,
    # TODO: Add back generate as `normal` when stable.
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.normal,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.normal,
        ModelTestingGroup.convert: ModelTestingGroupAction.normal,
        ModelTestingGroup.generate: ModelTestingGroupAction.broken,
        ModelTestingGroup.megatron: ModelTestingGroupAction.unimportant,
        ModelTestingGroup.distributed: ModelTestingGroupAction.unimportant,
    },
)

_update_and_add_testing_config(
    # Main tested model.
    "starcoder_2",
    "llama",
    updates={
        ("model", "base_model", "decoder", "block", "mixer", "add_linear_biases"): False,
        ("model", "base_model", "decoder", "block", "mlp", "gated"): True,
        ("model", "base_model", "decoder", "block", "mlp", "activation"): "silu",
        ("model", "base_model", "decoder", "block", "mlp", "add_linear_biases"): False,
        ("model", "base_model", "decoder", "block", "normalization", "type"): "rms_norm",
        ("model", "base_model", "output_layer", "normalization", "type"): "rms_norm",
        ("model", "base_model", "output_layer", "tied_weight"): False,
    },
    megatron_args=[
        "--swiglu",
        "--disable-bias-linear",
        "--normalization=RMSNorm",
        "--ffn-hidden-size=1024",
        "--untie-embeddings-and-output-weights",
    ],
    checkpoint_format=LlamaCheckpointFormat,
    # TODO: Add back generate as `normal` when stable.
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.main,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.main,
        ModelTestingGroup.convert: ModelTestingGroupAction.main,
        ModelTestingGroup.generate: ModelTestingGroupAction.broken,
        ModelTestingGroup.megatron: ModelTestingGroupAction.normal,
        ModelTestingGroup.distributed: ModelTestingGroupAction.normal,
    },
)

_update_and_add_testing_config(
    # Tests llama3-style rotary embeddings.
    "llama",
    "llama_3",
    updates={
        ("model", "base_model", "decoder", "block", "mixer", "rotary", "type"): "llama3",
    },
    # Megatron doesn't support Llama3-style Rotary Embeddings
    megatron_args=None,
    checkpoint_format=LlamaCheckpointFormat,
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.normal,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.normal,
        ModelTestingGroup.convert: ModelTestingGroupAction.unimportant,
        ModelTestingGroup.generate: ModelTestingGroupAction.unimportant,
        ModelTestingGroup.megatron: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.distributed: ModelTestingGroupAction.unimportant,
    },
)

_update_and_add_testing_config(
    # Tests yarn-style rotary embeddings.
    "llama",
    "llama_yarn",
    updates={
        ("model", "base_model", "decoder", "block", "mixer", "rotary", "type"): "yarn",
    },
    # Megatron doesn't support Yarn-style Rotary Embeddings
    megatron_args=None,
    checkpoint_format=LlamaCheckpointFormat,
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.normal,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.normal,
        ModelTestingGroup.convert: ModelTestingGroupAction.unimportant,
        ModelTestingGroup.generate: ModelTestingGroupAction.unimportant,
        ModelTestingGroup.megatron: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.distributed: ModelTestingGroupAction.unimportant,
    },
)

_update_and_add_testing_config(
    # Tests diffusion llama converter.
    "llama_yarn",
    "diffusion_llama",
    updates={},
    # Megatron doesn't support Yarn-style Rotary Embeddings
    megatron_args=None,
    checkpoint_format=DiffusionLlamaCheckpointFormat,
    # TODO: Conversion is broken.
    # TODO: Add back generate as `normal` when stable.
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.unimportant,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.normal,
        ModelTestingGroup.convert: ModelTestingGroupAction.broken,
        ModelTestingGroup.generate: ModelTestingGroupAction.broken,
        ModelTestingGroup.megatron: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.distributed: ModelTestingGroupAction.unimportant,
    },
)

_update_and_add_testing_config(
    # Tests multi-token prediction, custom HF model and converter.
    "llama",
    "mtp_llama",
    updates={
        ("model", "base_model", "output_layer", "prediction_heads"): 2,
    },
    # Megatron doesn't support multi-token prediction.
    megatron_args=None,
    checkpoint_format=MTPLlamaCheckpointFormat,
    # TODO: Add back generate as `normal` when stable.
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.normal,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.normal,
        ModelTestingGroup.convert: ModelTestingGroupAction.normal,
        ModelTestingGroup.generate: ModelTestingGroupAction.broken,
        ModelTestingGroup.megatron: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.distributed: ModelTestingGroupAction.unimportant,
    },
    compare_factor=2.0,
)

_update_and_add_testing_config(
    # Tests partial linear biases, Qwen2 converter.
    "llama",
    "qwen_2",
    # TODO: replace
    updates={
        ("model", "base_model", "decoder", "block", "add_linear_biases"): "only_attn_qkv",
    },
    # Megatron doesn't support per sub layer biases.
    megatron_args=None,
    checkpoint_format=Qwen2CheckpointFormat,
    # TODO: Add back generate as `normal` when stable.
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.broken,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.broken,
        ModelTestingGroup.convert: ModelTestingGroupAction.broken,
        ModelTestingGroup.generate: ModelTestingGroupAction.broken,
        ModelTestingGroup.megatron: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.distributed: ModelTestingGroupAction.unimportant,
    },
)

_update_and_add_testing_config(
    # Tests diffusion dream converter.
    "qwen_2",
    "dream",
    # TODO: replace only_attn_qkv
    updates={},
    # Megatron doesn't support per sub layer biases.
    megatron_args=None,
    checkpoint_format=DiffusionDreamCheckpointFormat,
    # TODO: Conversion is broken.
    # TODO: Add back generate as `normal` when stable.
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.unimportant,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.broken,
        ModelTestingGroup.convert: ModelTestingGroupAction.broken,
        ModelTestingGroup.generate: ModelTestingGroupAction.broken,
        ModelTestingGroup.megatron: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.distributed: ModelTestingGroupAction.unimportant,
    },
)

_update_and_add_testing_config(
    # Tests sliding window attention, mistral converter.
    "llama",
    "mistral",
    updates={
        ("model", "base_model", "decoder", "block", "mixer", "window_size"): 128,
    },
    # Megatron doesn't support sliding windows.
    megatron_args=None,
    checkpoint_format=MistralCheckpointFormat,
    # TODO: Add back generate as `normal` when stable.
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.normal,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.normal,
        ModelTestingGroup.convert: ModelTestingGroupAction.normal,
        ModelTestingGroup.generate: ModelTestingGroupAction.broken,
        ModelTestingGroup.megatron: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.distributed: ModelTestingGroupAction.unimportant,
    },
)

_update_and_add_testing_config(
    # Tests mixture of experts, mixtral converter.
    "llama",
    "mixtral",
    updates={
        ("model", "base_model", "decoder", "block", "mlp", "type"): "moe",
        ("model", "base_model", "decoder", "block", "mlp", "router", "weight"): init_1,
        ("model", "base_model", "decoder", "block", "mlp", "experts"): 4,
        ("model", "base_model", "decoder", "block", "mlp", "experts_per_token"): 4,
    },
    megatron_args=[
        "--num-experts=4",
        "--moe-router-topk=4",
    ],
    checkpoint_format=MixtralCheckpointFormat,
    # TODO: New base image broke mixtral
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.normal,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.normal,
        ModelTestingGroup.convert: ModelTestingGroupAction.normal,
        ModelTestingGroup.generate: ModelTestingGroupAction.broken,
        ModelTestingGroup.megatron: ModelTestingGroupAction.normal,
        ModelTestingGroup.distributed: ModelTestingGroupAction.normal,
    },
    compare_factor=2.0,
)

_llama_block = MODEL_CONFIGS["llama"].config_dict["model"]["base_model"]["decoder"]["block"]


_update_and_add_testing_config(
    # Tests hybrid Mamba, llamba converter.
    "llama",
    "hybrid_mamba",
    updates={
        ("model", "base_model", "decoder"): {
            "type": "pattern",
            "blocks": {
                "t": copy.deepcopy(_llama_block),
                "m": {
                    **copy.deepcopy(_llama_block),
                    "mixer": {
                        "type": "mamba",
                        "d_inner": 512,
                        "state_size": 16,
                        "dt_rank": 16,
                        "add_linear_biases": False,
                    },
                },
            },
            "num_blocks": 2,
            "pattern": ["t", "m"],
        },
    },
    megatron_args=None,
    checkpoint_format=AprielHybridSSMCheckpointFormat,
    # TODO: Add back generate as `normal` when stable.
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.normal,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.normal,
        # TODO: Fix and bring back to `testing_groups`
        ModelTestingGroup.convert: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.generate: ModelTestingGroupAction.broken,
        ModelTestingGroup.megatron: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.distributed: ModelTestingGroupAction.not_implemented,
    },
    compare_factor=2.0,
    # Micro-sequence split not supported.
    skip_tests=("sdp", "ms"),
)

_update_and_add_testing_config(
    # Tests hybrid Mamba 2.
    "llama",
    "hybrid_mamba_2",
    updates={
        ("model", "base_model", "decoder"): {
            "type": "pattern",
            "blocks": {
                "t": copy.deepcopy(_llama_block),
                "m2": {
                    **copy.deepcopy(_llama_block),
                    "mixer": {
                        "type": "mamba_2",
                        "dt_layer": {"bias": {"enabled": True}},
                        "d_inner": 512,
                        "state_size": 8,
                        "dt_rank": 16,
                        "d_xb": 256,
                        "add_linear_biases": False,
                    },
                },
            },
            "num_blocks": 2,
            "pattern": ["t", "m2"],
        },
    },
    megatron_args=None,
    checkpoint_format=AprielHybridSSMCheckpointFormat,
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.normal,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.normal,
        ModelTestingGroup.convert: ModelTestingGroupAction.normal,
        ModelTestingGroup.generate: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.megatron: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.distributed: ModelTestingGroupAction.normal,
    },
    compare_factor=2.0,
    # Micro-sequence split not supported.
    skip_tests=(
        "sdp",
        "ms",
    ),  # "pp","dp", "ce","16", "bf", "df", "stp"),
)


_update_and_add_testing_config(
    # Tests hybrid discrete Mamba 2.
    "llama",
    "hybrid_discrete_mamba_2",
    updates={
        ("model", "base_model", "decoder"): {
            "type": "pattern",
            "blocks": {
                "t": copy.deepcopy(_llama_block),
                "m2d": {
                    **copy.deepcopy(_llama_block),
                    "mixer": {
                        "type": "discrete_mamba_2",
                        "d_inner": 512,
                        "state_size": 8,
                        "n_qk_heads": 8,
                        "n_v_heads": 16,
                        "chunk_size": 32,
                        "add_linear_biases": False,
                    },
                },
            },
            "num_blocks": 2,
            "pattern": ["t", "m2d"],
        },
    },
    megatron_args=None,
    checkpoint_format=AprielHybridSSMCheckpointFormat,
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.normal,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.normal,
        ModelTestingGroup.convert: ModelTestingGroupAction.normal,
        ModelTestingGroup.generate: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.megatron: ModelTestingGroupAction.not_implemented,
        # TODO: Implement
        ModelTestingGroup.distributed: ModelTestingGroupAction.normal,
    },
    compare_factor=2.0,
    # Micro-sequence split and sequence-first not supported.
    skip_tests=("sdp", "ms"),
)


@pytest.fixture(scope="session", params=MODEL_CONFIGS.keys())
def model_testing_config(request) -> ModelTestingConfig:
    models = request.config.getoption("--models")
    if models and request.param not in models:
        pytest.skip(f"Skipping model {request.param}")
    return MODEL_CONFIGS[request.param]


def testing_group_enabled(item: pytest.Function, skip_slow: bool, skip_extra_slow: bool, show_skipped: bool) -> bool:
    if "model_testing_group" in item.keywords:
        assert hasattr(item, "callspec") and "model_testing_config" in item.callspec.params, item.nodeid
        groups: tuple[ModelTestingGroup] = item.keywords["model_testing_group"].args
        model_testing_config = item.callspec.params["model_testing_config"]
        model_config: ModelTestingConfig = MODEL_CONFIGS[model_testing_config]
        for group in groups:
            action = model_config.groups[group]
            if action == ModelTestingGroupAction.main:
                pass
            elif action == ModelTestingGroupAction.normal and not skip_slow:
                pass
            elif (
                action in (ModelTestingGroupAction.broken, ModelTestingGroupAction.unimportant) and not skip_extra_slow
            ):
                pass
            elif show_skipped:
                item.add_marker(
                    pytest.mark.skip(reason=f"Skipping testing group {group} for model {model_testing_config}.")
                )
            else:
                return False
    elif hasattr(item, "callspec"):
        assert "model_testing_config" not in item.callspec.params, item.nodeid

    return True
