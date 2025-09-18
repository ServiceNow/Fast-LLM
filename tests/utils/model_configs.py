import dataclasses
import enum
import functools
import os
import typing

import pytest

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.models.gpt.config import (
    DiffusionDreamGPTHuggingfaceCheckpointFormat,
    DiffusionLlamaGPTHuggingfaceCheckpointFormat,
    LlamaGPTHuggingfaceCheckpointFormat,
    MistralGPTHuggingfaceCheckpointFormat,
    MixtralGPTHuggingfaceCheckpointFormat,
    MTPLlamaGPTHuggingfaceCheckpointFormat,
    Qwen2GPTHuggingfaceCheckpointFormat,
    Starcoder2GPTHuggingfaceCheckpointFormat,
)
from fast_llm.models.ssm.config import (
    AprielSSMHHybridHuggingfaceCheckpointFormat,
    AprielThinkerSSMHHybridHuggingfaceCheckpointFormat,
    LLambaHuggingfaceCheckpointFormat,
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


@dataclasses.dataclass(kw_only=True, frozen=True)
class ModelTestingConfig:
    name: str = None
    model_type: str
    config_args: list[str]
    megatron_args: list[str] | None
    checkpoint_format: type[CheckpointFormat] | None
    groups: dict[ModelTestingGroup, ModelTestingGroupAction]
    # Scale the comparison thresholds for specific models.
    compare_factor: float = 1.0
    # Option to skip specific distributed configuration with name containing any of the provided strings.
    skip_tests: tuple[str] = ()

    @functools.cached_property
    def trainer_config_class(self) -> type[TrainerConfig]:
        return TrainerConfig.get_subclass(self.model_type)

    @functools.cached_property
    def trainer_config(self) -> TrainerConfig:
        # See `RunnableConfig._from_parsed_args`
        return self.trainer_config_class.from_dict(self.trainer_config_class._parse_updates(self.config_args))

    @functools.cached_property
    def evaluators_config_class(self) -> type[EvaluatorsConfig]:
        # EvaluatorsConfig is a base class that, during parse_and_run, replaces itself with the appropriate TrainingConfig subclass.
        # Therefore, the arguments passed to EvaluatorsConfig.parse_and_run must include the model type as the first element.
        return EvaluatorsConfig

    @functools.cached_property
    def evaluators_config(self) -> EvaluatorsConfig:
        # See `RunnableConfig._from_parsed_args`
        return self.evaluators_config_class.from_dict(self.evaluators_config_class._parse_updates(self.config_args))

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
    extra_args: list[str] | None = None,
    megatron_args: list[str] | None = ...,
    groups: dict[ModelTestingGroup, ModelTestingGroupAction],
    **kwargs,
):
    config = MODEL_CONFIGS[old_name]
    updates: dict[str, typing.Any] = {
        "name": new_name,
        "groups": groups,
    }
    if model_type is not None:
        updates["model_type"] = model_type
    if extra_args is not None:
        updates["config_args"] = config.config_args + extra_args
    if megatron_args is not ...:
        if megatron_args is None:
            updates["megatron_args"] = None
        elif config.megatron_args is None:
            updates["megatron_args"] = megatron_args
        else:
            updates["megatron_args"] = config.megatron_args + megatron_args
    updates.update(kwargs)

    MODEL_CONFIGS[new_name] = dataclasses.replace(config, **updates)


MODEL_CONFIGS: dict[str, ModelTestingConfig] = {}


MODEL_CONFIGS["gpt2"] = ModelTestingConfig(
    # Tests gpt2 features (absolute embeddings, layer norm,  relu activation, tied embeddings, MHA, linear biases).
    name="gpt2",
    model_type="gpt",
    config_args=[
        "training.logs.interval=1",
        "run.tensor_logs.save=True",
        "run.tensor_logs.show=False",
        "model.base_model.max_position_embeddings=512",
        "model.base_model.transformer.num_layers=2",
        "model.base_model.transformer.hidden_size=256",
        "model.base_model.transformer.num_attention_heads=8",
        "model.base_model.transformer.head_groups=8",
        "model.base_model.transformer.init_method_std=0.022",
        f"model.base_model.vocab_size={MODEL_TEST_VOCAB_SIZE}",
        f"model.multi_stage.debug_param_init={_LOG_LEVEL}",
        f"model.multi_stage.debug_layer_outputs={_LOG_LEVEL}",
        f"model.multi_stage.debug_layer_gradients={_LOG_LEVEL}",
        f"model.multi_stage.debug_all_param_gradients={_LOG_LEVEL}",
        "model.multi_stage.debug_tensor_parallel=True",
        "model.distributed.reproducible_init=True",
        "model.distributed.timeout=20",
        "training.train_iters=2",
        "training.num_workers=0",
        "training.timeout=30",
        "batch.batch_size=8",
        "batch.sequence_length=512",
        "data.datasets.training.type=slice",
        "data.datasets.training.end=0.969",
        "data.datasets.training.dataset.type=memmap",
        f"data.datasets.training.dataset.path={MODEL_DATASET_PREFIX}",
        "data.datasets.validation.type=slice",
        "data.datasets.validation.begin=0.969",
        "data.datasets.validation.end=0.999",
        "data.datasets.validation.dataset.type=memmap",
        f"data.datasets.validation.dataset.path={MODEL_DATASET_PREFIX}",
        "data.datasets.test.type=slice",
        "data.datasets.test.begin=0.999",
        "data.datasets.test.end=1",
        "data.datasets.test.dataset.type=memmap",
        f"data.datasets.test.dataset.path={MODEL_DATASET_PREFIX}",
        "optimizer.learning_rate.base=0.0001",
    ],
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
        "--init-method-std=0.022",
        "--lr=0.0001",
        "--num-workers=0",
        "--valid-num-workers=0",
        "--tokenizer-type=NullTokenizer",
        # Megatron messes with the vocab size, so we have to subtract 1.
        f"--vocab-size={MODEL_TEST_VOCAB_SIZE - 1}",
        f"--data-path={MODEL_DATASET_PREFIX}",
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
        ModelTestingGroup.convert: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.generate: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.megatron: ModelTestingGroupAction.normal,
        ModelTestingGroup.distributed: ModelTestingGroupAction.normal,
    },
)

_update_and_add_testing_config(
    # Tests MQA.
    "gpt2",
    "starcoder",
    extra_args=["model.base_model.transformer.head_groups=1"],
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
    "gpt2",
    "starcoder2",
    extra_args=[
        "model.base_model.transformer.head_groups=4",
        "model.base_model.transformer.rotary.type=default",
        # Unused, but prevents issues with conversion tests.
        "model.base_model.max_position_embeddings=2048",
    ],
    megatron_args=[
        "--group-query-attention",
        "--num-query-groups=4",
        "--use-rotary-position-embeddings",
        "--no-position-embedding",
    ],
    checkpoint_format=Starcoder2GPTHuggingfaceCheckpointFormat,
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
    "starcoder2",
    "llama",
    extra_args=[
        "model.base_model.transformer.gated=True",
        "model.base_model.transformer.activation_type=silu",
        "model.base_model.transformer.add_linear_biases=False",
        "model.base_model.transformer.normalization.type=rms_norm",
        "model.base_model.transformer.ffn_hidden_size=1024",
        "model.base_model.tie_word_embeddings=False",
    ],
    megatron_args=[
        "--swiglu",
        "--disable-bias-linear",
        "--normalization=RMSNorm",
        "--ffn-hidden-size=1024",
        "--untie-embeddings-and-output-weights",
    ],
    checkpoint_format=LlamaGPTHuggingfaceCheckpointFormat,
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
    "llama3",
    extra_args=["model.base_model.transformer.rotary.type=llama3"],
    # Megatron doesn't support Llama3-style Rotary Embeddings
    megatron_args=None,
    checkpoint_format=LlamaGPTHuggingfaceCheckpointFormat,
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
    extra_args=["model.base_model.transformer.rotary.type=yarn"],
    # Megatron doesn't support Yarn-style Rotary Embeddings
    megatron_args=None,
    checkpoint_format=LlamaGPTHuggingfaceCheckpointFormat,
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
    extra_args=[],
    # Megatron doesn't support Yarn-style Rotary Embeddings
    megatron_args=None,
    checkpoint_format=DiffusionLlamaGPTHuggingfaceCheckpointFormat,
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
    "llama_mtp",
    extra_args=["model.base_model.prediction_heads=4"],
    # Megatron doesn't support multi-token prediction.
    megatron_args=None,
    checkpoint_format=MTPLlamaGPTHuggingfaceCheckpointFormat,
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
    "qwen2",
    extra_args=["model.base_model.transformer.add_linear_biases=only_attn_qkv"],
    # Megatron doesn't support per sub layer biases.
    megatron_args=None,
    checkpoint_format=Qwen2GPTHuggingfaceCheckpointFormat,
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
    # Tests diffusion dream converter.
    "qwen2",
    "dream",
    extra_args=[],
    # Megatron doesn't support per sub layer biases.
    megatron_args=None,
    checkpoint_format=DiffusionDreamGPTHuggingfaceCheckpointFormat,
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
    # Tests sliding window attention, mistral converter.
    "llama",
    "mistral",
    extra_args=["model.base_model.transformer.window_size=128"],
    # Megatron doesn't support sliding windows.
    megatron_args=None,
    checkpoint_format=MistralGPTHuggingfaceCheckpointFormat,
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
    extra_args=[
        "model.base_model.transformer.num_experts=4",
        "model.base_model.transformer.num_experts_per_token=4",
    ],
    megatron_args=[
        "--num-experts=4",
        "--moe-router-topk=4",
    ],
    checkpoint_format=MixtralGPTHuggingfaceCheckpointFormat,
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

_update_and_add_testing_config(
    # Tests hybrid Mamba, llamba converter.
    "llama",
    "llamba",
    model_type="hybrid_ssm",
    extra_args=[
        "model.base_model.hybrid_block_layout=['t','m']",
        "model.base_model.ssm.d_inner=512",
        "model.base_model.ssm.state_size=16",
    ],
    megatron_args=None,
    checkpoint_format=LLambaHuggingfaceCheckpointFormat,
    # TODO: Add back generate as `normal` when stable.
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.normal,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.normal,
        # TODO: Fix and bring back to `testing_groups`
        ModelTestingGroup.convert: ModelTestingGroupAction.broken,
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
    "hybrid_mamba2",
    model_type="hybrid_ssm",
    extra_args=[
        "model.base_model.hybrid_block_layout=['t','m2']",
        "model.base_model.ssm.d_inner=512",
        "model.base_model.ssm.state_size=8",
        "model.base_model.ssm.d_xb=256",
        # f"model.base_model.transformer.debug_transformer={_LOG_LEVEL}"
    ],
    megatron_args=None,
    checkpoint_format=AprielThinkerSSMHHybridHuggingfaceCheckpointFormat,
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
    "hybrid_discrete_mamba2",
    model_type="hybrid_ssm",
    extra_args=[
        "model.base_model.hybrid_block_layout=['t','m2d']",
        "model.base_model.ssm.d_inner=512",
        "model.base_model.ssm.state_size=8",
        "model.base_model.ssm.n_qk_heads=8",
        "model.base_model.ssm.n_v_heads=16",
        "model.base_model.ssm.chunk_size=32",
    ],
    megatron_args=None,
    checkpoint_format=AprielSSMHHybridHuggingfaceCheckpointFormat,
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
