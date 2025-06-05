import dataclasses
import functools
import os
import typing

import pytest

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.models.auto import model_registry
from fast_llm.models.gpt.config import (
    LlamaGPTHuggingfaceCheckpointFormat,
    MistralGPTHuggingfaceCheckpointFormat,
    MixtralGPTHuggingfaceCheckpointFormat,
    MTPLlamaGPTHuggingfaceCheckpointFormat,
    Qwen2GPTHuggingfaceCheckpointFormat,
    Starcoder2GPTHuggingfaceCheckpointFormat,
)
from fast_llm.models.ssm.config import LLambaHuggingfaceCheckpointFormat
from tests.utils.dataset import DATASET_PREFIX, TEST_VOCAB_SIZE

_LOG_LEVEL = int(os.environ.get("LOG_LEVEL", 13))


@dataclasses.dataclass(kw_only=True, frozen=True)
class ModelTestingConfig:
    name: str = None
    model_type: str
    config_args: list[str]
    megatron_args: list[str] | None
    checkpoint_format: CheckpointFormat | None

    @functools.cached_property
    def model_config_class(self):
        return model_registry[self.model_type]

    @functools.cached_property
    def huggingface_model_for_causal_lm_class(self):
        return self.model_config_class.get_huggingface_model_for_causal_lm_class()

    @functools.cached_property
    def model_class(self):
        return self.model_config_class.get_model_class()

    @functools.cached_property
    def base_model_config_class(self):
        return self.model_config_class.get_base_model_config_class()


def _update_and_add_testing_config(
    old_name: str,
    new_name: str,
    *,
    model_type: str | None = None,
    extra_args: list[str] | None = None,
    megatron_args: list[str] | None = ...,
    checkpoint_format: CheckpointFormat | None = ...,
):
    config = _MODEL_CONFIGS[old_name]
    updates: dict[str, typing.Any] = {"name": new_name}
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
    if checkpoint_format is not ...:
        updates["checkpoint_format"] = checkpoint_format

    _MODEL_CONFIGS[new_name] = dataclasses.replace(config, **updates)


_MODEL_CONFIGS: dict[str, ModelTestingConfig] = {}


_MODEL_CONFIGS["gpt2"] = ModelTestingConfig(
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
        f"model.base_model.vocab_size={TEST_VOCAB_SIZE}",
        f"model.multi_stage.debug_param_init={_LOG_LEVEL}",
        f"model.multi_stage.debug_layer_outputs={_LOG_LEVEL}",
        f"model.multi_stage.debug_layer_gradients={_LOG_LEVEL}",
        f"model.multi_stage.debug_all_param_gradients={_LOG_LEVEL}",
        "model.multi_stage.debug_tensor_parallel=True",
        "model.distributed.reproducible_init=True",
        "model.distributed.timeout=10",
        "model.distributed.training_dtype=bf16",
        "training.train_iters=2",
        "training.num_workers=0",
        "training.timeout=30",
        "batch.batch_size=8",
        "batch.sequence_length=512",
        "data.datasets.training.type=slice",
        "data.datasets.training.end=0.969",
        "data.datasets.training.dataset.type=memmap",
        f"data.datasets.training.dataset.path={DATASET_PREFIX}",
        "data.datasets.validation.type=slice",
        "data.datasets.validation.begin=0.969",
        "data.datasets.validation.end=0.999",
        "data.datasets.validation.dataset.type=memmap",
        f"data.datasets.validation.dataset.path={DATASET_PREFIX}",
        "data.datasets.test.type=slice",
        "data.datasets.test.begin=0.999",
        "data.datasets.test.end=1",
        "data.datasets.test.dataset.type=memmap",
        f"data.datasets.test.dataset.path={DATASET_PREFIX}",
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
        f"--vocab-size={TEST_VOCAB_SIZE - 1}",
        f"--data-path={DATASET_PREFIX}",
        "--lr-decay-style=constant",
        # Initialization is set up to match MCore models (MCore inverts self-attn qkv and dense layers compared to original Megatron)
        "--use-mcore-models",
        # local implementation doesn't allow for RMS norm.
        "--transformer-impl=transformer_engine",
    ],
    checkpoint_format=None,
)

_update_and_add_testing_config(
    "gpt2",
    "starcoder",
    extra_args=["model.base_model.transformer.head_groups=1"],
    megatron_args=["--group-query-attention"],
    checkpoint_format=None,
)

_update_and_add_testing_config(
    "gpt2",
    "starcoder2",
    extra_args=[
        "model.base_model.transformer.head_groups=4",
        "model.base_model.transformer.rotary.type=default",
    ],
    megatron_args=[
        "--group-query-attention",
        "--num-query-groups=4",
        "--use-rotary-position-embeddings",
        "--no-position-embedding",
    ],
    checkpoint_format=Starcoder2GPTHuggingfaceCheckpointFormat,
)

_update_and_add_testing_config(
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
)

_update_and_add_testing_config(
    "llama",
    "llama3",
    extra_args=["model.base_model.transformer.rotary.type=llama3"],
    # Megatron doesn't support Llama3-style Rotary Embeddings
    megatron_args=None,
    checkpoint_format=LlamaGPTHuggingfaceCheckpointFormat,
)

_update_and_add_testing_config(
    "llama",
    "llama_yarn",
    extra_args=["model.base_model.transformer.rotary.type=yarn"],
    # Megatron doesn't support Yarn-style Rotary Embeddings
    megatron_args=None,
    checkpoint_format=LlamaGPTHuggingfaceCheckpointFormat,
)

_update_and_add_testing_config(
    "llama",
    "llama_mtp",
    extra_args=["model.base_model.prediction_heads=4"],
    # Megatron doesn't support multi-token prediction.
    megatron_args=None,
    checkpoint_format=MTPLlamaGPTHuggingfaceCheckpointFormat,
)

_update_and_add_testing_config(
    "llama",
    "qwen2",
    extra_args=["model.base_model.transformer.add_linear_biases=only_attn_qkv"],
    # Megatron doesn't support per sub layer biases
    megatron_args=None,
    checkpoint_format=Qwen2GPTHuggingfaceCheckpointFormat,
)

_update_and_add_testing_config(
    "llama",
    "mistral",
    extra_args=["model.base_model.transformer.window_size=128"],
    # Megatron doesn't support sliding windows.
    megatron_args=None,
    checkpoint_format=MistralGPTHuggingfaceCheckpointFormat,
)

_update_and_add_testing_config(
    # We ignore sliding windows to enable comparison with Megatron.
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
)

_update_and_add_testing_config(
    # We ignore sliding windows to enable comparison with Megatron.
    "llama",
    "llamba",
    model_type="hybrid_ssm",
    extra_args=["model.base_model.hybrid_block_layout=['t','m']"],
    megatron_args=None,
    checkpoint_format=LLambaHuggingfaceCheckpointFormat,
)


@pytest.fixture(scope="session", params=_MODEL_CONFIGS.keys())
def model_testing_config(request) -> ModelTestingConfig:
    return _MODEL_CONFIGS[request.param]
