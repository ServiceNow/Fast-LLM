import os

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
from fast_llm.models.ssm.config import LLambaHuggingfaceCheckpointFormat
from tests.utils.dataset import DATASET_PREFIX, TEST_VOCAB_SIZE

_LOG_LEVEL = int(os.environ.get("LOG_LEVEL", 13))
TEST_MODEL = os.environ.get("MODEL", "llama")
CONFIG_BASE_FAST_LLM = [
    "training.logs.interval=1",
    "run.tensor_logs.save=True",
    "run.tensor_logs.show=False",
    "model.base_model.transformer.num_layers=2",
    "model.base_model.transformer.hidden_size=256",
    "model.base_model.transformer.num_attention_heads=8",
    "model.base_model.transformer.init_method_std=0.022",
    f"model.base_model.vocab_size={TEST_VOCAB_SIZE}",
    f"model.multi_stage.debug_param_init={_LOG_LEVEL}",
    f"model.multi_stage.debug_layer_outputs={_LOG_LEVEL}",
    f"model.multi_stage.debug_layer_gradients={_LOG_LEVEL}",
    f"model.multi_stage.debug_all_param_gradients={_LOG_LEVEL}",
    "model.multi_stage.debug_tensor_parallel=True",
    "model.distributed.reproducible_init=True",
    "model.distributed.timeout=10",
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
]
CONFIG_BASE_MEGATRON = [
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
]
CONFIG_SC1_FAST_LLM = CONFIG_BASE_FAST_LLM + ["model.base_model.max_position_embeddings=512"]
CONFIG_SC1_MEGATRON = CONFIG_BASE_MEGATRON + ["--group-query-attention"]
CONFIG_SC1_COMMON = CONFIG_SC1_FAST_LLM + ["model.distributed.training_dtype=bf16"]
CONFIG_GPT2_FAST_LLM = CONFIG_SC1_FAST_LLM + ["model.base_model.transformer.head_groups=8"]
CONFIG_GPT2_MEGATRON = CONFIG_BASE_MEGATRON
CONFIG_GPT2_COMMON = CONFIG_GPT2_FAST_LLM + ["model.distributed.training_dtype=bf16"]
CONFIG_SC2_FAST_LLM = CONFIG_BASE_FAST_LLM + [
    "model.base_model.transformer.head_groups=4",
    "model.base_model.transformer.rotary.type=default",
]
CONFIG_SC2_MEGATRON = CONFIG_SC1_MEGATRON + [
    "--num-query-groups=4",
    "--use-rotary-position-embeddings",
    "--no-position-embedding",
]
CONFIG_SC2_COMMON = CONFIG_SC2_FAST_LLM + ["model.distributed.training_dtype=bf16"]
CONFIG_LLAMA_MEGATRON = CONFIG_SC2_MEGATRON + [
    "--swiglu",
    "--disable-bias-linear",
    "--normalization=RMSNorm",
    "--ffn-hidden-size=1024",
    "--untie-embeddings-and-output-weights",
]
CONFIG_LLAMA_FAST_LLM = CONFIG_SC2_FAST_LLM + [
    "model.base_model.transformer.gated=True",
    "model.base_model.transformer.activation_type=silu",
    "model.base_model.transformer.add_linear_biases=False",
    "model.base_model.transformer.normalization.type=rms_norm",
    "model.base_model.transformer.ffn_hidden_size=1024",
    "model.base_model.tie_word_embeddings=False",
]
CONFIG_LLAMA_COMMON = CONFIG_LLAMA_FAST_LLM + ["model.distributed.training_dtype=bf16"]
CONFIG_LLAMA3_MEGATRON = None
CONFIG_LLAMA3_FAST_LLM = CONFIG_LLAMA_FAST_LLM + [
    "model.base_model.transformer.rotary.type=llama3",
]
CONFIG_LLAMA3_COMMON = CONFIG_LLAMA3_FAST_LLM + ["model.distributed.training_dtype=bf16"]
CONFIG_QWEN2_MEGATRON = None
CONFIG_QWEN2_FAST_LLM = CONFIG_SC2_FAST_LLM + [
    "model.base_model.transformer.gated=True",
    "model.base_model.transformer.activation_type=silu",
    "model.base_model.transformer.add_linear_biases=only_attn_qkv",
    "model.base_model.transformer.normalization.type=rms_norm",
    "model.base_model.transformer.ffn_hidden_size=1024",
    "model.base_model.tie_word_embeddings=False",
]
CONFIG_QWEN2_COMMON = CONFIG_QWEN2_FAST_LLM + ["model.distributed.training_dtype=bf16"]
CONFIG_LLAMA_YARN_MEGATRON = None
CONFIG_LLAMA_YARN_FAST_LLM = CONFIG_LLAMA_FAST_LLM + [
    "model.base_model.transformer.rotary.type=yarn",
]
CONFIG_LLAMA_YARN_COMMON = CONFIG_LLAMA_YARN_FAST_LLM + ["model.distributed.training_dtype=bf16"]
CONFIG_MIXTRAL_MEGATRON = CONFIG_LLAMA_MEGATRON + [
    "--num-experts=4",
    "--moe-router-topk=4",
]
CONFIG_MIXTRAL_FAST_LLM = CONFIG_LLAMA_FAST_LLM + [
    "model.base_model.transformer.num_experts=4",
    "model.base_model.transformer.num_experts_per_token=4",
]
CONFIG_MIXTRAL_COMMON = CONFIG_MIXTRAL_FAST_LLM + ["model.distributed.training_dtype=bf16"]
CONFIG_MIXTRAL_YARN_MEGATRON = None
CONFIG_MIXTRAL_YARN_FAST_LLM = CONFIG_MIXTRAL_FAST_LLM + [
    "model.base_model.transformer.rotary.type=yarn",
]
CONFIG_MIXTRAL_YARN_COMMON = CONFIG_MIXTRAL_YARN_FAST_LLM + ["model.distributed.training_dtype=bf16"]
CONFIG_LLAMA_MTP_MEGATRON = None
CONFIG_LLAMA_MTP_FAST_LLM = CONFIG_LLAMA_FAST_LLM + [
    "model.base_model.prediction_heads=4",
]
CONFIG_LLAMA_MTP_COMMON = CONFIG_LLAMA_MTP_FAST_LLM + ["model.distributed.training_dtype=bf16"]
CONFIG_LLAMBA_FAST_LLM = CONFIG_LLAMA_FAST_LLM + ["model.base_model.hybrid_block_layout==['t','m']"]
CONFIG_LLAMBA_MEGATRON = CONFIG_LLAMA_MEGATRON + []
CONFIG_LLAMBA_COMMON = CONFIG_LLAMBA_FAST_LLM
_CONFIGS = {
    "gpt2": ("gpt", CONFIG_GPT2_FAST_LLM, CONFIG_GPT2_MEGATRON, CONFIG_GPT2_COMMON, None),
    "sc1": ("gpt", CONFIG_SC1_FAST_LLM, CONFIG_SC1_MEGATRON, CONFIG_SC1_COMMON, None),
    "starcoder2": (
        "gpt",
        CONFIG_SC2_FAST_LLM,
        CONFIG_SC2_MEGATRON,
        CONFIG_SC2_COMMON,
        Starcoder2GPTHuggingfaceCheckpointFormat,
    ),
    "llama": (
        "gpt",
        CONFIG_LLAMA_FAST_LLM,
        CONFIG_LLAMA_MEGATRON,
        CONFIG_LLAMA_COMMON,
        LlamaGPTHuggingfaceCheckpointFormat,
    ),
    "llama3": (
        "gpt",
        CONFIG_LLAMA3_FAST_LLM,
        CONFIG_LLAMA3_MEGATRON,
        CONFIG_LLAMA3_COMMON,
        LlamaGPTHuggingfaceCheckpointFormat,
    ),
    "qwen2": (
        "gpt",
        CONFIG_QWEN2_FAST_LLM,
        CONFIG_QWEN2_MEGATRON,
        CONFIG_QWEN2_COMMON,
        Qwen2GPTHuggingfaceCheckpointFormat,
    ),
    "dream": (
        "gpt",
        CONFIG_QWEN2_FAST_LLM,
        CONFIG_QWEN2_MEGATRON,
        CONFIG_QWEN2_COMMON,
        DiffusionDreamGPTHuggingfaceCheckpointFormat,
    ),
    "llama-yarn": (
        "gpt",
        CONFIG_LLAMA_YARN_FAST_LLM,
        CONFIG_LLAMA_YARN_MEGATRON,
        CONFIG_LLAMA_YARN_COMMON,
        LlamaGPTHuggingfaceCheckpointFormat,
    ),
    "diffusion_llama": (
        "gpt",
        CONFIG_LLAMA_YARN_FAST_LLM,
        CONFIG_LLAMA_YARN_MEGATRON,
        CONFIG_LLAMA_YARN_COMMON,
        DiffusionLlamaGPTHuggingfaceCheckpointFormat,
    ),
    "mistral": (
        "gpt",
        CONFIG_LLAMA_FAST_LLM,
        CONFIG_LLAMA_MEGATRON,
        CONFIG_LLAMA_COMMON,
        MistralGPTHuggingfaceCheckpointFormat,
    ),
    "mixtral": (
        "gpt",
        CONFIG_MIXTRAL_FAST_LLM,
        CONFIG_MIXTRAL_MEGATRON,
        CONFIG_MIXTRAL_COMMON,
        MixtralGPTHuggingfaceCheckpointFormat,
    ),
    "llamba": (
        "hybrid_ssm",
        CONFIG_LLAMBA_FAST_LLM,
        CONFIG_LLAMBA_MEGATRON,
        CONFIG_LLAMBA_COMMON,
        LLambaHuggingfaceCheckpointFormat,
    ),
    "mixtral-yarn": (
        "gpt",
        CONFIG_MIXTRAL_YARN_FAST_LLM,
        CONFIG_MIXTRAL_YARN_MEGATRON,
        CONFIG_MIXTRAL_YARN_COMMON,
        MixtralGPTHuggingfaceCheckpointFormat,
    ),
    "llama-mtp": (
        "gpt",
        CONFIG_LLAMA_MTP_FAST_LLM,
        CONFIG_LLAMA_MTP_MEGATRON,
        CONFIG_LLAMA_MTP_COMMON,
        MTPLlamaGPTHuggingfaceCheckpointFormat,
    ),
}

TEST_MODEL_TYPE, CONFIG_FAST_LLM, CONFIG_GPT2, CONFIG_COMMON, HUGGINGFACE_CHECKPOINT_FORMAT = _CONFIGS[TEST_MODEL]
