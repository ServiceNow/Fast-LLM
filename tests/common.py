import os
import pathlib
import shutil
import subprocess
import sys

import pytest
import torch

from fast_llm.models.gpt.config import HuggingfaceModelType
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
from tests.compare_tensor_logs import CompareConfig, compare_tensor_logs

# FIXME: figure out correct import of megatron modules without this hack
sys.path.append(os.getcwd())


# Keep all results in one place to allow recovering them for debugging in case of failure.
TEST_RESULTS_PATH = pathlib.Path(os.environ.get("TEST_RESULTS_PATH", "/tmp/test_fastllm_correctness"))
FORCE_REUSE_RESULTS = int(os.environ.get("FORCE_REUSE_RESULTS", 0)) != 0
REUSE_RESULTS = FORCE_REUSE_RESULTS or int(os.environ.get("REUSE_RESULTS", 0)) != 0
_LOG_LEVEL = int(os.environ.get("LOG_LEVEL", 13))

ARTIFACT_PATH = "runs/0/artifacts"

CONFIG_BASE_FAST_LLM = [
    "--num_layers=2",
    "--hidden_size=1024",
    "--num_attention_heads=8",
    "--log_interval=1",
    "--train_iters=2",
    "--validation_iters=0",
    "--hidden_dropout=0",
    "--attention_dropout=0",
    f"--debug_param_init={_LOG_LEVEL}",
    f"--debug_layer_outputs={_LOG_LEVEL}",
    f"--debug_layer_gradients={_LOG_LEVEL}",
    f"--debug_all_param_gradients={_LOG_LEVEL}",
    "--debug_tensor_parallel=1",
    "--debug_param_update=0",
    "--reproducible_init=1",
    "--batch_size=8",
    "--sequence_length=2048",
    "--init_method_std=0.022",
    "--lr=0.0001",
    "--vocab_size=49152",
    "--num_workers=4",
    "--data_path=/mnt/datasets/march_datasets/fixed_tokenizer_2/jupyter_structured/gpt2-preprocessed_content_document",
    "--save_tensor_log=1",
    "--show_tensor_logs=0",
    "--transposed_mlp_weight=1",
]
CONFIG_BASE_MEGATRON = [
    "--num-layers=2",
    "--hidden-size=1024",
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
    "--max-position-embeddings=2048",
    "--seq-length=2048",
    "--init-method-std=0.022",
    "--lr=0.0001",
    "--num-workers=4",
    "--valid-num-workers=4",
    "--tokenizer-type=TokenizerFromFile",
    "--tokenizer-file=/mnt/datasets/tokenizers/tokenizer-the-stack-march-sample-v3/tokenizer.json",
    "--make-vocab-size-divisible-by=128",
    "--data-path=/mnt/datasets/march_datasets/fixed_tokenizer_2/jupyter_structured/gpt2-preprocessed_content_document",
    "--lr-decay-style=constant",
    # Initialization is set up to match MCore models (MCore inverts self-attn qkv and dense layers compared to original Megatron)
    "--use-mcore-models",
    # local implementation doesn't allow for RMS norm.
    "--transformer-impl=transformer_engine",
]

CONFIG_SC1_FAST_LLM = CONFIG_BASE_FAST_LLM + ["--max_position_embeddings=2048"]
CONFIG_SC1_MEGATRON = CONFIG_BASE_MEGATRON + ["--group-query-attention"]
CONFIG_SC1_COMMON = CONFIG_SC1_FAST_LLM + ["--training-dtype=bf16"]

CONFIG_GPT2_FAST_LLM = CONFIG_SC1_FAST_LLM + ["--head_groups=8"]
CONFIG_GPT2_MEGATRON = CONFIG_BASE_MEGATRON
CONFIG_GPT2_COMMON = CONFIG_GPT2_FAST_LLM + ["--training-dtype=bf16"]

CONFIG_SC2_FAST_LLM = CONFIG_BASE_FAST_LLM + ["--head_groups=4", "--use_rotary_embeddings=1"]
CONFIG_SC2_MEGATRON = CONFIG_SC1_MEGATRON + [
    "--num-query-groups=4",
    "--use-rotary-position-embeddings",
    "--no-position-embedding",
]
CONFIG_SC2_COMMON = CONFIG_SC2_FAST_LLM + ["--training-dtype=bf16"]

CONFIG_MISTRAL_MEGATRON = CONFIG_SC2_MEGATRON + [
    "--swiglu",
    "--disable-bias-linear",
    "--normalization=RMSNorm",
    "--ffn-hidden-size=4096",
    "--untie-embeddings-and-output-weights",
]
CONFIG_MISTRAL_FAST_LLM = CONFIG_SC2_FAST_LLM + [
    "--gated=1",
    "--activation_type=silu",
    "--add_linear_biases=0",
    "--normalization_type=rms_norm",
    "--ffn_hidden_size=4096",
    "--tie_word_embeddings=0",
]
CONFIG_MISTRAL_COMMON = CONFIG_MISTRAL_FAST_LLM + ["--training-dtype=bf16"]

CONFIG_MIXTRAL_MEGATRON = CONFIG_MISTRAL_MEGATRON + ["--num-experts=4", "--moe-router-topk=4"]
CONFIG_MIXTRAL_FAST_LLM = CONFIG_MISTRAL_FAST_LLM + ["--num_experts=4", "--num_experts_per_token=4"]
CONFIG_MIXTRAL_COMMON = CONFIG_MIXTRAL_FAST_LLM + ["--training-dtype=bf16"]

_CONFIGS = {
    "gpt2": ("gpt", CONFIG_GPT2_FAST_LLM, CONFIG_GPT2_MEGATRON, CONFIG_GPT2_COMMON, None),
    "sc1": ("gpt", HuggingfaceGPTModelForCausalLM, CONFIG_SC1_FAST_LLM, CONFIG_SC1_MEGATRON, CONFIG_SC1_COMMON, None),
    "starcoder2": (
        "gpt",
        CONFIG_SC2_FAST_LLM,
        CONFIG_SC2_MEGATRON,
        CONFIG_SC2_COMMON,
        HuggingfaceModelType.starcoder2,
    ),
    "mistral": (
        "gpt",
        CONFIG_MISTRAL_FAST_LLM,
        CONFIG_MISTRAL_MEGATRON,
        CONFIG_MISTRAL_COMMON,
        HuggingfaceModelType.mistral,
    ),
    "mixtral": (
        "gpt",
        CONFIG_MIXTRAL_FAST_LLM,
        CONFIG_MIXTRAL_MEGATRON,
        CONFIG_MIXTRAL_COMMON,
        HuggingfaceModelType.mixtral,
    ),
}


TEST_MODEL = os.environ.get("MODEL", "mistral")

TEST_MODEL_TYPE, CONFIG_FAST_LLM, CONFIG_GPT2, CONFIG_COMMON, HUGGINGFACE_MODEL_TYPE = _CONFIGS[TEST_MODEL]


def run_test_script(
    name: str,
    script: list[str],
    num_gpus: int = 1,
    *,
    model_type: str = TEST_MODEL_TYPE,
    is_megatron: bool = False,
    compare: str | None = None,
    config: CompareConfig | None = None,
    prepare_fn=None,
    compare_fn=None,
):
    if torch.cuda.device_count() < num_gpus:
        pytest.skip(f"Not enough GPUs to run test ({torch.cuda.device_count()}<{num_gpus})")
    env = os.environ.copy()
    if is_megatron:
        # Prevent Megatron from complaining.
        env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        env["NVTE_FLASH_ATTN"] = "0"
    path = TEST_RESULTS_PATH.resolve() / name
    skip = False
    artifact_path = path / ARTIFACT_PATH
    if path.exists():
        assert path.is_dir()
        # TODO: Better way to check if the previous attempt succeeded.
        if (
            REUSE_RESULTS
            and artifact_path.is_dir()
            and len(list((artifact_path / "0").iterdir())) >= (1 if is_megatron else 3)
        ):
            skip = True
        elif FORCE_REUSE_RESULTS:
            raise RuntimeError(artifact_path)
        else:
            shutil.rmtree(path)
    elif FORCE_REUSE_RESULTS:
        raise RuntimeError(path)
    if prepare_fn is not None:
        skip = prepare_fn(TEST_RESULTS_PATH / name, None if compare is None else TEST_RESULTS_PATH / compare, skip)
    header = ["Megatron-LM/pretrain_gpt.py"] if is_megatron else ["--no-python", "fast-llm", "train", model_type]
    command = [
        "torchrun",
        f"--nproc-per-node={num_gpus}",
        *header,
        *script,
    ]
    if is_megatron:
        command.extend([f"--structured-logs-dir={path}", f"--data-cache-path={path}"])
    else:
        command.append(f"--experiment_dir={path}")
    print(" ".join(command))
    if skip:
        print("Reusing existing run.")
    else:
        completed_proc = subprocess.run(command, env=env)
        if completed_proc.returncode:
            raise RuntimeError(f"Process failed with return code {completed_proc.returncode}")
    if compare:
        if compare_fn is not None:
            compare_fn(TEST_RESULTS_PATH / name, TEST_RESULTS_PATH / compare)
        compare_tensor_logs(
            TEST_RESULTS_PATH / compare / ARTIFACT_PATH,
            TEST_RESULTS_PATH / name / ARTIFACT_PATH,
            config,
        )
