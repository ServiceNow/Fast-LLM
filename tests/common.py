import os
import pathlib
import random
import shutil
import string
import subprocess
import sys

import numpy as np
import pytest
import torch

from fast_llm.data.mmap import MMapIndexedDataset
from fast_llm.models.gpt.config import HuggingfaceModelType
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
from tests.compare_tensor_logs import CompareConfig, compare_tensor_logs

# FIXME: figure out correct import of megatron modules without this hack
sys.path.append(os.getcwd())


# Keep all results in one place to allow recovering them for debugging in case of failure.
TEST_RESULTS_PATH = pathlib.Path(os.environ.get("TEST_RESULTS_PATH", "/tmp/fast_llm_tests"))
FORCE_REUSE_RESULTS = int(os.environ.get("FORCE_REUSE_RESULTS", 0)) != 0
REUSE_RESULTS = FORCE_REUSE_RESULTS or int(os.environ.get("REUSE_RESULTS", 0)) != 0
_LOG_LEVEL = int(os.environ.get("LOG_LEVEL", 13))

ARTIFACT_PATH = "runs/0/artifacts"

TOKENIZER_PATH = TEST_RESULTS_PATH / "data" / "tokenizer"
TOKENIZER_FILE = TOKENIZER_PATH / "tokenizer.json"
DATASET_PREFIX = TEST_RESULTS_PATH / "data" / "dataset/data"


CONFIG_BASE_FAST_LLM = [
    "training.logs.interval=1",
    "run.tensor_logs.save=True",
    "run.tensor_logs.show=False",
    "model.base_model.transformer.num_layers=2",
    "model.base_model.transformer.hidden_size=1024",
    "model.base_model.transformer.num_attention_heads=8",
    "model.base_model.transformer.init_method_std=0.022",
    "model.base_model.vocab_size=49152",
    f"model.multi_stage.debug_param_init={_LOG_LEVEL}",
    f"model.multi_stage.debug_layer_outputs={_LOG_LEVEL}",
    f"model.multi_stage.debug_layer_gradients={_LOG_LEVEL}",
    f"model.multi_stage.debug_all_param_gradients={_LOG_LEVEL}",
    "model.multi_stage.debug_tensor_parallel=True",
    "model.distributed.reproducible_init=True",
    "training.train_iters=2",
    "training.num_workers=4",
    "batch.batch_size=8",
    "batch.sequence_length=2048",
    f"data.path={DATASET_PREFIX}",
    "optimizer.learning_rate.base=0.0001",
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
    "--tokenizer-type=NullTokenizer",
    # Megatron messes with the vocab size, so we have to subtract 1.
    "--vocab-size=49151",
    f"--data-path={DATASET_PREFIX}",
    "--lr-decay-style=constant",
    # Initialization is set up to match MCore models (MCore inverts self-attn qkv and dense layers compared to original Megatron)
    "--use-mcore-models",
    # local implementation doesn't allow for RMS norm.
    "--transformer-impl=transformer_engine",
]

CONFIG_SC1_FAST_LLM = CONFIG_BASE_FAST_LLM + ["model.base_model.max_position_embeddings=2048"]
CONFIG_SC1_MEGATRON = CONFIG_BASE_MEGATRON + ["--group-query-attention"]
CONFIG_SC1_COMMON = CONFIG_SC1_FAST_LLM + ["model.distributed.training_dtype=bf16"]

CONFIG_GPT2_FAST_LLM = CONFIG_SC1_FAST_LLM + ["model.base_model.transformer.head_groups=8"]
CONFIG_GPT2_MEGATRON = CONFIG_BASE_MEGATRON
CONFIG_GPT2_COMMON = CONFIG_GPT2_FAST_LLM + ["model.distributed.training_dtype=bf16"]

CONFIG_SC2_FAST_LLM = CONFIG_BASE_FAST_LLM + [
    "model.base_model.transformer.head_groups=4",
    "model.base_model.transformer.use_rotary_embeddings=True",
]
CONFIG_SC2_MEGATRON = CONFIG_SC1_MEGATRON + [
    "--num-query-groups=4",
    "--use-rotary-position-embeddings",
    "--no-position-embedding",
]
CONFIG_SC2_COMMON = CONFIG_SC2_FAST_LLM + ["model.distributed.training_dtype=bf16"]

CONFIG_MISTRAL_MEGATRON = CONFIG_SC2_MEGATRON + [
    "--swiglu",
    "--disable-bias-linear",
    "--normalization=RMSNorm",
    "--ffn-hidden-size=4096",
    "--untie-embeddings-and-output-weights",
]
CONFIG_MISTRAL_FAST_LLM = CONFIG_SC2_FAST_LLM + [
    "model.base_model.transformer.gated=True",
    "model.base_model.transformer.activation_type=silu",
    "model.base_model.transformer.add_linear_biases=False",
    "model.base_model.transformer.normalization.type=rms_norm",
    "model.base_model.transformer.ffn_hidden_size=4096",
    "model.base_model.tie_word_embeddings=False",
]
CONFIG_MISTRAL_COMMON = CONFIG_MISTRAL_FAST_LLM + ["model.distributed.training_dtype=bf16"]

CONFIG_MIXTRAL_MEGATRON = CONFIG_MISTRAL_MEGATRON + [
    "--num-experts=4",
    "--moe-router-topk=4",
]
CONFIG_MIXTRAL_FAST_LLM = CONFIG_MISTRAL_FAST_LLM + [
    "model.base_model.transformer.num_experts=4",
    "model.base_model.transformer.num_experts_per_token=4",
]
CONFIG_MIXTRAL_COMMON = CONFIG_MIXTRAL_FAST_LLM + ["model.distributed.training_dtype=bf16"]

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


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def get_test_data():
    if not TOKENIZER_FILE.is_file():
        import transformers

        transformers.AutoTokenizer.from_pretrained("bigcode/santacoder").save_pretrained(TOKENIZER_PATH)

    if not (DATASET_PREFIX.with_suffix(".idx").is_file() and DATASET_PREFIX.with_suffix(".bin").is_file()):
        import transformers

        characters = (string.ascii_lowercase) * 5 + " " * 30 + "\n"
        documents = "".join(random.Random(1234).choices(characters, k=1000000)).splitlines()
        tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)

        documents = [np.array(tokenizer(document)["input_ids"], dtype=np.uint16) for document in documents]
        MMapIndexedDataset.write_dataset(DATASET_PREFIX, documents)


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
        command.append(f"run.experiment_dir={path}")
    print(" ".join(command))
    if skip:
        print("Reusing existing run.")
    else:
        get_test_data()
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
