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
import yaml

from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset
from fast_llm.data.dataset.gpt.sampled import GPTSample
from fast_llm.layers.ssm.config import SSMConfig
from fast_llm.layers.transformer.config import TransformerConfig
from fast_llm.models.gpt.config import (
    LlamaGPTHuggingfaceCheckpointFormat,
    MistralGPTHuggingfaceCheckpointFormat,
    MixtralGPTHuggingfaceCheckpointFormat,
    MTPLlamaGPTHuggingfaceCheckpointFormat,
    Qwen2GPTHuggingfaceCheckpointFormat,
    Starcoder2GPTHuggingfaceCheckpointFormat,
)
from fast_llm.models.ssm.config import HybridSSMBaseModelConfig, LLambaHuggingfaceCheckpointFormat
from fast_llm.tools.train import CliTrainingConfig
from tests.compare_tensor_logs import CompareConfig, compare_tensor_logs

# FIXME: figure out correct import of megatron modules without this hack
sys.path.append(os.getcwd())

# TODO: Use `pytest_addoption` instead?
# Keep all results in one place to allow recovering them for debugging in case of failure.
TEST_RESULTS_PATH = pathlib.Path(os.environ.get("TEST_RESULTS_PATH", "/tmp/fast_llm_tests"))
FORCE_REUSE_RESULTS = int(os.environ.get("FORCE_REUSE_RESULTS", 0)) != 0
REUSE_RESULTS = FORCE_REUSE_RESULTS or int(os.environ.get("REUSE_RESULTS", 0)) != 0
_LOG_LEVEL = int(os.environ.get("LOG_LEVEL", 13))
TEST_MODEL = os.environ.get("MODEL", "llama")

ARTIFACT_PATH = "runs/0/artifacts"

TOKENIZER_PATH = TEST_RESULTS_PATH / "tokenizer" / "common"
TOKENIZER_FILE = TOKENIZER_PATH / "tokenizer.json"
DATASET_CACHE = TEST_RESULTS_PATH / "dataset"
DATASET_PREFIX = DATASET_CACHE / "common" / "dataset"
DATASET_SAMPLING_CACHE = TEST_RESULTS_PATH / "dataset" / "cache"

TEST_VOCAB_SIZE = 8192
# Random lowercase: 80.7% (3.1% each); space: 18.6%; doc end: 0.6%
TEST_CHARACTERS = (string.ascii_lowercase) * 5 + " " * 30 + "\n"
TEST_DATASET_TOKENS = 1000000

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
    f"--vocab-size={TEST_VOCAB_SIZE-1}",
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

# Megatron does not support Llama3-style Rotary Embeddings
CONFIG_LLAMA3_MEGATRON = None
CONFIG_LLAMA3_FAST_LLM = CONFIG_LLAMA_FAST_LLM + [
    "model.base_model.transformer.rotary.type=llama3",
]
CONFIG_LLAMA3_COMMON = CONFIG_LLAMA3_FAST_LLM + ["model.distributed.training_dtype=bf16"]

# Megatron does not support per sub layer biases
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

# Yarn-style Rotary Embeddings
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
    "llama-yarn": (
        "gpt",
        CONFIG_LLAMA_YARN_FAST_LLM,
        CONFIG_LLAMA_YARN_MEGATRON,
        CONFIG_LLAMA_YARN_COMMON,
        LlamaGPTHuggingfaceCheckpointFormat,
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


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def get_test_dataset(
    prefix: pathlib.Path = DATASET_PREFIX,
    seed: int = 1234,
    num_tokens: int = TEST_DATASET_TOKENS,
    characters: str = TEST_CHARACTERS,
    vocab_size: int = TEST_VOCAB_SIZE,
    max_spans: int = 0,
):
    if not TOKENIZER_FILE.is_file():
        import transformers

        transformers.AutoTokenizer.from_pretrained("bigcode/santacoder").save_pretrained(TOKENIZER_PATH)

    if not (
        prefix.with_suffix(".idx").is_file()
        and prefix.with_suffix(".bin").is_file()
        and prefix.parent.joinpath("fast_llm_config.yaml").is_file()
    ):
        import transformers

        texts = "".join(random.Random(seed).choices(characters, k=num_tokens)).splitlines()
        tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)

        samples = [
            GPTSample(np.array(tokenizer(document)["input_ids"], dtype=np.uint16) % vocab_size) for document in texts
        ]
        if max_spans > 0:
            lengths = np.array([max(len(sample.token_ids), 1) for sample in samples])
            spans = np.sort(np.random.RandomState(seed + 3847).randint(0, lengths[:, None], [len(samples), max_spans]))
            for sample, span in zip(samples, spans):
                span = np.unique(span)
                sample.loss_masking_spans = span[: len(span) // 2 * 2].reshape(-1, 2)

        GPTMemmapDataset.write_dataset(prefix, samples)
        yaml.safe_dump(
            {"type": "memmap", "path": prefix.name}, prefix.parent.joinpath("fast_llm_config.yaml").open("w")
        )


def get_test_concatenated_memmap_dataset(
    path: pathlib.Path,
    num_files: int,
    seed: int = 1234,
    num_tokens: int = TEST_DATASET_TOKENS,
    characters: str = TEST_CHARACTERS,
    vocab_size: int = TEST_VOCAB_SIZE,
    seed_shift: int = 55,
):
    index_file = path / "index.txt"
    if not index_file.is_file():
        for i in range(num_files):
            get_test_dataset(
                prefix=path / f"dataset_{i}",
                seed=seed + i * seed_shift,
                num_tokens=num_tokens,
                characters=characters,
                vocab_size=vocab_size,
            )
        index_file.open("w").writelines([str(path / f"dataset_{i}") + "\n" for i in range(num_files)])


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
    if is_megatron:
        script = [*script, f"--structured-logs-dir={path}", f"--data-cache-path={path}"]
    else:
        script = [model_type, *script, f"run.experiment_dir={path}"]
    header = ["Megatron-LM/pretrain_gpt.py"] if is_megatron else ["--no-python", "fast-llm", "train"]
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={num_gpus}",
        *header,
        *script,
    ]
    print(" ".join(command))
    if skip:
        print("Reusing existing run.")
    else:
        get_test_dataset()
        if num_gpus == 1 and not is_megatron:
            CliTrainingConfig.parse_and_run(script)
        else:
            completed_proc = subprocess.run(command, env=env, timeout=60)
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


def materialize_meta_tensors(model, tensor_space):
    # Materialize parameters that are on meta device
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            # Check if the parameter is a custom tensor type
            if hasattr(param, "tensor_name") and hasattr(param, "init_parameter"):
                param_data = param.new_empty(param.shape, device="cuda")
                # Initialize param_data
                param.init_parameter(param_data, tensor_space.distributed)
                # Replace the parameter in the module
                module_path, param_name = name.rsplit(".", 1) if "." in name else (None, name)
                module = model
                if module_path is not None:
                    for part in module_path.split("."):
                        module = getattr(module, part)
                param = torch.nn.Parameter(param_data, requires_grad=param.requires_grad)
                # TODO: add param_grad_is_zero etc., grad_buffer, etc., see test_mlp_recomputation
                param.grad = None
                param.grad_buffer = torch.empty_like(param)
                param.param_grad_is_zero = True
                module._parameters[param_name] = param
    return model


def get_hybrid_config(hybrid_block_layout=["t", "m"], prediction_heads=1, default_mtp_type=None):
    config = HybridSSMBaseModelConfig(
        transformer=TransformerConfig(num_layers=len(hybrid_block_layout)),
        ssm=SSMConfig(),
        hybrid_block_layout=hybrid_block_layout,
        prediction_heads=prediction_heads,
        default_mtp_type=default_mtp_type,
        init_method_std_embed=0.02,
        init_method_min_embed=-0.02,
        init_method_max_embed=0.02,
        use_position_embeddings=True,
        tie_word_embeddings=False,
    )
    return config
