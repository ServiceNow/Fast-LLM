import pathlib
import subprocess

import pytest
import yaml

from fast_llm.config import NoAutoValidate
from fast_llm.data.dataset.gpt.config import GPTSamplingConfig
from fast_llm.engine.checkpoint.config import CheckpointSaveMetadataConfig, ModelConfigType
from fast_llm.engine.distributed.config import DistributedConfig
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
                "transformer": {
                    "normalization": {"type": "rms_norm"},  # Nested
                    "rotary": {"type": "default"},
                    "num_layers": 12,  # Default
                    "hidden_size": 1024,  # Default
                    "window_size": 32,
                    "ffn_hidden_size": 4096,  # Implicit default, default value
                    "activation_type": "silu",  # Implicit default, non-default value
                    "head_groups": 4,
                },
                "tie_word_embeddings": False,
            },
            "multi_stage": {"zero_stage": 3},
            "distributed": {"training_dtype": "bfloat16"},
        }
    )
    with NoAutoValidate():
        save_config = CheckpointSaveMetadataConfig.from_dict({"format": "fast_llm", "path": config_path})
    save_config.setup(GPTModelConfig)
    save_config.validate()
    pretrained_model_config.save_metadata(save_config)

    base_model_update = {
        "transformer": {
            # rotary: Don't override nested.
            "normalization": {"implementation": "triton"},  # Update non-default nested
            "peft": {"type": "lora", "freeze_others": False},  # Update default nested, change type
            "hidden_size": 512,  # Override, affects derived value (kv channels)
            "head_groups": 1,  # Override to default
        },
        "vocab_size": 1000,
    }
    pretrained_config = PretrainedGPTModelConfig.from_dict(
        {
            "model": {
                "base_model": base_model_update,
                "distributed": {"seed": 1234, "training_dtype": "float16"},
            },
            "pretrained": {"format": "fast_llm", "path": config_path, "load_config": load_config},
        }
    )
    Assert.eq(pretrained_config.model.base_model.transformer.kv_channels, 64)
    serialized_config = pretrained_config.model.to_dict()
    expected_config = {"type": "gpt", "distributed": DistributedConfig().to_dict()}

    if load_config == ModelConfigType.fast_llm:
        expected_config["multi_stage"] = {"zero_stage": 3}
    expected_config["distributed"].update({"seed": 1234, "training_dtype": "float16"})
    if load_config in (ModelConfigType.fast_llm, ModelConfigType.model):
        expected_config["base_model"] = {
            "transformer": {
                "normalization": {"type": "rms_norm", "implementation": "triton"},
                "rotary": {"type": "default"},
                "peft": {"type": "lora", "freeze_others": False, "layers": ["query", "value"]},
                "num_layers": 12,
                "hidden_size": 512,
                "ffn_hidden_size": 4096,
                "activation_type": "silu",
                "head_groups": 1,
                "window_size": 32,
            },
            "tie_word_embeddings": False,
            "vocab_size": 1000,
        }
    else:
        base_model_update["transformer"]["peft"] = {
            "type": "lora",
            "freeze_others": False,
            "layers": ["query", "value"],
        }
        base_model_update["transformer"]["normalization"]["type"] = "layer_norm"
        base_model_update["transformer"]["rotary"] = {"type": "none"}
        expected_config["base_model"] = base_model_update

    check_equal_nested(serialized_config, expected_config)
