import pathlib
import subprocess
import unittest.mock

import pytest
import yaml

from fast_llm.config import NoAutoValidate
from fast_llm.data.dataset.gpt.config import GPTSamplingConfig
from fast_llm.engine.checkpoint.config import CheckpointSaveMetadataConfig, ModelConfigType
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.transformer.config import AddLinearBiasChoices, TransformerArchitectureConfig, TransformerConfig
from fast_llm.models.auto import trainer_registry
from fast_llm.models.gpt.config import GPTModelConfig, PretrainedGPTModelConfig
from fast_llm.utils import Assert, check_equal_nested
from tests.common import TEST_RESULTS_PATH


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
                "from fast_llm.tools.cli import fast_llm as main",
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
    trainer_registry["gpt"].from_dict(fast_llm_config_dict)


def test_do_use_flash_attention():
    # Create a mock DistributedConfig
    mock_distributed_config = unittest.mock.Mock(spec=DistributedConfig)

    # Test case 1: use_flash_attention is True and training_dtype is float16
    config = TransformerConfig(use_flash_attention=True, window_size=None)
    mock_distributed_config.training_dtype = DataType.float16
    assert config.do_use_flash_attention(mock_distributed_config) is True

    # Test case 2: use_flash_attention is False
    config = TransformerConfig(use_flash_attention=False, window_size=None)
    mock_distributed_config.training_dtype = DataType.float16
    assert config.do_use_flash_attention(mock_distributed_config) is False

    # Test case 3: use_flash_attention is True but training_dtype is not float16 or bfloat16
    config = TransformerConfig(use_flash_attention=True, window_size=None)
    mock_distributed_config.training_dtype = DataType.float32
    assert config.do_use_flash_attention(mock_distributed_config) is False

    # Test case 4: use_flash_attention is False and window_size is not None
    config = TransformerConfig(use_flash_attention=False, window_size=512)
    mock_distributed_config.training_dtype = DataType.float32
    with pytest.raises(AssertionError):
        config.do_use_flash_attention(mock_distributed_config)


def test_add_mlp_bias():
    assert TransformerArchitectureConfig(add_linear_biases=True).add_mlp_bias is True
    assert TransformerArchitectureConfig(add_linear_biases=False).add_mlp_bias is False
    assert TransformerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.everywhere).add_mlp_bias is True
    assert TransformerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.nowhere).add_mlp_bias is False
    assert TransformerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.only_attn_qkv).add_mlp_bias is False


def test_add_attn_qkv_bias():
    assert TransformerArchitectureConfig(add_linear_biases=True).add_attn_qkv_bias is True
    assert TransformerArchitectureConfig(add_linear_biases=False).add_attn_qkv_bias is False
    assert TransformerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.everywhere).add_attn_qkv_bias is True
    assert TransformerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.nowhere).add_attn_qkv_bias is False
    assert (
        TransformerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.only_attn_qkv).add_attn_qkv_bias is True
    )


def test_add_attn_dense_bias():
    assert TransformerArchitectureConfig(add_linear_biases=True).add_attn_dense_bias is True
    assert TransformerArchitectureConfig(add_linear_biases=False).add_attn_dense_bias is False
    assert TransformerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.everywhere).add_attn_dense_bias is True
    assert TransformerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.nowhere).add_attn_dense_bias is False
    assert (
        TransformerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.only_attn_qkv).add_attn_dense_bias
        is False
    )


@pytest.mark.parametrize(
    ("cls", "default"),
    ((GPTSamplingConfig, {}), (GPTModelConfig, {"distributed": {"world_size": 1, "rank": 0, "local_world_size": 1}})),
)
def test_serialize_default_config_updates(cls, default):
    # Config classes used as config updates should have a default that serializes to an empty dict
    #   so no value is incorrectly overridden.
    check_equal_nested(cls.from_dict({}).to_dict(), default)


@pytest.mark.parametrize("load_config", tuple(ModelConfigType))
def test_pretrained_config(load_config: ModelConfigType):
    config_path = TEST_RESULTS_PATH / "pretrained_config"
    pretrained_model_config = GPTModelConfig.from_dict(
        {
            "base_model": {
                "transformer": {
                    "normalization": {"type": "rms_norm"},  # Nested
                    "rotary": {"type": "default"},
                    "num_layers": 12,  # Default
                    "hidden_size": 1024,  # Default
                    "window_size": 32,  # Non-architecture
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
            "peft": {"freeze_others": False},  # Update default nested, non-architecture
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
    expected_config = {"distributed": DistributedConfig().to_dict()}

    if load_config == ModelConfigType.fast_llm:
        expected_config["multi_stage"] = {"zero_stage": 3}
    expected_config["distributed"].update({"seed": 1234, "training_dtype": "float16"})
    if load_config in (ModelConfigType.architecture, ModelConfigType.fast_llm, ModelConfigType.model):
        expected_config["base_model"] = {
            "transformer": {
                "normalization": {"type": "rms_norm", "implementation": "triton"},
                "rotary": {"type": "default"},
                "peft": {"freeze_others": False},
                "num_layers": 12,
                "hidden_size": 512,
                "ffn_hidden_size": 4096,
                "activation_type": "silu",
                "head_groups": 1,
            },
            "tie_word_embeddings": False,
            "vocab_size": 1000,
        }
        if load_config != ModelConfigType.architecture:
            expected_config["base_model"]["transformer"]["window_size"] = 32
    else:
        expected_config["base_model"] = base_model_update

    check_equal_nested(serialized_config, expected_config)
