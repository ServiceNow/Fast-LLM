import pathlib
import pytest
import subprocess
import unittest.mock
import yaml


from fast_llm.layers.transformer.config import (
    TransformerConfig,
    TransformerArchitectureConfig,
    TransformerSubLayerKeys,
)
from fast_llm.utils import Assert
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.config_utils.data_type import DataType

from fast_llm.models.auto import trainer_registry


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


@pytest.fixture
def config_with_true_biases():
    """Fixture for TransformerArchitectureConfig with True add_linear_biases."""
    return TransformerArchitectureConfig(add_linear_biases=True)


@pytest.fixture
def config_with_false_biases():
    """Fixture for TransformerArchitectureConfig with False add_linear_biases."""
    return TransformerArchitectureConfig(add_linear_biases=False)


@pytest.fixture
def config_with_dict_biases():
    """Fixture for TransformerArchitectureConfig with dict add_linear_biases."""
    return TransformerArchitectureConfig(
        num_layers = 10, 
        add_linear_biases={
            "layers.self_attn.query": "*",
            "layers.mlp.layer_1": "1:10:3, 9",
            "layers.mlp.layer_2": "5:7",
        }
    )


def test_add_linear_biases_bool_true(config_with_true_biases):
    """Test case for add_linear_biases set to True (default)."""
    assert config_with_true_biases._parsed_add_linear_biases == True


def test_add_linear_biases_bool_false(config_with_false_biases):
    """Test case for add_linear_biases set to False."""
    assert config_with_false_biases._parsed_add_linear_biases == False


def test_add_linear_biases_dict_valid(config_with_dict_biases):
    """Test case for add_linear_biases with valid dictionary."""
    assert config_with_dict_biases._parsed_add_linear_biases == {
        TransformerSubLayerKeys.attn_query: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        TransformerSubLayerKeys.mlp_layer1: {2, 5, 8, 10},
        TransformerSubLayerKeys.mlp_layer2: {6, 7},
    }


def test_invalid_key_in_dict():
    """Test case where an invalid key is provided in add_linear_biases dictionary."""
    with pytest.raises(AssertionError):
        # Using an invalid key in the dictionary.
        TransformerArchitectureConfig(add_linear_biases={"invalid_key": "*"})


def test_invalid_range_format():
    """Test case where invalid range format is provided."""
    with pytest.raises(AssertionError):
        TransformerArchitectureConfig(add_linear_biases={TransformerSubLayerKeys.mlp_layer1: "1:10:3, abc"})


def test_empty_add_linear_biases():
    """Test case for empty add_linear_biases dictionary."""
    with pytest.raises(AssertionError):  # Expecting AssertionError for invalid empty dictionary
        TransformerArchitectureConfig(add_linear_biases={})


def test_should_add_linear_bias_for_layer_and_sublayer(config_with_dict_biases):
    """Test case for should_add_linear_bias based on layer index and sublayer key."""

    # Layer 1 and sublayer mlp_layer1
    assert config_with_dict_biases.should_add_linear_bias(1, TransformerSubLayerKeys.mlp_layer1) == False

    # Layer 2 and sublayer mlp_layer1
    assert config_with_dict_biases.should_add_linear_bias(2, TransformerSubLayerKeys.mlp_layer1) == True

    # Layer 9 and sublayer mlp_layer1
    assert config_with_dict_biases.should_add_linear_bias(9, TransformerSubLayerKeys.mlp_layer1) == False

    # Layer 6 and sublayer mlp_layer2
    assert config_with_dict_biases.should_add_linear_bias(6, TransformerSubLayerKeys.mlp_layer2) == True

    # Layer 5 and sublayer attn_query
    assert config_with_dict_biases.should_add_linear_bias(5, TransformerSubLayerKeys.attn_query) == True


def test_should_add_linear_bias_for_bool_true(config_with_true_biases):
    """Test case for add_linear_biases set to True (should always return True)."""
    assert config_with_true_biases.should_add_linear_bias(10, TransformerSubLayerKeys.mlp_layer1) == True


def test_should_add_linear_bias_for_bool_false(config_with_false_biases):
    """Test case for add_linear_biases set to False (should always return False)."""
    assert config_with_false_biases.should_add_linear_bias(10, TransformerSubLayerKeys.mlp_layer1) == False

