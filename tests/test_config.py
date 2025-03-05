import pathlib
import subprocess

import pytest
import yaml

from fast_llm.config import ValidationError
from fast_llm.layers.transformer.config import AddLinearBiasChoices, TransformerLayerArchitectureConfig
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


def test_add_linear_biases_valid_values():
    # Valid boolean values
    assert TransformerLayerArchitectureConfig(add_linear_biases=True).add_linear_biases is True
    assert TransformerLayerArchitectureConfig(add_linear_biases=False).add_linear_biases is False

    # Valid enum values
    assert (
        TransformerLayerArchitectureConfig(add_linear_biases="nowhere").add_linear_biases
        == AddLinearBiasChoices.nowhere
    )
    assert (
        TransformerLayerArchitectureConfig(add_linear_biases="everywhere").add_linear_biases
        == AddLinearBiasChoices.everywhere
    )
    assert (
        TransformerLayerArchitectureConfig(add_linear_biases="only_attn_qkv").add_linear_biases
        == AddLinearBiasChoices.only_attn_qkv
    )


def test_add_linear_biases_invalid_values():
    with pytest.raises(ValidationError):
        TransformerLayerArchitectureConfig(add_linear_biases="invalid_value")

    with pytest.raises(ValidationError):
        TransformerLayerArchitectureConfig(add_linear_biases=123)

    with pytest.raises(ValidationError):
        TransformerLayerArchitectureConfig(add_linear_biases=None)


def test_add_mlp_bias():
    assert TransformerLayerArchitectureConfig(add_linear_biases=True).add_mlp_bias is True
    assert TransformerLayerArchitectureConfig(add_linear_biases=False).add_mlp_bias is False
    assert TransformerLayerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.everywhere).add_mlp_bias is True
    assert TransformerLayerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.nowhere).add_mlp_bias is False
    assert (
        TransformerLayerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.only_attn_qkv).add_mlp_bias is False
    )


def test_add_attn_qkv_bias():
    assert TransformerLayerArchitectureConfig(add_linear_biases=True).add_attn_qkv_bias is True
    assert TransformerLayerArchitectureConfig(add_linear_biases=False).add_attn_qkv_bias is False
    assert (
        TransformerLayerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.everywhere).add_attn_qkv_bias is True
    )
    assert (
        TransformerLayerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.nowhere).add_attn_qkv_bias is False
    )
    assert (
        TransformerLayerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.only_attn_qkv).add_attn_qkv_bias
        is True
    )


def test_add_attn_dense_bias():
    assert TransformerLayerArchitectureConfig(add_linear_biases=True).add_attn_dense_bias is True
    assert TransformerLayerArchitectureConfig(add_linear_biases=False).add_attn_dense_bias is False
    assert (
        TransformerLayerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.everywhere).add_attn_dense_bias
        is True
    )
    assert (
        TransformerLayerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.nowhere).add_attn_dense_bias is False
    )
    assert (
        TransformerLayerArchitectureConfig(add_linear_biases=AddLinearBiasChoices.only_attn_qkv).add_attn_dense_bias
        is False
    )
