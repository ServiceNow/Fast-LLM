"""
Integration tests for HuggingFace Hub model conversion and forward pass equivalence.

These tests download real production models from HuggingFace Hub, truncate them to a small
number of layers to reduce memory requirements, convert them to Fast-LLM format, and verify
that the forward passes produce equivalent results to the original HuggingFace implementation.

Test flow (with pytest dependencies):
1. test_download_and_truncate_{model} - Downloads and truncates model from Hub
2. test_conversion_{model} - Converts to Fast-LLM format (depends on step 1)
3. test_forward_equivalence_{model} - Compares HF vs Fast-LLM outputs (depends on step 2)
4. test_{variant}_implementation_{model} - Tests implementation variants (depends on step 2)

These tests are marked as @pytest.mark.extra_slow and are skipped by default.
Run with: pytest tests/integration --run-extra-slow
"""

import logging
import pathlib
import shutil

import pytest
import torch
import transformers
from huggingface_hub import snapshot_download

from fast_llm.engine.checkpoint.config import (
    CheckpointLoadConfig,
    CheckpointSaveConfig,
    FastLLMCheckpointFormat,
    ModelConfigType,
)
from fast_llm.engine.checkpoint.convert import ConvertConfig
from fast_llm.engine.config_utils.logging import TensorLogs, TensorLogsConfig
from fast_llm.logging import set_model_debug_level
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.models.gpt.conversion.config import MixtralCheckpointFormat
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
from fast_llm.models.gpt.model import GPTModel
from tests.utils.compare_tensor_logs import CompareConfig
from tests.utils.utils import requires_cuda

logger = logging.getLogger(__name__)


# Model configurations for hub integration tests
HUB_TEST_CONFIGS = {
    "mixtral": {
        "model_id": "mistralai/Mixtral-8x7B-v0.1",
        "checkpoint_format": MixtralCheckpointFormat,
        "model_config": GPTModelConfig,
        "num_layers_to_keep": 2,  # Truncate to 2 layers to reduce memory
        "test_params": {
            "batch_size": 2,
            "sequence_length": 32,
            "compare_factor": 2.0,  # MoE models have higher numerical error
        },
    },
}


@pytest.fixture(scope="module", autouse=True)
def reset_gpu_memory_limit():
    """Reset GPU memory limit for integration tests (they need the full model)."""
    if torch.cuda.is_available():
        # Reset to allow full GPU memory (tests/conftest.py limits to 5GB by default)
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
    yield


@pytest.fixture(scope="module")
def hub_test_cache_dir(tmp_path_factory):
    """Create a cache directory for hub integration tests."""
    cache_dir = tmp_path_factory.mktemp("hub_integration_cache")
    yield cache_dir
    # Cleanup after all tests complete
    if cache_dir.exists():
        logger.info(f"Cleaning up cache directory: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture(scope="module", params=["mixtral"])
def model_name(request):
    """Parametrized fixture for model names."""
    return request.param


@pytest.fixture(scope="module")
def model_config(model_name):
    """Get configuration for a specific model."""
    if model_name not in HUB_TEST_CONFIGS:
        pytest.skip(f"Unknown model: {model_name}")
    return HUB_TEST_CONFIGS[model_name]


@pytest.fixture(scope="module")
def truncated_hf_path(hub_test_cache_dir, model_name, model_config):
    """
    Download model from HF Hub and truncate to first N layers to reduce memory.

    Steps:
    1. Download from HuggingFace Hub
    2. Load model (with any necessary dequantization)
    3. Truncate to num_layers_to_keep
    4. Update config (including model-specific fields)
    5. Save truncated model
    """
    model_id = model_config["model_id"]
    num_layers = model_config["num_layers_to_keep"]
    truncated_path = hub_test_cache_dir / f"{model_name}_truncated"

    if truncated_path.exists():
        logger.info(f"Truncated model already exists at {truncated_path}")
        return truncated_path

    logger.info(f"Downloading and truncating {model_id} to {num_layers} layers...")

    # Download from HF Hub
    logger.info(f"  Downloading from Hub: {model_id}")
    hf_local_path = snapshot_download(repo_id=model_id, local_dir_use_symlinks=False)
    hf_local_path = pathlib.Path(hf_local_path)

    # Load model on CPU to avoid OOM when loading full model
    logger.info(f"  Loading model on CPU...")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        hf_local_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # Truncate to first N layers
    logger.info(f"  Truncating to {num_layers} layers...")
    original_num_layers = len(hf_model.model.layers)
    logger.info(f"    Original layers: {original_num_layers}, keeping: {num_layers}")
    hf_model.model.layers = hf_model.model.layers[:num_layers]
    hf_model.config.num_hidden_layers = num_layers

    # Handle model-specific config updates (e.g., layer_types for GPT-OSS)
    if hasattr(hf_model.config, "layer_types"):
        hf_model.config.layer_types = hf_model.config.layer_types[:num_layers]
        logger.info(f"    Updated layer_types: {hf_model.config.layer_types}")

    # Save truncated model
    logger.info(f"  Saving truncated model to {truncated_path}")
    hf_model.save_pretrained(truncated_path)

    # Also save tokenizer if available
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(hf_local_path, trust_remote_code=True)
        tokenizer.save_pretrained(truncated_path)
    except Exception as e:
        logger.warning(f"  Failed to save tokenizer: {e}")

    logger.info(f"✓ Truncated model saved to {truncated_path}")
    logger.info(f"  Vocab size: {hf_model.config.vocab_size}")
    logger.info(f"  Hidden size: {hf_model.config.hidden_size}")
    logger.info(f"  Num layers: {hf_model.config.num_hidden_layers}")

    # Free CPU memory
    del hf_model

    return truncated_path


@pytest.fixture(scope="module")
def fast_llm_path(hub_test_cache_dir, model_name, model_config, truncated_hf_path):
    """Convert truncated HF model to Fast-LLM format (default MoE settings)."""
    fast_llm_path = hub_test_cache_dir / f"{model_name}_fast_llm"

    if fast_llm_path.exists():
        logger.info(f"Fast-LLM checkpoint already exists at {fast_llm_path}")
        return fast_llm_path

    logger.info(f"Converting {model_name} to Fast-LLM format (on CPU)...")

    ConvertConfig(
        input=CheckpointLoadConfig(
            path=truncated_hf_path,
            format=model_config["checkpoint_format"],
            load_config=ModelConfigType.model,
        ),
        output=CheckpointSaveConfig(
            path=fast_llm_path,
            format=FastLLMCheckpointFormat,
        ),
        model=model_config["model_config"],
        use_cpu=True,  # Convert on CPU to avoid OOM
    ).run()

    logger.info(f"✓ Converted to {fast_llm_path}")
    return fast_llm_path




# ============================================================================
# Test 1: Download and Truncate
# ============================================================================


@requires_cuda
@pytest.mark.extra_slow
def test_download_and_truncate(model_name, model_config, truncated_hf_path):
    """Test that model can be downloaded and truncated."""
    assert truncated_hf_path.exists(), f"Truncated model not found at {truncated_hf_path}"
    assert (truncated_hf_path / "config.json").exists(), "config.json not found"

    # Verify the truncation worked
    config = transformers.AutoConfig.from_pretrained(truncated_hf_path, trust_remote_code=True)
    expected_layers = model_config["num_layers_to_keep"]
    assert config.num_hidden_layers == expected_layers, (
        f"Expected {expected_layers} layers, got {config.num_hidden_layers}"
    )
    logger.info(f"✓ Model truncated to {config.num_hidden_layers} layers")


# ============================================================================
# Test 2: Conversion
# ============================================================================


@requires_cuda
@pytest.mark.extra_slow
@pytest.mark.depends_on(on=["test_download_and_truncate[{model_name}]"])
def test_conversion(model_name, fast_llm_path):
    """Test that truncated model can be converted to Fast-LLM format."""
    assert fast_llm_path.exists(), f"Fast-LLM checkpoint not found at {fast_llm_path}"
    assert (fast_llm_path / "metadata.yaml").exists(), "metadata.yaml not found"
    logger.info(f"✓ Conversion successful: {fast_llm_path}")


# ============================================================================
# Test 3: Forward Pass Equivalence
# ============================================================================


@requires_cuda
@pytest.mark.extra_slow
@pytest.mark.depends_on(on=["test_conversion[{model_name}]"])
def test_forward_equivalence(model_name, model_config, truncated_hf_path, fast_llm_path):
    """Test that HuggingFace and Fast-LLM produce equivalent forward pass results."""
    test_params = model_config["test_params"]
    batch_size = test_params["batch_size"]
    sequence_length = test_params["sequence_length"]
    compare_factor = test_params.get("compare_factor", 1.0)

    # Load HF config to get vocab size
    hf_config = transformers.AutoConfig.from_pretrained(truncated_hf_path, trust_remote_code=True)
    vocab_size = hf_config.vocab_size

    # Create test input
    torch.manual_seed(42)
    test_input = torch.randint(
        0,
        vocab_size,
        size=(batch_size, sequence_length),
        dtype=torch.int64,
        device="cuda",
    )

    # Run HuggingFace model
    logger.info("Loading HuggingFace model...")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        truncated_hf_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda()

    with torch.no_grad():
        hf_output = hf_model(test_input)

    hf_logits = hf_output.logits.clone().cpu()

    # Cleanup HF model
    del hf_model, hf_output
    torch.cuda.empty_cache()

    # Run Fast-LLM model
    logger.info("Loading Fast-LLM model...")
    TensorLogs.reset(TensorLogsConfig(save=False, show=False))
    set_model_debug_level(0)

    gpt_model = GPTModel.from_pretrained(
        CheckpointLoadConfig(
            path=fast_llm_path,
            format=FastLLMCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )
    fast_llm_model = HuggingfaceGPTModelForCausalLM(gpt_model)

    with torch.no_grad():
        fast_llm_output = fast_llm_model(test_input)

    fast_llm_logits = fast_llm_output.logits.clone()

    # Compare outputs
    logger.info("Comparing outputs...")
    hf_logits = hf_logits.cuda()

    errors = []
    compare_config = CompareConfig()
    if compare_factor != 1.0:
        # Scale thresholds for models with higher numerical error (e.g., MoE)
        compare_config = CompareConfig(
            max_rms_diff_abs=compare_config.max_rms_diff_abs * compare_factor,
            max_rms_diff_scaled=compare_config.max_rms_diff_scaled * compare_factor,
            max_max_diff_abs=compare_config.max_max_diff_abs * compare_factor,
            max_max_diff_scaled=compare_config.max_max_diff_scaled * compare_factor,
        )

    compare_config.compare_tensors(
        {"samples": hf_logits, "shape": hf_logits.shape, "step": 0},
        {"samples": fast_llm_logits, "shape": fast_llm_logits.shape, "step": 0},
        errors,
        f"{model_name}_HF_vs_FastLLM",
        "logits",
    )

    if errors:
        for error in errors:
            logger.error(error)
        pytest.fail(f"Forward pass comparison failed with {len(errors)} errors")

    logger.info(f"✓ Forward pass equivalence test passed for {model_name}")


# ============================================================================
# Test 4: MoE Implementation Variants (Dropless vs Looped)
# ============================================================================


@requires_cuda
@pytest.mark.extra_slow
@pytest.mark.depends_on(on=["test_conversion[{model_name}]"])
def test_moe_implementation(model_name, model_config, fast_llm_path):
    """Test that dropless and looped MoE implementations produce equivalent results."""
    # Only run for MoE models
    if model_name not in ["mixtral"]:
        pytest.skip(f"MoE implementation test not applicable for {model_name}")

    test_params = model_config["test_params"]
    batch_size = test_params["batch_size"]
    sequence_length = test_params["sequence_length"]
    compare_factor = test_params.get("compare_factor", 1.0)

    # Load config to get vocab size
    import yaml
    with open(fast_llm_path / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)
    vocab_size = metadata["config"]["base_model"]["embeddings"]["vocab_size"]

    # Create test input
    torch.manual_seed(42)
    test_input = torch.randint(
        0,
        vocab_size,
        size=(batch_size, sequence_length),
        dtype=torch.int64,
        device="cuda",
    )

    # Test both implementations
    outputs = {}
    for variant_name, dropless_value in [("dropless", True), ("looped", False)]:
        logger.info(f"Testing {variant_name} MoE implementation (dropless={dropless_value})...")
        TensorLogs.reset(TensorLogsConfig(save=False, show=False))
        set_model_debug_level(0)

        # Load model with config override
        gpt_model = GPTModel.from_pretrained(
            CheckpointLoadConfig(
                path=fast_llm_path,
                format=FastLLMCheckpointFormat,
                load_config=ModelConfigType.model,
            ),
            {("base_model", "decoder", "block", "mlp", "dropless"): dropless_value},
        )
        fast_llm_model = HuggingfaceGPTModelForCausalLM(gpt_model)

        with torch.no_grad():
            output = fast_llm_model(test_input)

        outputs[variant_name] = output.logits.clone()

        # Cleanup
        del gpt_model, fast_llm_model, output
        torch.cuda.empty_cache()

        logger.info(f"✓ {variant_name} implementation forward pass complete")

    # Compare dropless vs looped implementations
    logger.info("Comparing dropless vs looped implementations...")
    errors = []
    compare_config = CompareConfig()
    if compare_factor != 1.0:
        # Scale thresholds for models with higher numerical error
        compare_config = CompareConfig(
            max_rms_diff_abs=compare_config.max_rms_diff_abs * compare_factor,
            max_rms_diff_scaled=compare_config.max_rms_diff_scaled * compare_factor,
            max_max_diff_abs=compare_config.max_max_diff_abs * compare_factor,
            max_max_diff_scaled=compare_config.max_max_diff_scaled * compare_factor,
        )

    compare_config.compare_tensors(
        {"samples": outputs["dropless"], "shape": outputs["dropless"].shape, "step": 0},
        {"samples": outputs["looped"], "shape": outputs["looped"].shape, "step": 0},
        errors,
        f"{model_name}_dropless_vs_looped",
        "logits",
    )

    if errors:
        for error in errors:
            logger.error(error)
        pytest.fail(f"MoE implementation comparison failed with {len(errors)} errors")

    logger.info(f"✓ MoE implementation variant test passed for {model_name}")


