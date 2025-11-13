# Integration Tests

These tests verify that real production models from the HuggingFace Hub can be converted to Fast-LLM format and produce equivalent forward pass results.

## Overview

The integration tests (`tests/integration/test_hub_integration.py`) perform the following steps:

1. **Download real models** from HuggingFace Hub
2. **Truncate to first N layers** to reduce memory requirements (default: 2 layers)
3. **Convert to Fast-LLM format**
4. **Verify forward pass equivalence** between HuggingFace and Fast-LLM implementations
5. **Test implementation variants** where applicable (e.g., different kernel paths)

## Test Flow (with Dependencies)

Tests are organized with pytest dependencies to ensure proper execution order:

1. `test_download_and_truncate_{model}` - Downloads and truncates model
2. `test_conversion_{model}` - Converts to Fast-LLM (depends on step 1)
3. `test_forward_equivalence_{model}` - Compares outputs (depends on step 2)
4. `test_{variant}_implementation_{model}` - Tests implementation variants (depends on step 2)

## Why Skip by Default?

These tests are marked with `@pytest.mark.extra_slow` and are **skipped by default** because they:
- Download large models from the Hub (multi-GB downloads)
- Require significant GPU memory
- Take considerable time to run

## Running the Tests

### Run all integration tests:
```bash
pytest tests/integration --run-extra-slow
```

### Run a specific test:
```bash
pytest tests/integration/test_hub_integration.py::test_hub_model_conversion --run-extra-slow
```

### Run with specific model:
```bash
pytest tests/integration -k mixtral --run-extra-slow
```

### Run only implementation variant tests:
```bash
pytest tests/integration -k "test_moe_implementation or test_implementation" --run-extra-slow
```

### Run with verbose output:
```bash
pytest tests/integration --run-extra-slow -v -s
```

## Test Structure

### Test Functions

1. **`test_download_and_truncate`**
   - Downloads model from HuggingFace Hub
   - Truncates to first N layers to reduce memory
   - Verifies config is updated correctly

2. **`test_conversion`**
   - Converts truncated model to Fast-LLM format
   - Verifies checkpoint files exist

3. **`test_forward_equivalence`**
   - Compares forward pass outputs between HF and Fast-LLM
   - Uses CompareConfig with appropriate thresholds
   - Scales thresholds per model as needed

4. **`test_moe_implementation` (or other variants)**
   - Parametrized tests for implementation variants
   - Verifies all variants produce correct results
   - Critical for ensuring correctness after code changes

### Fixtures

- **`hub_test_cache_dir`**: Temporary directory with automatic cleanup
- **`model_name`**: Parametrized fixture for model names
- **`model_config`**: Configuration for specific model
- **`truncated_hf_path`**: Downloads and truncates model from Hub
- **`fast_llm_path`**: Converts to Fast-LLM with default settings
- **`fast_llm_path_{variant}`**: Converts with specific variant settings

## Supported Models

Currently supported models in `HUB_TEST_CONFIGS`:

- **Mixtral** (`mistralai/Mixtral-8x7B-v0.1`)
  - Truncated to 2 layers
  - Tests conversion and implementation variants
  - Compare factor: 2.0

## Adding New Models

To add a new model to the integration tests:

1. Add configuration to `HUB_TEST_CONFIGS`:

```python
HUB_TEST_CONFIGS["model_name"] = {
    "model_id": "org/model-name",  # HuggingFace Hub ID
    "checkpoint_format": ModelCheckpointFormat,  # Format class
    "model_config": GPTModelConfig,  # Model config class
    "num_layers_to_keep": 2,  # Number of layers after truncation
    "test_params": {
        "batch_size": 2,
        "sequence_length": 32,
        "compare_factor": 1.0,  # Increase for models with higher numerical error
    },
}
```

2. Add the model name to the `model_name` fixture parameters:

```python
@pytest.fixture(scope="module", params=["mixtral", "model_name"])
def model_name(request):
    return request.param
```

3. (Optional) Add variant-specific fixtures and tests if the model has multiple implementation paths

## Requirements

- **GPU Memory**: Tests require sufficient GPU memory (varies by model)
- **Disk Space**: Models are cached in temp directory during tests
- **Network**: HuggingFace Hub access for model downloads

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size or sequence length in test_params, or use a larger GPU.

### Download Failures
Check HuggingFace Hub access and network connectivity. Models may require authentication for gated models.

### Comparison Failures
- Check if recent code changes affected model implementations or conversion logic
- Verify compare_factor is appropriate for the model architecture
- Review error messages for specific tensor mismatches
- Compare against known baseline results if available

## Development Workflow

After making changes to model code or conversion logic:

1. Run local unit tests first
2. Run integration tests to verify real models still work:
   ```bash
   pytest tests/integration -k model_name --run-extra-slow
   ```
3. If tests fail, investigate numerical differences or conversion issues
4. Update compare thresholds only if the differences are acceptable and understood

## CI/CD Integration

These tests are **not part of the regular CI pipeline** due to their resource requirements. They should be run:

- **Manually** before major releases
- **After significant changes** to model implementations or conversion code
- **Periodically** to ensure compatibility with upstream models

To run in CI (if infrastructure supports it):
```bash
pytest tests/integration --run-extra-slow --tb=short
```
