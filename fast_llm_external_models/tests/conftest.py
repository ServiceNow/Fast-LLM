"""Shared test fixtures for fast_llm_external_models.

This conftest.py contains only fixtures that are shared across multiple model test suites.
Model-specific fixtures should be in the respective model's test directory
(e.g., test_apriel2/conftest.py, test_apriel_hybrid_ssm/conftest.py).
"""

import pytest
import torch


@pytest.fixture
def device():
    """Get available device (CPU or CUDA)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
