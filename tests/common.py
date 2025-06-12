import os
import sys

# FIXME: figure out correct import of megatron modules without this hack
sys.path.append(os.getcwd())

# TODO: Use `pytest_addoption` instead?
# Keep all results in one place to allow recovering them for debugging in case of failure.

# Random lowercase: 80.7% (3.1% each); space: 18.6%; doc end: 0.6%

# Megatron does not support Llama3-style Rotary Embeddings

# Megatron does not support per sub layer biases

# Yarn-style Rotary Embeddings
