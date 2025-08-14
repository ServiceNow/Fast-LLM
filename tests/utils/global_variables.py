"""
This files holds global variables and settings that need to be defined before importing any third-party package.
They are kept in a separate file to prevent circular imports.
"""

import os
import pathlib
import string

from fast_llm.utils import set_global_variables

# Directory for all test data and results.
# Cannot be a fixture because it's used outside testing environment (ex. distributed scripts).
TEST_RESULTS_PATH = pathlib.Path("/tmp/fast_llm_tests")

WORKER_NAME = os.environ.get("PYTEST_XDIST_WORKER")
GPUS = os.environ.get("CUDA_VISIBLE_DEVICES")
SHARED_RESULT_PATH = TEST_RESULTS_PATH / (f"common_{WORKER_NAME}" if WORKER_NAME else "common")


def set_testing_global_variables():
    set_global_variables()  # isort: skip
    if WORKER_NAME:
        if gpus := os.environ.get("CUDA_VISIBLE_DEVICES"):
            # We set the device through "CUDA_VISIBLE_DEVICES", and this needs to happen before importing torch.
            assert WORKER_NAME.startswith("gw")
            worker_id = int(WORKER_NAME[2:])
            gpus = [int(i) for i in gpus.split(",")]
            num_gpus = len(gpus)
            gpus = [gpus[(i + worker_id) % num_gpus] for i in range(num_gpus)]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpus)
    # os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(SHARED_RESULT_PATH / "torchinductor_cache")
    # os.environ["TRITON_CACHE_DIR"] = str(SHARED_RESULT_PATH / "triton_cache")


# TODO: Fixtures
TOKENIZER_PATH = SHARED_RESULT_PATH / "tokenizer"
TOKENIZER_FILE = TOKENIZER_PATH / "tokenizer.json"
DATASET_CACHE = SHARED_RESULT_PATH / "dataset"
DATASET_PREFIX = DATASET_CACHE / "common_dataset"
DATASET_SAMPLING_CACHE = TEST_RESULTS_PATH / "dataset_sampling_cache"
TEST_VOCAB_SIZE = 8192
# Random lowercase: 80.7% (3.1% each); space: 18.6%; doc end: 0.6%
TEST_CHARACTERS = (string.ascii_lowercase) * 5 + " " * 30 + "\n"
TEST_DATASET_TOKENS = 1000000

MODEL_DATASET_PREFIX = DATASET_CACHE / "model_dataset"
MODEL_TEST_VOCAB_SIZE = 384
