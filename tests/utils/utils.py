import contextlib
import logging
import typing

import pytest
import torch

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.base_model.config import set_model_names
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, StageConfig
from fast_llm.engine.multi_stage.stage import Stage
from fast_llm.functional.triton import triton_available
from tests.utils.global_variables import TEST_RESULTS_PATH

logger = logging.getLogger(__name__)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
requires_triton = pytest.mark.skipif(not triton_available, reason="Triton is not available")


@contextlib.contextmanager
def no_tf32():
    # TF32 (PyTorch's CUDA matmul default) has ~1e-3 mantissa precision, which amplifies
    # sub-FP32 input perturbations (e.g. Triton-vs-`torch.rms_norm` differing by ~1e-7) into
    # ~1e-5 output drift through matmuls. Disable for tests comparing block forward to an eager
    # reference; otherwise mixing tile/batch sizes can produce ~1e-4 swings unrelated to the
    # feature under test.
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


@pytest.fixture(scope="session")
def result_path():
    return TEST_RESULTS_PATH


def get_base_model(config: FastLLMModelConfig):
    # Create a base model (and distributed).
    # Using a full model config so we have the model type and distributed config in the same argument.
    base_model = config.get_base_model_config_class().get_base_model(config.base_model, config.distributed)
    base_model.setup(distributed := Distributed(config.distributed))
    return base_model, distributed


def get_stage(
    layers: list[Layer],
    distributed: Distributed,
    tied_parameter_duplicates: typing.Iterable[str] = (),
    tied_parameter_duplicate_buffers: dict[str, torch.nn.Parameter] | None = None,
    set_names: bool = True,
):

    for layer in layers:
        if not layer._is_setup:
            layer.setup(distributed)
    if set_names:
        # Normally called in `BaseModelConfig.get_base_model`, but may be missing here.
        set_model_names(torch.nn.ModuleList(layers))
    # Create a fast-llm stage which allocates and initializes meta tensors correctly.
    stage = Stage(
        config=StageConfig(),
        layers=layers,
        distributed_config=distributed.config,
        index=0,
        tied_parameter_duplicates=tied_parameter_duplicates,
    )
    stage.setup(distributed=distributed, tied_parameter_duplicate_buffers=tied_parameter_duplicate_buffers)
    stage.initialize_weights()
    stage.restore_parameters()
    stage.reset_gradients()
    return stage
