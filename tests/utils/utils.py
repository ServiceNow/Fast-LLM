import pathlib

import pytest
import torch

from fast_llm.engine.base_model.base_model import BaseModel, Layer
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, StageConfig
from fast_llm.engine.multi_stage.stage import Stage

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


TEST_RESULTS_PATH = pathlib.Path("/tmp/fast_llm_tests")


@pytest.fixture(scope="session")
def result_path():
    return TEST_RESULTS_PATH


def get_base_model(config: FastLLMModelConfig):
    # Create a base model (and distributed).
    # Using a full model config so we have the model type and distributed config in the same argument.
    distributed = Distributed(config.distributed)
    tensor_space = TensorSpace(config.distributed)
    config.base_model.setup_tensor_space(tensor_space)
    tensor_space.setup(distributed)
    base_model = config.get_model_class().base_model_class(config.base_model, config.distributed)
    base_model.setup(distributed)
    return base_model, distributed


def get_stage(base_model: BaseModel | list[Layer], distributed: Distributed):
    # Create a fast-llm stage which allocates and initializes meta tensors correctly.
    stage = Stage(
        config=StageConfig(),
        base_model=base_model,
        distributed_config=distributed.config,
        begin=0,
        end=1,
        index=0,
    )
    stage.setup(distributed=distributed)
    stage.initialize_weights()
    stage.restore_parameters()
    stage.reset_gradients()
    return stage
