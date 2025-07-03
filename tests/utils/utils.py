import logging
import pathlib
import sys
import traceback
import typing

import _pytest.capture
import pytest
import torch

from fast_llm.engine.base_model.base_model import BaseModel, Layer
from fast_llm.engine.config_utils.logging import configure_logging
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, StageConfig
from fast_llm.engine.multi_stage.stage import Stage
from fast_llm.utils import header

logger = logging.getLogger(__name__)

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


class DistributedSubtestContext:
    def __init__(self, path: pathlib.Path, rank: int) -> None:
        self._path = path
        self._rank = rank
        self._capture_manager = _pytest.capture.CaptureManager("fd")
        self.success = False

    def __enter__(self) -> typing.Self:
        self._capture_manager.start_global_capturing()
        # Logging is set to log to the old stdout, so we need to reconfigure.
        configure_logging()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._capture_manager.suspend_global_capture()
            out, err = self._capture_manager.read_global_capture()
            self._path.mkdir(parents=True, exist_ok=True)
            self._path.joinpath(f"pytest_stdout_{self._rank}").write_text(out)
            self._path.joinpath(f"pytest_stderr_{self._rank}").write_text(err)
            if exc_type is None:
                self.success = True
            else:
                self._path.joinpath(f"pytest_traceback_{self._rank}").write_text(traceback.format_exc())
            return True
        finally:
            self._capture_manager.stop_global_capturing()
            configure_logging()


def report_subtest(path: pathlib.Path, world_size: int):
    try:
        success = bool(int(path.joinpath("pytest_success").read_text()))
    except OSError:
        success = False
    if not success:
        for rank in range(world_size):
            for fd, file_ in (("stdout", sys.stdout), ("stderr", sys.stdout), ("traceback", sys.stderr)):
                print(header(f"{fd} rank {rank}", 80), file=file_)
                file_path = path / f"pytest_{fd}_{rank}"
                try:
                    print(file_path.read_text(), file=file_)
                except OSError:
                    print(f"<<< not found {file_path}>>>", file=file_)
        raise RuntimeError(f"test {path.name} failed")
