import logging
import os
import pathlib
import typing
import warnings

import yaml

from fast_llm.config import Config, Field, FieldHint, FieldVerboseLevel, config_class
from fast_llm.engine.config_utils.logging import TensorLogs, TensorLogsConfig, configure_logging
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.utils import log

if typing.TYPE_CHECKING:
    from fast_llm.engine.distributed.distributed import Distributed

logger = logging.getLogger(__name__)


@config_class()
class RunConfig(Config):
    tensor_logs: TensorLogsConfig = Field(desc="Configuration for debug tensor logs.", hint=FieldHint.logging)
    # TODO v0.3: Adjust (now only affects logging to file).
    structured_logs: bool = Field(
        default=True, desc="Configure logging to the Fast-LLM format.", hint=FieldHint.logging
    )
    experiment_dir: pathlib.Path | None = Field(
        default=None, desc="Directory where every checkpoint, artifact, etc., will be saved.", hint=FieldHint.core
    )
    enable_all_loggers: bool = Field(
        default=False,
        desc="Enable all existing loggers, including those external to Fast-LLM, by setting their level to `info`.",
        hint=FieldHint.logging,
    )
    log_timestamps: bool = Field(
        default=True, desc="Add a timestamp to every Fast-LLM (structured) log.", hint=FieldHint.logging
    )
    # TODO: Only needed for wandb?
    experiment_name: str | None = Field(
        default=None,
        desc="A custom name for the experiment. Default: the experiment directory name or 'default'",
        hint=FieldHint.feature,
    )
    # Enable torch compile.
    torch_dynamo_enable: bool = Field(
        default=True,
        desc="Set to False to disable torch compile entirely. Not recommended unless there is a good reason to do so.",
        hint=FieldHint.expert,
    )
    enable_triton_kernels: bool = Field(
        default=True,
        desc="Global switch to allow disabling triton kernels. This parameter may be ignored when no alternative is available.",
        hint=FieldHint.expert,
    )
    # Use triton implementation for all linear kernels (slower, for testing only).
    triton_linear_kernels: bool = Field(
        default=False,
        desc="Global switch to use triton kernels for linear layers. These may be slightly slower than the defaults.",
        hint=FieldHint.performance,
    )

    def _validate(self):
        if self.experiment_dir is None:
            assert not self.tensor_logs.save
        super()._validate()


@config_class()
class ExperimentConfig(RunnableConfig):
    run: RunConfig = Field(desc="Global properties for the experiment.", hint=FieldHint.core)

    def _show(
        self,
        verbose: int = FieldVerboseLevel.core,
        *,
        log_fn=logger.info,
        title: str | None = None,
        width: int = 60,
        fill_char: str = "-",
    ) -> None:
        if is_main_rank():
            return super()._show(verbose, log_fn=log_fn, title=title, width=width, fill_char=fill_char)

    def configure_logging(
        self, directory: pathlib.Path | str | None = None, distributed: DistributedConfig | None = None
    ) -> None:
        configure_logging(
            log_timestamps=self.run.log_timestamps,
            enable_all_loggers=self.run.enable_all_loggers,
            rank=DistributedConfig.default_rank if distributed is None else distributed.rank,
            world_size=DistributedConfig.default_world_size if distributed is None else distributed.world_size,
            directory=directory,
        )

    def get_run(self, distributed: "Distributed") -> "Run":
        from fast_llm.functional.config import TritonConfig

        TritonConfig.TRITON_ENABLED = self.run.enable_triton_kernels
        TritonConfig.TRITON_LINEAR = self.run.triton_linear_kernels
        run = Run(config=self, distributed=distributed)
        self._set_external_variables()
        return run

    def _set_external_variables(self) -> None:
        import torch._dynamo

        # TODO: Find an alternative to get reliable tensor-parallel overlap.
        if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", ""):
            warnings.warn("Setting CUDA_DEVICE_MAX_CONNECTIONS breaks things.")
        if "PYTHONHASHSEED" not in os.environ:
            warnings.warn("PYTHONHASHSEED should be set and to the same value for all workers.")

        torch._dynamo.config.disable = not self.run.torch_dynamo_enable  # noqa


_MAIN_RANK = 0


class Run:
    """
    This (singleton) deals with much of the boilerplate training code.
    TODO: Improve checkpointing (speed and robustness)
    TODO: Add a status file to the directory to (ex. RUNNING, INTERRUPTED, FAILED, DONE)
    """

    _experiment_dir: pathlib.Path | None

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        distributed: "Distributed",
    ):
        self._config = config.run
        self._distributed = distributed

        # TODO: Main rank should contain the last pipeline stage so it calculates loss
        self._is_main_rank = self._distributed.config.rank == _MAIN_RANK
        self._is_model_parallel_main_rank = self._distributed.config.data_rank == 0
        self._is_pipeline_parallel_main_rank = (
            self._distributed.config.data_rank == 0 and self._distributed.config.tensor_rank == 0
        )
        config_dict = config.to_dict()
        config_dict_verbose = config.to_dict(verbose=FieldVerboseLevel.performance)

        if self._config.experiment_dir is not None:
            self._experiment_directory = self._config.experiment_dir.resolve()
            if self._is_main_rank:
                (self._experiment_directory / "runs").mkdir(exist_ok=True, parents=True)
                run = len(list((self._experiment_directory / "runs").iterdir()))
                (self._experiment_directory / "runs" / str(run)).mkdir()
                yaml.safe_dump(config_dict, (self._experiment_directory / "config.yaml").open("w"))
                # Dumping a verbose version of the config
                yaml.safe_dump(config_dict_verbose, (self._experiment_directory / "config_verbose.yaml").open("w"))
            else:
                run = 0
            # Make sure all the workers agree on the run. This also acts as a barrier.
            self.index = self.broadcast_int(run)
            run_dir = self._experiment_directory / "runs" / str(self.index)
            self._artifact_dir = run_dir / "artifacts" / str(self._distributed.config.rank)
            log_dir = run_dir / "logs"
        else:
            self._experiment_directory, self._artifact_dir, log_dir = None, None, None
            self.index = None

        # Finalize logging configuration.
        config.configure_logging(log_dir)

        self._experiment_name = self._config.experiment_name or (
            "default" if self._experiment_directory is None else self._experiment_directory.name
        )

    @property
    def is_main_rank(self) -> bool:
        return self._is_main_rank

    @property
    def experiment_directory(self) -> pathlib.Path | None:
        return self._experiment_directory

    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    @property
    def _is_running(self) -> bool:
        return _run == self

    def save_logged_tensors(self, iteration: int | str) -> None:
        import torch

        assert self._is_running
        tensor_stats = TensorLogs.get()
        if tensor_stats:
            torch.save(tensor_stats, self.open_artifact(f"tensor_logs_{iteration}.pt", mode="wb"))
            TensorLogs.reset(self._config.tensor_logs)

    def broadcast_int(self, value: int) -> int:
        import torch

        from fast_llm.core.distributed import broadcast_scalar

        return broadcast_scalar(value, dtype=torch.int64, src=_MAIN_RANK, group=self._distributed.world_group)

    def open_artifact(self, name: str, mode: str | None = "w", verbose=True) -> pathlib.Path | typing.IO:
        assert self._is_running
        if self._artifact_dir is None:
            # TODO: Open a file that writes to logger.info when possible?
            path = pathlib.Path(os.devnull)
        else:
            self._artifact_dir.mkdir(parents=True, exist_ok=True)
            path = self._artifact_dir / name
        if verbose:
            logger.info(f"Saving artifact to {path}")
        return path if mode is None else path.open(mode)

    def __enter__(self):
        assert not self._is_running
        global _run
        _run = self
        TensorLogs.reset(self._config.tensor_logs)

    def __exit__(self, exc_type, exc_val: OSError, exc_tb):
        assert self._is_running
        global _run
        self.save_logged_tensors("none")
        _run = None


_run: Run | None = None


def get_run() -> Run:
    assert _run is not None
    return _run


def is_main_rank() -> bool:
    return DistributedConfig.default_rank == _MAIN_RANK


def log_main_rank[
    T
](*message, log_fn: type[BaseException] | typing.Callable[[str], T] = logger.info, join: str = ", ") -> T:
    if is_main_rank():
        return log(*message, log_fn=log_fn, join=join)


def is_model_parallel_main_rank() -> bool:
    return is_main_rank() if _run is None else _run._is_model_parallel_main_rank  # Noqa


def log_model_parallel_main_rank[
    T
](*message, log_fn: type[BaseException] | typing.Callable[[str], T] = logger.info) -> T:
    if is_model_parallel_main_rank():
        return log(*message, log_fn=log_fn)


def is_pipeline_parallel_main_rank() -> bool:
    return is_main_rank() if _run is None else _run._is_pipeline_parallel_main_rank  # Noqa


def log_pipeline_parallel_main_rank[
    T
](*message, log_fn: type[BaseException] | typing.Callable[[str], T] = logger.info) -> T:
    if is_pipeline_parallel_main_rank():
        return log(*message, log_fn=log_fn)
