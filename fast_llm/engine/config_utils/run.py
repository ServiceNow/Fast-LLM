import logging
import os
import pathlib
import shutil
import typing
import warnings

import yaml

from fast_llm.config import Config, Field, FieldHint, FieldVerboseLevel, config_class
from fast_llm.engine.config_utils.logging import TensorLogs, TensorLogsConfig, configure_logging
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.utils import Assert, log

if typing.TYPE_CHECKING:
    from fast_llm.engine.distributed.distributed import Distributed

logger = logging.getLogger(__name__)


@config_class()
class RunConfig(Config):
    tensor_logs: TensorLogsConfig = Field(
        default_factory=TensorLogsConfig, desc="Configuration for debug tensor logs.", hint=FieldHint.logging
    )
    # TODO v0.2: Adjust (now only affects logging to file).
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
    run: RunConfig = Field(
        default_factory=RunConfig, desc="Global properties for the experiment.", hint=FieldHint.core
    )

    def _show(
        self,
        verbose: int = FieldVerboseLevel.core,
        *,
        log_fn=logger.info,
        title: str | None = None,
        width: int = 60,
        fill_char: str = "-",
    ):
        if is_main_rank():
            return super()._show(verbose, log_fn=log_fn, title=title, width=width, fill_char=fill_char)

    def configure_logging(self, directory: pathlib.Path | str | None = None):
        configure_logging(
            log_timestamps=self.run.log_timestamps,
            enable_all_loggers=self.run.enable_all_loggers,
            rank=DistributedConfig.default_rank,
            world_size=DistributedConfig.default_world_size,
            directory=directory,
        )

    def get_run(self, distributed: "Distributed"):
        from fast_llm.functional.config import TritonConfig

        TritonConfig.TRITON_ENABLED = self.run.enable_triton_kernels
        TritonConfig.TRITON_LINEAR = self.run.triton_linear_kernels
        run = Run(config=self, distributed=distributed)
        self._set_external_variables()
        return run

    def _set_external_variables(self):
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
    _checkpoint_dir: pathlib.Path | None

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        distributed: "Distributed",
    ):
        self._config = config.run
        self._distributed_config = distributed.config
        Assert.eq(self._distributed_config.world_size, DistributedConfig.default_world_size)
        Assert.eq(self._distributed_config.local_world_size, DistributedConfig.default_local_world_size)
        Assert.eq(self._distributed_config.rank, DistributedConfig.default_rank)
        self._distributed = distributed

        # TODO: Main rank should contain the last pipeline stage so it calculates loss
        self._is_main_rank = self._distributed_config.rank == _MAIN_RANK
        self._is_model_parallel_main_rank = self._distributed_config.data_rank == 0
        self._is_pipeline_parallel_main_rank = (
            self._distributed_config.data_rank == 0 and self._distributed_config.tensor_rank == 0
        )
        config_dict = config.to_serialized()

        if self._config.experiment_dir is not None:
            self._experiment_directory = self._config.experiment_dir.resolve()
            self.dataset_cache_dir = self._experiment_directory / "dataset_cache"
            self._checkpoint_dir = self._experiment_directory / "checkpoints"
            self._export_dir = self._experiment_directory / "export"
            if self._is_main_rank:
                self._checkpoint_dir.mkdir(exist_ok=True, parents=True)
                (self._experiment_directory / "runs").mkdir(exist_ok=True, parents=True)
                run = len(list((self._experiment_directory / "runs").iterdir()))
                (self._experiment_directory / "runs" / str(run)).mkdir()
                yaml.safe_dump(config_dict, (self._experiment_directory / "config.yaml").open("w"))
                self.dataset_cache_dir.mkdir(exist_ok=True)
            else:
                run = 0
            # Make sure all the workers agree on the run. This also acts as a barrier.
            self.index = self._broadcast_int(run)
            run_dir = self._experiment_directory / "runs" / str(self.index)
            self._artifact_dir = run_dir / "artifacts" / str(self._distributed_config.rank)
            log_dir = run_dir / "logs"
        else:
            _experiment_directory, self._checkpoint_dir, self._artifact_dir, log_dir = None, None, None, None
            self.dataset_cache_dir = None
            self.index = None

        if self._config.structured_logs:
            config.configure_logging(log_dir)

        self._experiment_name = self._config.experiment_name or (
            "default" if self._experiment_directory is None else self._experiment_directory.name
        )

    @property
    def is_main_rank(self):
        return self._is_main_rank

    @property
    def experiment_directory(self):
        return self._experiment_directory

    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def _is_running(self):
        return _run == self

    def save_logged_tensors(self, iteration: int | str):
        import torch

        assert self._is_running
        tensor_stats = TensorLogs.get()
        if tensor_stats:
            torch.save(tensor_stats, self.open_artifact(f"tensor_logs_{iteration}.pt", mode="wb"))
            TensorLogs.reset(self._config.tensor_logs)

    def get_save_checkpoint_context(self, iteration: int, export: bool = False, keep: int | None = None):
        return self._SaveCheckpointContext(self, iteration, export, keep)

    def get_load_checkpoint_context(self, iteration: int):
        return self._LoadCheckpointContext(self, iteration)

    def barrier(self, value: int | str = 1):
        from fast_llm.core.distributed import safe_barrier

        safe_barrier(self._distributed.world_group, value)

    def _broadcast_int(self, value: int):
        import torch

        from fast_llm.core.distributed import broadcast_scalar

        return broadcast_scalar(value, dtype=torch.int64, src=_MAIN_RANK, group=self._distributed.world_group)

    class _CheckpointContext:
        def __init__(self, run: "Run", iteration: int):
            self._run = run
            self._iteration = iteration
            assert self._run._checkpoint_dir is not None
            self._directory = self._run._checkpoint_dir / str(self._iteration)

        @property
        def directory(self):
            return self._directory

    class _SaveCheckpointContext(_CheckpointContext):
        def __init__(self, run: "Run", iteration: int, export: bool = False, keep: int | None = None):
            super().__init__(run, iteration)
            self._export = export
            self._keep = keep
            if self._export:
                self._link_directory = self._directory
                self._directory = self._run._export_dir / str(self._iteration)

        def __enter__(self):
            assert self._run._is_running
            if self._run._is_main_rank:
                logger.info(f"Saving checkpoint at iteration {self._iteration}")
                self._directory.mkdir(parents=True)
                if self._export:
                    (self._run._checkpoint_dir / str(self._iteration)).symlink_to(self._directory)
            # Barrier to ensure the directory is created correctly (and didn't exist before).
            self._run.barrier(f"save {self._iteration} enter")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not exc_type:
                self._run.barrier(f"save {self._iteration} exit")
                if self._run._is_main_rank:
                    # Prevent corrupted checkpoint.
                    (self._directory / "ok").open("w")
                    logger.info(f"Checkpoint saved to {self._directory}")
                    self._run._delete_old_checkpoints(self._keep)

    class _LoadCheckpointContext(_CheckpointContext):
        def __enter__(self):
            assert self._run._is_running
            Assert.custom(pathlib.Path.is_file, self._directory / "ok")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not exc_type:
                self._run.barrier(f"load {self._iteration} exit")

    def _delete_old_checkpoints(self, keep: int | None):
        assert self._is_running
        if keep is None:
            return
        checkpoints = sorted(int(path.name) for path in self._checkpoint_dir.iterdir())
        for checkpoint in checkpoints[:-keep]:
            path = self._checkpoint_dir / str(checkpoint)
            logger.info(f"Deleting checkpoint at {path}")
            try:
                shutil.rmtree(path, ignore_errors=True)
            except OSError as e:
                logger.warning(f"Could not remove checkpoint directory: {e.args}")

    def get_last_checkpoint(self):
        assert self._is_running
        if self._checkpoint_dir is None:
            return None
        if self._is_main_rank:
            checkpoints = [int(path.name) for path in self._checkpoint_dir.iterdir()]
            iteration = max(checkpoints) if checkpoints else -1
        else:
            iteration = -1
        iteration = self._broadcast_int(iteration)
        return iteration if iteration >= 0 else None

    def open_artifact(self, name: str, mode: str | None = "w", verbose=True):
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


def is_main_rank():
    return DistributedConfig.default_rank == _MAIN_RANK


def log_main_rank(*message, log_fn: typing.Union[BaseException, typing.Callable] = logger.info, join: str = ", "):
    if is_main_rank():
        log(*message, log_fn=log_fn, join=join)


def is_model_parallel_main_rank():
    return is_main_rank() if _run is None else _run._is_model_parallel_main_rank  # Noqa


def log_model_parallel_main_rank(*message, log_fn=logger.info):
    if is_model_parallel_main_rank():
        return log(*message, log_fn=log_fn)


def is_pipeline_parallel_main_rank():
    return is_main_rank() if _run is None else _run._is_pipeline_parallel_main_rank  # Noqa


def log_pipeline_parallel_main_rank(*message, log_fn=logger.info):
    if is_pipeline_parallel_main_rank():
        return log(*message, log_fn=log_fn)
