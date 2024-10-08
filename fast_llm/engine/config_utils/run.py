import logging
import os
import pathlib
import shutil
import typing
import warnings

import yaml

from fast_llm.config import Config, Field, FieldHint, FieldVerboseLevel, check_field, config_class, skip_valid_if_none
from fast_llm.engine.config_utils.logging import TensorLogs, configure_logging, log
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.distributed.distributed import Distributed

logger = logging.getLogger(__name__)


@config_class()
class RunConfig(Config):
    log_interval: int = Field(
        default=100,
        desc="Number of iteration between each progress and metric logging.",
        hint=FieldHint.logging,
        valid=check_field(Assert.gt, 0),
    )
    log_offset: int = Field(
        default=1,
        desc="Determine the first logging iteration, for example to log after the first iteration.",
        hint=FieldHint.logging,
        valid=check_field(Assert.geq, 0),
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
        default=False, desc="Add a timestamp to every Fast-LLM (structured) log.", hint=FieldHint.logging
    )
    checkpoint_interval: int | None = Field(
        default=None,
        desc="The number of training iterations between each checkpoint.",
        doc="Checkpoints are temporary saves of the model kept to enable resuming in case of a shutdown.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )
    checkpoint_offset: int = Field(
        default=0,
        desc="Determine the first checkpoint iteration, if applicable.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    # Drop checkpoints if there are more than this amount.
    # TODO: Set default to 5?
    max_checkpoints: int | None = Field(
        default=None,
        desc="The maximum number of checkpoints to keep. When exceeding this value, checkpoints are deleted starting from the older ones.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )
    # Exclude these checkpoints from the `max_checkpoints`
    # (counted in training steps, must be a multiple of `checkpoint_interval`)
    export_interval: int | None = Field(
        default=None,
        desc="The number of training iterations between each export. Must be a multiple of the checkpoint interval.",
        doc="Export are permanent saves of the model, which may for example be kept for downstream usage such as benchmarking, for future reference, or as additional backup.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )
    stop_interval: int | None = Field(
        default=None,
        desc="Perform automated shutdowns at predefined intervals.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )
    stop_offset: int = Field(
        default=0,
        desc="Determine the iteration for the first automated shutdown, if applicable.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    experiment_name: str | None = Field(
        default=None,
        desc="A custom name for the experiment. Default: the experiment directory name or 'default'",
        hint=FieldHint.feature,
    )
    wandb_group_name: str = Field(default="default", desc="A group name for Wandb", hint=FieldHint.feature)
    wandb_project_name: str = Field(default="fast_llm", desc="A project name for Wandb", hint=FieldHint.feature)
    wandb_entity_name: str | None = Field(default=None, desc="An entity (user) name for Wandb", hint=FieldHint.feature)
    wandb_status_interval: int | None = Field(
        default=None,
        desc="The number of training iterations between each Wandb log. Must be a multiple of the logging interval.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )
    wandb_post_alerts: bool = Field(
        default=None,
        desc="Post wandb status updates on status changes (run begin/end) and optionally every `wandb_status_interval` iterations. "
        "The update may be posted by email and/or slack depending on the Wandb account configuration.",
        hint=FieldHint.feature,
    )
    # Enable torch compile.
    torch_dynamo_enable: bool = Field(
        default=True,
        desc="Set to False to disable torch compile entirely. Not recommended unless there is a good reason to do so.",
        hint=FieldHint.expert,
    )
    save_tensor_logs: bool = Field(
        default=False,
        desc="Save tensor logs to an artifact file.",
        hint=FieldHint.logging,
    )
    show_tensor_logs: bool = Field(
        default=True,
        desc="Post all tensor logs to stdout. May lead to extremely large log",
        hint=FieldHint.logging,
    )
    tensor_logs_show_elements: int = Field(
        default=8,
        desc="Maximum number of tensor values to print for each tensor when posting tensor logs to stdout.",
        hint=FieldHint.logging,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
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
        if self.wandb_post_alerts is None:
            self.wandb_post_alerts = bool(self.wandb_status_interval)
        super()._validate()
        if self.wandb_status_interval:
            assert self.wandb_post_alerts
            assert self.wandb_status_interval % self.log_interval == 0
        if self.experiment_dir is None:
            assert not self.checkpoint_interval
        if not self.checkpoint_interval:
            assert not self.export_interval
        elif self.export_interval:
            assert self.checkpoint_interval and self.export_interval % self.checkpoint_interval == 0


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
        self.is_main_rank = self._distributed_config.rank == _MAIN_RANK
        self.is_model_parallel_main_rank = self._distributed_config.data_rank == 0
        self.is_pipeline_parallel_main_rank = (
            self._distributed_config.data_rank == 0 and self._distributed_config.tensor_rank == 0
        )
        config_dict = config.to_serialized()

        if self._config.experiment_dir is not None:
            experiment_dir = self._config.experiment_dir.resolve()
            self.dataset_cache_dir = experiment_dir / "dataset_cache"
            self._checkpoint_dir = experiment_dir / "checkpoints"
            self._export_dir = experiment_dir / "export"
            if self.is_main_rank:
                self._checkpoint_dir.mkdir(exist_ok=True, parents=True)
                (experiment_dir / "runs").mkdir(exist_ok=True, parents=True)
                run = len(list((experiment_dir / "runs").iterdir()))
                (experiment_dir / "runs" / str(run)).mkdir()
                yaml.safe_dump(config_dict, (experiment_dir / "config.yaml").open("w"))
                self.dataset_cache_dir.mkdir(exist_ok=True)
            else:
                run = 0
            # Make sure all the workers agree on the run. This also acts as a barrier.
            self.index = self._broadcast_int(run)
            run_dir = experiment_dir / "runs" / str(self.index)
            self._artifact_dir = run_dir / "artifacts" / str(self._distributed_config.rank)
            log_dir = run_dir / "logs"
            self._save_tensor_logs = self._config.save_tensor_logs
        else:
            experiment_dir, self._checkpoint_dir, self._artifact_dir, log_dir = None, None, None, None
            self.dataset_cache_dir = None
            self.index = None
            self._save_tensor_logs = False

        if self._config.structured_logs:
            config.configure_logging(log_dir)

        self.use_wandb = self._config.wandb_entity_name is not None and self.is_main_rank
        self.experiment_name = self._config.experiment_name or (
            "default" if experiment_dir is None else experiment_dir.name
        )
        if self.use_wandb:
            import wandb

            # Wandb login from file
            api_key_path = os.environ.get("WANDB_API_KEY_PATH")
            if api_key_path:
                os.environ["WANDB_API_KEY"] = pathlib.Path(api_key_path).open("r").read().strip()
            wandb_path = None if experiment_dir is None else experiment_dir / "wandb_config.yaml"
            if wandb_path is not None and wandb_path.is_file():
                wandb_config = yaml.safe_load(wandb_path.open("r"))
            else:
                wandb_config = {
                    "id": wandb.sdk.lib.runid.generate_id(16),
                    "project": self._config.wandb_project_name,
                    "name": self.experiment_name,
                    "entity": self._config.wandb_entity_name,
                    "group": self._config.wandb_group_name,
                    "save_code": False,
                    "resume": "allow",
                }
                if wandb_path is not None:
                    yaml.safe_dump(wandb_config, wandb_path.open("w"))
            wandb.init(config=config_dict, **wandb_config)

    @property
    def _is_running(self):
        return _run == self

    def save_logged_tensors(self, iteration: int | str):
        import torch

        assert self._is_running
        if self._save_tensor_logs:
            tensor_stats = TensorLogs.get()
            if tensor_stats:
                torch.save(tensor_stats, self.open_artifact(f"tensor_logs_{iteration}.pt", mode="wb"))
            TensorLogs.reset()

    def get_save_checkpoint_context(self, iteration: int, export: bool = False):
        return self._SaveCheckpointContext(self, iteration, export)

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
        def __init__(self, run: "Run", iteration: int, export: bool = False):
            super().__init__(run, iteration)
            self._export = export
            if self._export:
                self._link_directory = self._directory
                self._directory = self._run._export_dir / str(self._iteration)

        def __enter__(self):
            assert self._run._is_running
            if self._run.is_main_rank:
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
                if self._run.is_main_rank:
                    # Prevent corrupted checkpoint.
                    (self._directory / "ok").open("w")
                    logger.info(f"Checkpoint saved to {self._directory}")
                    self._run._delete_old_checkpoints()

    class _LoadCheckpointContext(_CheckpointContext):
        def __enter__(self):
            assert self._run._is_running
            Assert.custom(pathlib.Path.is_file, self._directory / "ok")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not exc_type:
                self._run.barrier(f"load {self._iteration} exit")

    def _delete_old_checkpoints(self):
        assert self._is_running
        if self._config.max_checkpoints is None:
            return
        checkpoints = sorted(int(path.name) for path in self._checkpoint_dir.iterdir())
        for checkpoint in checkpoints[: -self._config.max_checkpoints]:
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
        if self.is_main_rank:
            checkpoints = [int(path.name) for path in self._checkpoint_dir.iterdir()]
            iteration = max(checkpoints) if checkpoints else -1
        else:
            iteration = -1
        iteration = self._broadcast_int(iteration)
        return iteration if iteration >= 0 else None

    def log_wandb_metrics(self, completed_steps: int, metrics: dict[str, dict[str, float | int]]):
        assert self._is_running
        # Note: metrics modified in-place
        if self.use_wandb:
            import wandb

            wandb.log(metrics, step=completed_steps)  # noqa

    def post_wandb_alert(self, title, text, level="INFO", wait=0.001):
        assert self._is_running
        if self.use_wandb and self._config.wandb_post_alerts:
            import wandb

            wandb.alert(
                title=title() if callable(title) else title,
                text=f"[{self._config.wandb_project_name}/{self.experiment_name}, run {self.index}]"
                f" {text() if callable(text) else text}",
                level=level,
                wait_duration=wait,
            )

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
        self.post_wandb_alert(f"Run started!", "", "ERROR")
        if self._save_tensor_logs:
            TensorLogs.reset()
        TensorLogs.verbose = self._config.show_tensor_logs
        TensorLogs.max_logged_elements = self._config.tensor_logs_show_elements

    def __exit__(self, exc_type, exc_val: OSError, exc_tb):
        assert self._is_running
        global _run
        if exc_val:
            self.post_wandb_alert(f"Run crashed!", (lambda: ", ".join(exc_val.args)), "ERROR")
        else:
            self.post_wandb_alert(f"Run ended!", "", "INFO")
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
    return is_main_rank() if _run is None else _run.is_model_parallel_main_rank


def log_model_parallel_main_rank(*message, log_fn=logger.info):
    if is_model_parallel_main_rank():
        return log(*message, log_fn=log_fn)


def is_pipeline_parallel_main_rank():
    return is_main_rank() if _run is None else _run.is_pipeline_parallel_main_rank


def log_pipeline_parallel_main_rank(*message, log_fn=logger.info):
    if is_pipeline_parallel_main_rank():
        return log(*message, log_fn=log_fn)
