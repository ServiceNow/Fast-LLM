import logging.config
import os
import pathlib
import shutil

import torch
import torch._dynamo.config  # noqa
import wandb
import wandb.sdk.lib.runid
import yaml

from fast_llm.core.distributed import broadcast_scalar, safe_barrier
from fast_llm.engine.config_utils.logging import configure_logging
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.run.config import ExperimentConfig
from fast_llm.logging import (
    get_logged_tensor_stats,
    log,
    reset_tensor_stats_logging,
    set_max_logged_elements,
    set_show_tensor_logs,
)
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


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
        main_rank: int = 0,
    ):
        self._config = config.run
        self._distributed_config = distributed.config
        self._distributed = distributed
        # TODO: Main rank should contain the last pipeline stage so it calculates loss
        self.main_rank = main_rank
        self.is_main_rank = self._distributed_config.rank == main_rank
        self.is_model_parallel_main_rank = self._distributed_config.data_rank == 0
        self.is_pipeline_parallel_main_rank = (
            self._distributed_config.data_rank == 0 and self._distributed_config.tensor_rank == 0
        )
        config_dict = config.to_flat_dict()

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
            self.index = broadcast_scalar(
                run, dtype=torch.int64, src=self.main_rank, group=self._distributed.world_group
            )
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
            configure_logging(
                log_timestamps=self._config.log_timestamps,
                enable_all_loggers=self._config.enable_all_loggers,
                rank=self._distributed_config.rank,
                world_size=self._distributed_config.world_size,
                directory=log_dir,
            )

        self.use_wandb = self._config.wandb_entity_name is not None and self.is_main_rank
        self.experiment_name = self._config.experiment_name or (
            "default" if experiment_dir is None else experiment_dir.name
        )
        if self.use_wandb:
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
    def has_checkpoint_dir(self):
        return self._checkpoint_dir is not None

    @property
    def _is_running(self):
        return _run == self

    def save_logged_tensors(self, iteration: int | str):
        assert self._is_running
        if self._save_tensor_logs:
            tensor_stats = get_logged_tensor_stats()
            if tensor_stats:
                torch.save(tensor_stats, self.open_artifact(f"tensor_logs_{iteration}.pt", mode="wb"))
            reset_tensor_stats_logging()

    def get_save_checkpoint_context(self, iteration: int, export: bool = False):
        return self._SaveCheckpointContext(self, iteration, export)

    def get_load_checkpoint_context(self, iteration: int):
        return self._LoadCheckpointContext(self, iteration)

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
            safe_barrier(self._run._distributed.world_group, f"save {self._iteration} enter")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not exc_type:
                safe_barrier(self._run._distributed.world_group, f"save {self._iteration} exit")
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
                safe_barrier(self._run._distributed.world_group, f"load {self._iteration} exit")

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
        iteration = broadcast_scalar(
            iteration, dtype=torch.int64, src=self.main_rank, group=self._distributed.world_group
        )
        return iteration if iteration >= 0 else None

    def log_wandb_metrics(self, completed_steps: int, metrics: dict[str, dict[str, float | int]]):
        assert self._is_running
        # Note: metrics modified in-place
        if self.use_wandb:
            wandb.log(metrics, step=completed_steps)  # noqa

    def post_wandb_alert(self, title, text, level=wandb.AlertLevel.INFO, wait=0.001):
        assert self._is_running
        if self.use_wandb and self._config.wandb_post_alerts:
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
        self.post_wandb_alert(f"Run started!", "", wandb.AlertLevel.ERROR)
        if self._save_tensor_logs:
            reset_tensor_stats_logging()
        set_show_tensor_logs(self._config.show_tensor_logs)
        set_max_logged_elements(self._config.tensor_logs_show_elements)

    def __exit__(self, exc_type, exc_val: OSError, exc_tb):
        assert self._is_running
        global _run
        if exc_val:
            self.post_wandb_alert(f"Run crashed!", (lambda: ", ".join(exc_val.args)), wandb.AlertLevel.ERROR)
        else:
            self.post_wandb_alert(f"Run ended!", "", wandb.AlertLevel.INFO)
        self.save_logged_tensors("none")
        _run = None


_run: Run | None = None


def get_dataset_cache_dir():
    return None if _run is None else _run.dataset_cache_dir


def is_main_rank():
    return True if _run is None else _run.is_main_rank


def log_main_rank(*message, log_fn=logger.info):
    if is_main_rank():
        return log(*message, log_fn=log_fn)


def is_model_parallel_main_rank():
    return True if _run is None else _run.is_model_parallel_main_rank


def log_model_parallel_main_rank(*message, log_fn=logger.info):
    if is_model_parallel_main_rank():
        return log(*message, log_fn=log_fn)


def is_pipeline_parallel_main_rank():
    return True if _run is None else _run.is_pipeline_parallel_main_rank


def log_pipeline_parallel_main_rank(*message, log_fn=logger.info):
    if is_pipeline_parallel_main_rank():
        return log(*message, log_fn=log_fn)


def open_artifact(name: str, mode: str | None = "w", verbose=True):
    assert _run is not None
    return _run.open_artifact(name, mode, verbose)
