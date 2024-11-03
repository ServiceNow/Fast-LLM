import abc
import argparse
import os
import pathlib
import shlex
import subprocess
import typing

from fast_llm.config import Config, Field, FieldHint, FieldUpdate, check_field, config_class, skip_valid_if_none
from fast_llm.data.config import AbstractDataConfig
from fast_llm.engine.checkpoint.config import (
    CheckpointLoadConfig,
    CheckpointSaveConfig,
    CheckpointStateSaveConfigBase,
    DistributedCheckpointFormat,
)
from fast_llm.engine.config_utils.run import ExperimentConfig
from fast_llm.engine.multi_stage.config import PretrainedFastLLMModelConfig
from fast_llm.engine.optimizer.config import OptimizerConfig
from fast_llm.engine.schedule.config import BatchConfig, ScheduleConfig
from fast_llm.profile import ProfilingConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.training.trainer import Trainer


@config_class()
class IntervalConfig(Config):
    # Intervals are a common pattern, so we standardize them with this base class.
    interval: int | None = Field(
        default=None,
        desc="The number of training iterations between each interval. Setting to None will disable.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )
    offset: int = Field(
        default=0,
        desc="Offset for the first interval.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )

    def _validate(self):
        if self.interval:
            self.offset %= self.interval
        super()._validate()

    def enabled(self, iteration: int | None = None):
        return self.interval and (iteration is None or (iteration - self.offset) % self.interval == 0)

    def is_sub_interval(self, other: "IntervalConfig"):
        if not self.enabled():
            return True
        elif not other.enabled():
            return False
        return self.interval % other.interval == 0 and (other.offset % other.interval) == (
            self.offset % other.interval
        )

    def assert_sub_interval(self, other: "IntervalConfig"):
        assert self.is_sub_interval(other), f"{self} is not a sub-interval of {other}"

    def get_count(self, iteration):
        # Number of times this interval was enabled after a given iteration.
        return (iteration - self.offset) // self.interval + 1 if self.enabled() else 0


def _validate_script(value):
    if isinstance(value, str):
        value = shlex.split(value)
    Assert.geq(len(value), 1)
    return value


@config_class()
class CallbackConfig(Config):
    script: list[str] | None = Field(
        default=None,
        desc="Shell script to run.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(_validate_script),
    )
    environment: dict[str, str] = Field(
        default_factory=dict,
        desc="Environment variables to add to the script.",
        hint=FieldHint.feature,
    )

    def run(self):
        if self.script is not None:
            environment = os.environ.copy()
            environment.update(self.environment)
            subprocess.Popen(self.script, env=environment)


@config_class()
class WandbAlertConfig(IntervalConfig):
    interval = FieldUpdate(
        desc="The number of training iterations between each Wandb status post (alert)."
        " Setting to None will disable iteration-based wandb alerts."
        " Must be a sub-interval of the logging interval."
    )
    offset = FieldUpdate(
        desc="Offset for the first Wandb status post (alert)." " Must be compatible with the logging offset.",
    )
    status_updates: bool | None = Field(
        default=None,
        desc="Post wandb status updates on status changes (run begin/end). "
        "The update may be posted by email and/or slack depending on the Wandb account configuration.",
        hint=FieldHint.feature,
    )

    def _validate(self):
        if self.status_updates is None:
            self.post_alerts = self.enabled()
        super()._validate()


@config_class()
class MetricsLogsConfig(IntervalConfig):
    interval = FieldUpdate(
        default=100,
        desc="The number of training iterations between each metric logs."
        " Setting to None will disable metric logging.",
    )
    offset = FieldUpdate(desc="Offset for the first metric logs.")


@config_class()
class WandbConfig(Config):
    alert: WandbAlertConfig = Field(
        default_factory=WandbAlertConfig,
        desc="Configuration for Wandb alerts."
        " The alerts may be posted by email and/or slack depending on the Wandb account configuration.",
        hint=FieldHint.core,
    )
    group_name: str = Field(default="default", desc="A group name for Wandb", hint=FieldHint.feature)
    project_name: str = Field(default="fast_llm", desc="A project name for Wandb", hint=FieldHint.feature)
    entity_name: str | None = Field(default=None, desc="An entity (user) name for Wandb", hint=FieldHint.feature)


@config_class()
class ValidationConfig(IntervalConfig):
    interval = FieldUpdate(
        desc="The number of training iterations between each validation phase."
        " Setting to None will disable validation."
    )
    offset = FieldUpdate(desc="Offset for the first validation phase.")
    iterations: int | None = Field(
        default=None,
        desc="Number of iterations for each validation phase. Setting to None will disable.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )

    def get_iteration_count(self, training_iterations: int, extra_validations: int = 0):
        # Number of completed validation iterations
        return (self.get_count(training_iterations) + extra_validations) * self.iterations if self.enabled() else 0


@config_class()
class TrainingCheckpointBaseConfig(IntervalConfig):
    _abstract = True
    save_name: typing.ClassVar[str] = "save"
    callback: CallbackConfig = Field(
        default_factory=CallbackConfig,
        desc="Callback (shell script).",
        hint=FieldHint.core,
    )
    keep: int | None = Field(
        default=None,
        desc="The maximum number of saves to keep. When exceeding this value, checkpoints are deleted starting from the older ones.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )
    keep_every: int | None = Field(
        default=None,
        desc="Keep every nth saves, i.e. Exclude it from the checkpoint count and deletion in `keep`.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )

    @abc.abstractmethod
    def get_save_directory(self, experiment_directory: pathlib.Path) -> pathlib.Path:
        pass

    def get_save_config(self, path: pathlib.Path):
        raise NotImplementedError()

    def to_delete(self, iterations: list[int]):
        if not self.keep:
            return []
        # Ignore checkpoints that aren't supposed to be there.
        iterations = [iteration for iteration in iterations if self.enabled(iteration)]
        # Ignore excluded checkpoints.
        if self.keep_every:
            iterations = [iteration for iteration in iterations if self.get_count(iteration) % self.keep_every != 0]
        # Exclude the last `keep`.
        return iterations[: -self.keep]


@config_class()
class TrainingCheckpointConfig(TrainingCheckpointBaseConfig):
    _abstract = False
    save_name: typing.ClassVar[str] = "checkpoint"
    # TODO v0.2: Rename to `checkpoint` so we don't need this extra variable?
    interval = FieldUpdate(
        desc="The number of training iterations between each checkpoint." " Setting to None will disable checkpoints."
    )
    offset = FieldUpdate(desc="Offset for the first checkpoint.")
    callback: CallbackConfig = FieldUpdate(desc="Callback (shell script) to run after checkpoint.")
    keep: int | None = FieldUpdate(default=5)

    def get_save_directory(self, experiment_directory: pathlib.Path) -> pathlib.Path:
        # TODO v0.2: Remove backward compatibility.
        old_path = experiment_directory / "checkpoints"
        new_path = experiment_directory / "checkpoint"
        return old_path if old_path.is_dir() and not new_path.is_dir() else new_path

    def get_save_config(self, path: pathlib.Path):
        return CheckpointSaveConfig(
            path=path,
            format=DistributedCheckpointFormat,
            model_weights=True,
            optimizer_state=True,
        )

    def get_load_config(self, path: pathlib.Path):
        return CheckpointLoadConfig(
            path=path,
            format=DistributedCheckpointFormat,
            model_weights=True,
            optimizer_state=True,
        )


@config_class()
class TrainingExportConfig(TrainingCheckpointBaseConfig, CheckpointStateSaveConfigBase):
    _abstract = False
    save_name: typing.ClassVar[str] = "export"
    interval = FieldUpdate(
        desc="The number of training iterations between each export." " Setting to None will disable exports."
    )
    offset = FieldUpdate(desc="Offset for the first export.")
    callback: CallbackConfig = FieldUpdate(desc="Callback (shell script) to run after export.")

    @abc.abstractmethod
    def get_save_directory(self, experiment_directory: pathlib.Path) -> pathlib.Path:
        return experiment_directory / "export" / self.format.name

    def get_save_config(self, path: pathlib.Path):
        return CheckpointSaveConfig.from_dict(self, {"path": path}, strict=False)


@config_class()
class ShutdownConfig(IntervalConfig):
    interval = FieldUpdate(
        desc="The number of training iterations between each automated shutdown."
        " Setting to None will disable automated shutdowns."
        " Must be a sub-interval of the checkpoint interval."
    )
    offset = FieldUpdate(
        desc="Offset for the first automated shutdown." " Must be compatible with the checkpoint offset."
    )


@config_class()
class TrainingConfig(Config):
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        desc="Configuration for the validation phase",
        hint=FieldHint.core,
    )
    logs: MetricsLogsConfig = Field(
        default_factory=MetricsLogsConfig, desc="Configuration for metric logging.", hint=FieldHint.core
    )
    checkpoint: TrainingCheckpointConfig = Field(
        default_factory=MetricsLogsConfig, desc="Configuration for checkpoints.", hint=FieldHint.core
    )
    export: TrainingExportConfig = Field(
        default_factory=MetricsLogsConfig, desc="Configuration for exports.", hint=FieldHint.core
    )
    shutdown: ShutdownConfig = Field(
        default_factory=ShutdownConfig, desc="Configuration for automated shutdown.", hint=FieldHint.core
    )
    wandb: WandbConfig = Field(default_factory=WandbConfig, desc="Configuration for Wandb.", hint=FieldHint.core)
    train_iters: int = Field(
        default=0, desc="Total number of training iterations.", hint=FieldHint.core, valid=check_field(Assert.geq, 0)
    )
    test_iters: int = Field(
        default=0,
        desc="Number of iterations for the test phase at the end of training. Setting to 0 will disable the test phase.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    num_workers: int = Field(
        default=2,
        desc="Number of data loading processes for each data iterator.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 0),
    )
    prefetch_factor: int | None = Field(
        default=None,
        desc="Prefetch factor for the data loaders, i.e., number of micro-batches that each worker may prepare in advance.",
        hint=FieldHint.performance,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )

    def _validate(self):
        super()._validate()
        self.shutdown.assert_sub_interval(self.checkpoint)
        self.wandb.alert.assert_sub_interval(self.logs)


@config_class()
class TrainerConfig(PretrainedFastLLMModelConfig, ExperimentConfig):
    _abstract = True
    # TODO: Generalize data, schedule, logging, etc.
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        desc="Configuration for the training phases and global properties.",
        hint=FieldHint.core,
    )
    batch: BatchConfig = Field(
        default_factory=BatchConfig,
        desc="Configuration for the training, validation and test batches.",
        hint=FieldHint.core,
    )
    schedule: ScheduleConfig = Field(
        default_factory=ScheduleConfig, desc="Configuration for the scheduling of each iteration.", hint=FieldHint.core
    )
    data: AbstractDataConfig = Field(
        default_factory=AbstractDataConfig,
        desc="Configuration for the dataset and model-independent preprocessing.",
        hint=FieldHint.core,
    )
    profiling: ProfilingConfig = Field(
        default_factory=ProfilingConfig,
        desc="Configuration for the optional profiling of GPU and CPU CUDA operations.",
        hint=FieldHint.logging,
    )
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        desc="Configuration for the training optimizer and learning rate schedule.",
        hint=FieldHint.core,
    )

    def _validate(self):
        self.training.export.setup(self.model)
        super()._validate()
        if self.run.experiment_dir is None:
            assert not self.training.checkpoint.enabled()

    @classmethod
    def get_trainer_class(cls) -> type["Trainer"]:
        raise NotImplementedError

    def _setup(self):
        super()._setup()
        self.batch.setup(self.distributed)

    def _get_runnable(self, parsed: argparse.Namespace) -> typing.Callable[[], None]:
        from fast_llm.engine.distributed.distributed import Distributed

        distributed = Distributed(self.distributed)
        run = self.get_run(distributed)
        trainer = self.get_trainer_class()(config=self)

        def runnable():
            with run:
                trainer.setup(distributed, run)
                trainer.run()

        return runnable
