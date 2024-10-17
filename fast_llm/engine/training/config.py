import argparse
import os
import shlex
import subprocess
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.data.config import AbstractDataConfig
from fast_llm.engine.config_utils.run import ExperimentConfig
from fast_llm.engine.multi_stage.config import PretrainedFastLLMModelConfig
from fast_llm.engine.optimizer.config import OptimizerConfig
from fast_llm.engine.schedule.config import BatchConfig, ScheduleConfig
from fast_llm.profile import ProfilingConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.training.trainer import Trainer


def get_interval_config_class(desc: str, offset_desc: str | None = None):
    # Intervals are a common pattern, so we standardize them with this helper.
    @config_class()
    class IntervalConfig(Config):
        interval: int | None = Field(
            default=None,
            desc=f"The number of training iterations between each {desc}. Setting to None will disable.",
            hint=FieldHint.feature,
            valid=skip_valid_if_none(check_field(Assert.gt, 0)),
        )
        offset: int = Field(
            default=0,
            desc=f"Offset for the first {offset_desc or desc}.",
            hint=FieldHint.feature,
            valid=check_field(Assert.geq, 0),
        )

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

    return IntervalConfig


@config_class()
class WandbAlertConfig(
    get_interval_config_class(
        "Wandb status post (alert). Must be a multiple of the logging interval",
        "Wandb status post (alert). Must be compatible with the logging offset",
    )
):
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
class MetricsLogsConfig(get_interval_config_class("metric logs")):
    pass


@config_class()
class WandbConfig(Config):
    alert: WandbAlertConfig = Field(
        default_factory=WandbAlertConfig,
        desc="Configuration for Wandb alerts. The alerts may be posted by email and/or slack depending on the Wandb account configuration.",
        hint=FieldHint.core,
    )
    group_name: str = Field(default="default", desc="A group name for Wandb", hint=FieldHint.feature)
    project_name: str = Field(default="fast_llm", desc="A project name for Wandb", hint=FieldHint.feature)
    entity_name: str | None = Field(default=None, desc="An entity (user) name for Wandb", hint=FieldHint.feature)


@config_class()
class ValidationConfig(get_interval_config_class("validation")):
    iterations: int | None = Field(
        default=None,
        desc="Number of iterations for each validation phase. Setting to None will disable.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )

    def get_completed_iterations(self, training_iterations: int, completed_validations: int = 0):
        # Number of completed validation iterations
        return (
            (training_iterations // self.interval + completed_validations) * self.iterations if self.enabled() else 0
        )


@config_class()
class CheckpointConfig(get_interval_config_class("checkpoint")):
    keep: int | None = Field(
        default=5,
        desc="The maximum number of checkpoints to keep. When exceeding this value, checkpoints are deleted starting from the older ones.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )


def _validate_script(value):
    if isinstance(value, str):
        value = shlex.split(value)
    Assert.geq(len(value), 1)
    return value


@config_class()
class CallbackConfig(Config):
    script: list[str] | None = Field(
        default=None,
        desc="Shell script to run after.",
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
class ExportConfig(get_interval_config_class("export")):
    callback: CallbackConfig = Field(
        default_factory=CallbackConfig,
        desc="Callback (shell script) to run after export.",
        hint=FieldHint.core,
    )


@config_class()
class ShutdownConfig(get_interval_config_class("automated shutdown")):
    pass


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
    checkpoint: CheckpointConfig = Field(
        default_factory=MetricsLogsConfig, desc="Configuration for checkpoints.", hint=FieldHint.core
    )
    export: ExportConfig = Field(
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
        self.export.assert_sub_interval(self.checkpoint)
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
