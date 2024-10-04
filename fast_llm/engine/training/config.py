import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.data.config import DataConfig
from fast_llm.engine.config_utils.run import ExperimentConfig
from fast_llm.engine.multi_stage.config import PretrainedFastLLMModelConfig
from fast_llm.engine.optimizer.config import OptimizerConfig
from fast_llm.engine.schedule.config import BatchConfig, ScheduleConfig
from fast_llm.profile import ProfilingConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.training.trainer import Trainer


@config_class()
class TrainingConfig(Config):
    train_iters: int = Field(
        default=0, desc="Total number of training iterations.", hint=FieldHint.core, valid=check_field(Assert.geq, 0)
    )
    validation_iters: int = Field(
        default=0,
        desc="Number of iterations for each validation phase. Setting to 0 will disable the validation phase.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    validation_interval: int = Field(
        default=1000,
        desc="Number of training steps between each validation phase.",
        hint=FieldHint.feature,
        valid=check_field(Assert.gt, 0),
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
    export_callback_script: str = Field(default="", desc="Shell script to run after export.", hint=FieldHint.feature)
    export_callback_env: str = Field(
        default="",
        desc="Environment variables to add to the export script, encoded in json format.",
        hint=FieldHint.feature,
    )


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
    data: DataConfig = Field(
        default_factory=DataConfig,
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

    @classmethod
    def get_trainer_class(cls) -> type["Trainer"]:
        raise NotImplementedError

    def _setup(self):
        super()._setup()
        self.batch.setup(self.distributed)
