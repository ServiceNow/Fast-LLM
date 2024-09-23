import logging
import os
import pathlib
import shlex
import sys
import typing
import warnings

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, skip_valid_if_none
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
class ExperimentConfig(Config):
    run: RunConfig = Field(
        default_factory=RunConfig, desc="Global properties for the experiment.", hint=FieldHint.core
    )

    def show_main_rank(self, distributed_config: DistributedConfig, main_rank: int = 0, log_fn=logger.info):
        if distributed_config.rank == main_rank:
            log_fn(f"Command run:\n{shlex.join(sys.argv)}")
            self.show(log_fn=log_fn)

    def get_run(self, distributed: "Distributed", main_rank: int = 0):
        from fast_llm.engine.run.run import Run
        from fast_llm.functional.config import TritonConfig

        TritonConfig.TRITON_ENABLED = self.run.enable_triton_kernels
        TritonConfig.TRITON_LINEAR = self.run.triton_linear_kernels
        run = Run(config=self, distributed=distributed, main_rank=main_rank)
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
