import enum
import json
import logging
import pathlib
import typing

import torch
import yaml

from fast_llm.config import (
    Config,
    ConfigDictFormat,
    Field,
    FieldHint,
    NoAutoValidate,
    check_field,
    config_class,
    skip_valid_if_none,
)
from fast_llm.distributed import DistributedConfig, get_float_dtype
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class StageMode(str, enum.Enum):
    # Allow forward and backward passes and optimizer.
    # TODO: Add mode for forward and backward but not optimizer?
    training = "training"
    # Allow forward pass but not backward.
    inference = "inference"
    # Load the model but can't run it.
    weights = "weights"
    # Don't load the model.
    off_device = "off_device"

    @property
    def support_forward(self):
        return self in (StageMode.training, StageMode.inference)

    @property
    def support_backward(self):
        return self == StageMode.training

    @property
    def support_training(self):
        return self == StageMode.training

    @property
    def on_device(self):
        return self != StageMode.off_device


@config_class()
class StageConfig(Config):
    full_precision_gradients: bool = Field(
        default=True,
        desc="Reduce and accumulate gradients in fp32 to improve numerical stability.",
        hint=FieldHint.optional,
    )
    debug_layer_outputs: int = Field(
        default=0,
        desc="Log the output of each layer.",
        hint=FieldHint.logging,
        valid=check_field(Assert.geq, 0),
    )
    debug_layer_gradients: int = Field(
        default=0,
        desc="Log the (input) gradients of each layer.",
        hint=FieldHint.logging,
        valid=check_field(Assert.geq, 0),
    )
    debug_param_init: int = Field(
        default=0,
        desc="Log the parameters after initialization.",
        hint=FieldHint.logging,
        valid=check_field(Assert.geq, 0),
    )
    debug_param_gradients: int = Field(
        default=0,
        desc="Log the gradient shard after reduction.",
        hint=FieldHint.logging,
        valid=check_field(Assert.geq, 0),
    )
    debug_all_param_gradients: int = Field(
        default=0,
        desc="Log each parameter gradient after reduction.",
        hint=FieldHint.logging,
        valid=check_field(Assert.geq, 0),
    )
    debug_param_update: int = Field(
        default=0,
        desc="Log the parameters after update.",
        hint=FieldHint.logging,
        valid=check_field(Assert.geq, 0),
    )
    debug_global_tensors: bool = Field(
        default=True,
        desc="Reconstruct global tensors for debug logs (slow, uses lots of memory, does not concat sequential micro-batches).",
        hint=FieldHint.logging,
    )
    debug_activation_memory: bool = Field(
        default=False,
        desc="Log memory usage after each layer.",
        hint=FieldHint.logging,
    )
    # TODO: Option to check each n steps (by default?)
    debug_tensor_parallel: bool = Field(
        default=False,
        desc="Check for tensor-parallel desyncs and log an error if a desync is found. High overhead",
        hint=FieldHint.logging,
    )
    compile_all: bool = Field(
        default=False,
        desc="Compile the whole model using torch.compile.",
        hint=FieldHint.expert,
    )


@config_class()
class MultiStageConfig(StageConfig):
    layers_per_stage: float = Field(
        default=1.0,
        desc="Number of layers to include in each Fast LLM stage.",
        hint=FieldHint.performance,
        valid=check_field(Assert.gt, 0),
    )
    # TODO: multiple base stages per pp stage?
    stages_per_pipeline_stage: int = Field(
        default=1,
        desc="Number of Fast LLM stages on each pipeline stage.",
        hint=FieldHint.wip,
        valid=check_field(Assert.eq, 1),
    )
    num_weight_buffers: int | None = Field(
        default=None,
        desc="Number of stage buffer for weights. Normally set through the ZeRO stage.",
        hint=FieldHint.expert,
        valid=skip_valid_if_none(check_field(Assert.geq, 1)),
    )
    num_grad_buffers: int | None = Field(
        default=None,
        desc="Number of stage buffer for gradients. Normally set through the ZeRO stage.",
        hint=FieldHint.expert,
        valid=skip_valid_if_none(check_field(Assert.geq, 1)),
    )
    zero_stage: int | None = Field(
        default=None,
        desc="The ZeRO stage.",
        hint=FieldHint.performance,
    )
    # TODO: Implement
    pipeline_delay: float = Field(
        default=0.0,
        desc="Estimated delay (in steps) for data to go around the pipeline, used to improve pipeline-parallel network overlap. Currently unused",
        hint=FieldHint.expert,
        valid=check_field(Assert.geq, 0),
    )
    # TODO: Implement these
    # activation_checkpoint:bool=False
    # Log the output of each stage.
    # debug_stage_outputs: int = 0
    # Log the (input) gradients of each stage.
    # debug_stage_gradients: int = 0

    def _validate(self):
        super()._validate()
        if self.zero_stage is not None:
            Assert.in_range_incl(self.zero_stage, 1, 3)
            if self.zero_stage >= 2:
                self.num_grad_buffers = 2
            if self.zero_stage >= 3:
                self.num_weight_buffers = 2
        if self.num_grad_buffers is not None:
            Assert.geq(self.num_grad_buffers, 1)
        if self.num_weight_buffers is not None:
            Assert.geq(self.num_weight_buffers, 1)


# TODO: Does this matter? Best value?
SHARD_PAD_TO_MULTIPLE = 32

CHECKPOINT_VERSION = 0


class CheckpointType(str, enum.Enum):
    distributed = "distributed"
    # TODO: Rename
    huggingface = "huggingface"
    # Model state dict, mostly for debug.
    state_dict = "state_dict"


@config_class()
class PretrainedConfig(Config):
    pretrained_checkpoint_path: pathlib.Path | None = Field(
        default=None,
        desc="Path to the checkpoint.",
        hint=FieldHint.core,
    )
    pretrained_checkpoint_type: CheckpointType = Field(
        default=CheckpointType.distributed,
        desc="Format of the checkpoint.",
        hint=FieldHint.core,
    )
    imported_model_type: str | None = Field(
        default=None,
        desc="Model type for external models (ex. Huggingace type).",
        hint=FieldHint.feature,
    )
    use_pretrained_config: bool = Field(
        default=True,
        desc="Load the architecture config from the pretrained checkpoint.",
        hint=FieldHint.feature,
    )
    ignore_pretrained_config: bool = Field(
        default=False,
        desc="Ignore the pretrained checkpoint architecture config, i.e., disable verification.",
        hint=FieldHint.feature,
    )
    load_full_base_model_config: bool = Field(
        default=False,
        desc="Load the non-architecture model config from the checkpoint.",
        hint=FieldHint.feature,
    )
    load_full_fast_llm_config: bool = Field(
        default=False,
        desc="Load the distributed and multi-stage config from the checkpoint.",
        hint=FieldHint.feature,
    )


@config_class()
class PretrainedCheckpointConfig(PretrainedConfig):
    # Load weights from pretrained_checkpoint_path (if applicable),
    # otherwise reinitialize them (i.e. load the config only.)
    load_pretrained_weights: bool = Field(
        default=True, desc="Load model weights from the checkpoint.", hint=FieldHint.feature
    )
    load_pretrained_optimizer: bool = Field(
        default=False, desc="Load the optimizer state from the checkpoint.", hint=FieldHint.feature
    )


@config_class()
class CheckpointConfig(Config):
    # TODO: Merge/match with PretrainedConfig?
    checkpoint_path: pathlib.Path = Field(desc="Path to the checkpoint.", hint=FieldHint.core)
    checkpoint_type: CheckpointType = Field(
        default=CheckpointType.distributed, desc="Format of the checkpoint.", hint=FieldHint.core
    )
    exported_model_type: str | None = Field(
        default=None, desc="Model type for external models (ex. Huggingace type).", hint=FieldHint.feature
    )
    save_optimizer: bool = Field(
        default=True, desc="Save the optimizer state from the checkpoint.", hint=FieldHint.feature
    )
    target_params_per_file: int = Field(
        default=2**32,
        desc="Limit the number of parameters saved in each file.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 2**20),
    )
    dtype: torch.dtype | None = Field(
        default=None,
        desc="Data type to save the checkpoint.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(get_float_dtype),
    )


@config_class()
class FastLLMModelConfig(Config):
    _abstract = True
    base_model: BaseModelConfig = Field(
        default_factory=BaseModelConfig, desc="Configuration for the base model.", hint=FieldHint.core
    )
    multi_stage: MultiStageConfig = Field(
        default_factory=MultiStageConfig,
        desc="Configuration for the stage breakdown of the model.",
        hint=FieldHint.core,
    )
    distributed: DistributedConfig = Field(
        default_factory=DistributedConfig, desc="Distributed configuration.", hint=FieldHint.core
    )

    @classmethod
    def get_base_model_config_cls(cls) -> type[BaseModelConfig]:
        return cls.get_field("base_model").type

    @classmethod
    def from_pretrained(
        cls,
        pretrained: PretrainedConfig,
        default: "FastLLMModelConfig" = None,
    ):
        assert pretrained.pretrained_checkpoint_path is not None
        metadata = cls.load_pretrained_metadata(pretrained)
        return cls.from_metadata(pretrained, metadata, default)

    @classmethod
    def from_metadata(
        cls,
        pretrained: PretrainedConfig,
        metadata: dict,
        default: "FastLLMModelConfig" = None,
        updates: dict[str | tuple[str, ...], typing.Any] | None = None,
    ):
        base_model_config_cls = cls.get_base_model_config_cls()
        # TODO: Use nested format (not backward compatible)
        if "checkpoint_type" in metadata:
            Assert.eq(metadata["checkpoint_type"], CheckpointType.distributed.value)
        if "checkpoint_version" in metadata:
            Assert.eq(metadata["checkpoint_version"], str(CHECKPOINT_VERSION))

        architecture_config = base_model_config_cls.architecture_cls.from_dict(
            metadata["model_config"].copy(), format_=ConfigDictFormat.flat, strict=False
        )

        with NoAutoValidate():
            if default is None:
                assert pretrained.use_pretrained_config
                config = cls(base_model=base_model_config_cls())
            else:
                config = default.from_other(default)

        if pretrained.use_pretrained_config:
            if pretrained.load_full_base_model_config:
                # Replace the whole config
                config.base_model = base_model_config_cls.from_dict(
                    metadata["model_config"], format_=ConfigDictFormat.flat
                )
            else:
                # Replace the architecture parts of the config.
                config.base_model = base_model_config_cls.from_other(
                    config.base_model, architecture_config.to_dict(format_=ConfigDictFormat.tuple)
                )
            if pretrained.load_full_fast_llm_config:
                config.multi_stage = MultiStageConfig.from_dict(
                    metadata["multi_stage_config"], format_=ConfigDictFormat.nested
                )
                config.distributed = DistributedConfig.from_dict(
                    metadata["distributed_config"], format_=ConfigDictFormat.nested
                )
        else:
            # TODO: Redundant with FastLLMModel._check_model_config
            pretrained_architecture_config = architecture_config.to_dict(format_=ConfigDictFormat.flat)
            default_architecture_config = default.base_model.get_architecture().to_dict(format_=ConfigDictFormat.flat)
            invalid_keys = {
                key
                for key, value in default_architecture_config.items()
                if value != pretrained_architecture_config[key]
            }
            if invalid_keys:
                msg = f"Model config is incompatible with pretrained checkpoint: {invalid_keys}"
                if config.training.ignore_pretrained_config:
                    logger.warning(msg)
                else:
                    raise ValueError(msg)

        config.validate()
        if updates:
            config = config.from_other(config, updates)
        return config

    @classmethod
    def load_pretrained_metadata(cls, pretrained):
        base_model_config_cls = cls.get_base_model_config_cls()
        if pretrained.pretrained_checkpoint_type == CheckpointType.distributed:
            return yaml.safe_load((pretrained.pretrained_checkpoint_path / "metadata.yaml").open("r"))
        elif pretrained.pretrained_checkpoint_type == CheckpointType.state_dict:
            return json.load((pretrained.pretrained_checkpoint_path / f"state_dict.safetensors.index.json").open("r"))[
                "metadata"
            ]
        elif pretrained.pretrained_checkpoint_type == CheckpointType.huggingface:
            converter_class = base_model_config_cls.get_converter_class(pretrained.imported_model_type)
            imported_model_config = converter_class.import_config(
                converter_class.load_config(pretrained.pretrained_checkpoint_path), True
            )
            return {"model_config": imported_model_config.to_dict(format_=ConfigDictFormat.flat)}
        else:
            raise NotImplementedError(pretrained.pretrained_checkpoint_type)


@config_class()
class PretrainedFastLLMModelConfig(Config):
    # TODO: Generalize data, schedule, logging, etc.
    _abstract = True
    model: FastLLMModelConfig = Field(
        default_factory=FastLLMModelConfig, desc="Configuration for the Fast-LLM model.", hint=FieldHint.core
    )
    pretrained: PretrainedCheckpointConfig = Field(
        default_factory=PretrainedCheckpointConfig,
        desc="Configuration for loading the configuration and state of a pretrained model.",
        hint=FieldHint.feature,
    )
    # These configs may be overridden with the pretrained config during validation, so we should be careful about accessing them before.
    _base_model: BaseModelConfig = Field(
        init=False,
        desc="Pointer to the base model configuration of the Fast-LLM model.",
        hint=FieldHint.derived,
    )
    _multi_stage: MultiStageConfig = Field(
        init=False,
        desc="Pointer to the stage breakdown configuration of the Fast-LLM model.",
        hint=FieldHint.derived,
    )
    _distributed: DistributedConfig = Field(
        init=False,
        desc="Pointer to the distributed configuration of the Fast-LLM model.",
        hint=FieldHint.derived,
    )

    @property
    def distributed(self):
        return self._distributed

    @property
    def multi_stage(self):
        return self._multi_stage

    @property
    def base_model(self):
        return self._base_model

    def _validate(self):
        assert self.model is not None
        self.pretrained.validate()
        if self.pretrained.pretrained_checkpoint_path is not None:
            self.model = self.model.from_pretrained(self.pretrained, default=self.model)
        self._setup()
        super()._validate()

    def _setup(self):
        # Setup to run once the model is known, but before field validation
        self._distributed = self.model.distributed
        self._multi_stage = self.model.multi_stage
        self._base_model = self.model.base_model
