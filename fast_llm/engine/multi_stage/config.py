import enum
import json
import logging
import typing

from fast_llm.config import Config, Field, FieldHint, NoAutoValidate, check_field, config_class, skip_valid_if_none
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.checkpoint.config import (
    CHECKPOINT_VERSION,
    KNOWN_CHECKPOINT_VERSIONS,
    CheckpointFormat,
    CheckpointLoadConfig,
    CheckpointLoadMetadataConfig,
)
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.huggingface.model import HuggingfacePreTrainedModel
    from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel

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
    def get_model_class(cls) -> type["FastLLMModel"]:
        raise NotImplementedError

    @classmethod
    def get_huggingface_model_class(cls) -> type["HuggingfacePreTrainedModel"]:
        raise NotImplementedError

    @classmethod
    def get_base_model_config_cls(cls) -> type[BaseModelConfig]:
        # TODO v0.2: Still needed?
        return cls.get_field("base_model").type

    @classmethod
    def from_pretrained(
        cls,
        pretrained: CheckpointLoadMetadataConfig,
        default: "FastLLMModelConfig" = None,
    ):
        # TODO: Add *updates?
        assert pretrained.path is not None
        metadata = cls.load_pretrained_metadata(pretrained)
        return cls.from_metadata(pretrained, metadata, default)

    @classmethod
    def from_metadata(
        cls,
        pretrained: CheckpointLoadMetadataConfig,
        metadata: dict,
        default: "FastLLMModelConfig" = None,
        updates: dict[str | tuple[str, ...], typing.Any] | None = None,
    ):
        # TODO v0.2: Make checkpoint type mandatory
        # TODO: Standardize to *updates?
        if "checkpoint_type" in metadata:
            # TODO python 3.12: Assert.incl(metadata["checkpoint_type"], CheckpointType)
            CheckpointFormat(metadata["checkpoint_type"])
        version = metadata["checkpoint_version"]
        if version not in KNOWN_CHECKPOINT_VERSIONS:
            raise ValueError(f"Unrecognised checkpoint version: {version}")
        if version == "0":
            return cls._from_metadata_v0(pretrained, metadata, default, updates)

        pretrained_config = cls.from_dict(metadata["fast_llm_config"])
        if not pretrained.load_config.load_architecture:
            assert default is not None
            config = default.to_copy()
            config.base_model.compare_architecture(pretrained_config.base_model, pretrained.compare_log_fn)
        elif pretrained.load_config.load_fast_llm:
            config = pretrained_config
        else:
            with NoAutoValidate():
                config = cls() if default is None else default.to_copy()
            if pretrained.load_config.load_base_model:
                config.base_model = pretrained_config.base_model
            else:
                config.base_model = config.base_model.to_copy(pretrained_config.base_model.get_architecture())
            config.validate()

        if updates:
            config = config.to_copy(updates)
        return config

    @classmethod
    def _from_metadata_v0(
        cls,
        pretrained: CheckpointLoadMetadataConfig,
        metadata: dict,
        default: "FastLLMModelConfig" = None,
        updates: dict[str | tuple[str, ...], typing.Any] | None = None,
    ):
        # TODO v0.2: Remove
        base_model_config_cls = cls.get_base_model_config_cls()
        architecture_config = base_model_config_cls.architecture_cls.from_flat_dict(
            metadata["model_config"].copy(), strict=False
        )

        with NoAutoValidate():
            if default is None:
                assert pretrained.load_config.load_architecture
                config = cls(base_model=base_model_config_cls())
            else:
                config = default.to_copy()

        if pretrained.load_config.load_architecture:
            config.validate()
            architecture_config.compare_architecture(default.base_model, pretrained.compare_log_fn)
        else:
            if pretrained.load_config.load_base_model:
                # Replace the whole config
                config.base_model = base_model_config_cls.from_flat_dict(metadata["model_config"])
            else:
                # Replace the architecture parts of the config.
                config.base_model = config.base_model.to_copy(architecture_config)
            if pretrained.load_config.load_fast_llm:
                config.multi_stage = MultiStageConfig.from_flat_dict(metadata["multi_stage_config"])
                config.distributed = DistributedConfig.from_flat_dict(
                    metadata["distributed_config"],
                )

        config.validate()
        if updates:
            config = config.to_copy(updates)
        return config

    @classmethod
    def load_pretrained_metadata(cls, pretrained: CheckpointLoadMetadataConfig):
        import yaml

        base_model_config_cls = cls.get_base_model_config_cls()
        if pretrained.format == CheckpointFormat.distributed:
            return yaml.safe_load((pretrained.path / "metadata.yaml").open("r"))
        elif pretrained.format == CheckpointFormat.state_dict:
            return json.load((pretrained.path / f"state_dict.safetensors.index.json").open("r"))["metadata"]
        elif pretrained.format == CheckpointFormat.external:
            converter_class = base_model_config_cls.get_converter_class(pretrained.model_type)
            imported_model_config = converter_class.import_config(converter_class.load_config(pretrained.path), True)
            return {
                "fast_llm_config": {"base_model": imported_model_config.to_serialized()},
                "checkpoint_type": CheckpointFormat.external.value,
                "checkpoint_version": CHECKPOINT_VERSION,
            }
        else:
            raise NotImplementedError(pretrained.format)


@config_class()
class PretrainedFastLLMModelConfig(Config):
    # TODO: Generalize data, schedule, logging, etc.
    _abstract = True
    model: FastLLMModelConfig = Field(
        default_factory=FastLLMModelConfig, desc="Configuration for the Fast-LLM model.", hint=FieldHint.core
    )
    pretrained: CheckpointLoadConfig = Field(
        default_factory=CheckpointLoadConfig,
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
        if self.pretrained.path is not None:
            self.model = self.model.from_pretrained(self.pretrained, default=self.model)
        self._setup()
        super()._validate()

    def _setup(self):
        # Setup to run once the model is known, but before field validation
        self._distributed = self.model.distributed
        self._multi_stage = self.model.multi_stage
        self._base_model = self.model.base_model
