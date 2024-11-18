import enum
import logging
import typing

import packaging.version

from fast_llm import __version__
from fast_llm.config import (
    Config,
    Field,
    FieldHint,
    NoAutoValidate,
    ValidationError,
    check_field,
    config_class,
    skip_valid_if_none,
)
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.checkpoint.config import (
    CheckpointFormat,
    CheckpointHandler,
    CheckpointLoadConfig,
    CheckpointLoadMetadataConfig,
    CheckpointSaveMetadataConfig,
    DistributedCheckpointFormat,
    FastLLMCheckpointFormat,
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
    checkpoint_formats: typing.ClassVar[tuple[type[CheckpointFormat], ...]] = (
        DistributedCheckpointFormat,
        FastLLMCheckpointFormat,
    )
    model_name: typing.ClassVar[str]
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
    def __fast_llm_serialize__(cls):
        return cls.model_name

    @classmethod
    def get_checkpoint_format(cls, format: typing.Union[type[CheckpointFormat], str]) -> type[CheckpointFormat]:
        if isinstance(format, type) and issubclass(format, CheckpointFormat):
            format_ = cls.get_checkpoint_format(format.name)
            Assert.is_(format, format_)
            return format_
        # TODO v0.2: Remove backward compatibility.
        if format == "state_dict":
            format = "fast_llm"
        for format_ in cls.checkpoint_formats:
            if format_.name == format:
                return format_
        raise ValueError(f"Checkpoint format {format} not supported for model {cls.model_name}")

    @classmethod
    def get_checkpoint_handler_class(
        cls, format: typing.Union[type[CheckpointFormat], str]
    ) -> type[CheckpointHandler]:
        return cls.get_checkpoint_format(format).get_handler_class()

    @classmethod
    def get_model_class(cls) -> type["FastLLMModel"]:
        raise NotImplementedError

    @classmethod
    def get_huggingface_model_class(cls) -> type["HuggingfacePreTrainedModel"]:
        raise NotImplementedError

    @classmethod
    def get_base_model_config_class(cls) -> type[BaseModelConfig]:
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
        metadata = cls.load_metadata(pretrained)
        return cls.from_metadata(pretrained, metadata, default)

    @classmethod
    def from_metadata(
        cls,
        pretrained: CheckpointLoadMetadataConfig,
        metadata: "CheckpointMetadata",
        default: "FastLLMModelConfig" = None,
        updates: dict[str | tuple[str, ...], typing.Any] | None = None,
    ):
        # TODO: Standardize to *updates?
        # TODO v0.2: Update, remove support for older checkpoints.
        if metadata.fast_llm_version.major != 0 or metadata.fast_llm_version.minor not in (0, 1):
            raise ValueError(f"Invalid checkpoint version: {metadata.fast_llm_version}")
        pretrained_config = cls.from_dict(metadata.config)
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
    def load_metadata(cls, config: CheckpointLoadMetadataConfig) -> "CheckpointMetadata":
        with NoAutoValidate():
            metadata = config.format.get_handler_class().load_metadata(config)
        try:
            metadata.validate()
        except ValidationError:
            metadata.to_logs(log_fn=logger.error, title="Loaded metadata")
            raise ValueError(f"Validation failed for checkpoint metadata. See logs above for details.")
        Assert.eq(metadata.model, cls)
        return metadata

    def to_metadata(self, config: CheckpointSaveMetadataConfig, **kwargs):
        return CheckpointMetadata(
            fast_llm_version=__version__,
            model=self.__class__,
            format=config.format,
            config=self,
            **kwargs,
        )


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
        self.pretrained.setup(self.model)
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


@config_class
class CheckpointMetadata(Config):
    # TODO: Make entries more flexible?
    #  I.e.. model / format / usage (ex. training) - specific entries instead of a generic metadata?
    fast_llm_version: packaging.version.Version = Field(
        default=__version__,
        desc="The Fast-LLM version this checkpoint was saved with.",
        hint=FieldHint.core,
    )
    # TODO: Model-specific versioning?
    model: type[FastLLMModelConfig] = Field(
        default=None,
        desc="The name of the model saved in this checkpoint (ex. gpt).",
        hint=FieldHint.core,
    )
    format: type[CheckpointFormat] = Field(
        default=None,
        desc="The format this checkpoint was saved in.",
        hint=FieldHint.core,
    )
    config: FastLLMModelConfig = Field(
        default_factory=FastLLMModelConfig,
        desc="The Fast-LLM model configuration for the saved model.",
        hint=FieldHint.core,
    )
    shards: list[str] = Field(
        default_factory=list,
        desc="The name of the model shards saved in this checkpoint.",
        hint=FieldHint.optional,
    )
    # TODO: Anything not included here.
    metadata: dict = Field(
        default_factory=dict,
        desc="Additional information for this checkpoint.",
        hint=FieldHint.optional,
    )

    def _validate(self):
        if isinstance(self.fast_llm_version, str):
            self.fast_llm_version = packaging.version.Version(self.fast_llm_version)

        self.format = self.model.get_checkpoint_format(self.format)
        super()._validate()
        Assert.eq(self.config.__class__, self.model)

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ):
        # TODO v0.2: Remove backward compatibility.
        cls._handle_renamed_field(default, "checkpoint_type", "format")
        cls._handle_renamed_field(default, "checkpoint_version", "fast_llm_version")
        cls._handle_renamed_field(default, "fast_llm_config", "config")
        cls._handle_renamed_field(default, "state_shard_names", "shards")
        if "model" not in default:
            default["model"] = "gpt"
        if "format" not in default:
            default["format"] = DistributedCheckpointFormat
        if "fast_llm_version" not in default:
            default["fast_llm_version"] = "0"

        # Determine the model config class.
        from fast_llm.models.auto import model_registry

        model_config_class = default["model"]
        if isinstance(model_config_class, str):
            Assert.incl(model_config_class, model_registry)
            model_config_class = model_registry[model_config_class]
            default["model"] = model_config_class

        # TODO v0.2: Remove backward compatibility.
        if "config" not in default:
            default["config"] = {
                "base_model": model_config_class.get_base_model_config_class().from_flat_dict(
                    default.pop("model_config", {})
                ),
                "multi_stage": default.pop("multi_stage_config", {}),
                "distributed": default.pop("distributed_config", {}),
            }
        # Instantiate the config with the appropriate class
        config = default.get("config", {})
        if isinstance(config, dict):
            default["config"] = model_config_class.from_dict(config)
        return super()._from_dict(default, strict, flat)
