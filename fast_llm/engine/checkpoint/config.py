# TODO: Use packaging.version? (Safer but extra requirement)
import enum
import logging
import pathlib
import typing
import warnings

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)

# TODO: Use packaging.version? (Safer but extra requirement)
CHECKPOINT_VERSION = "0.1"
KNOWN_CHECKPOINT_VERSIONS = ("0", "0.1")


class CheckpointFormat(str, enum.Enum):
    # Distributed checkpoint for fast checkpointing and resuming.
    distributed = "distributed"
    # Model state dict, for safe long-term storage in Fast-LLM format.
    state_dict = "state_dict"
    # A checkpoint format external to Fast-LLM.
    external = "external"


class ModelConfigType(str, enum.Enum):
    none = "none"
    architecture = "architecture"
    model = "model"
    fast_llm = "fast_llm"

    @property
    def load_architecture(self):
        return self != ModelConfigType.none

    @property
    def load_base_model(self):
        return self in (ModelConfigType.model, ModelConfigType.fast_llm)

    @property
    def load_fast_llm(self):
        return self == ModelConfigType.fast_llm


@config_class()
class CheckpointPathConfigBase(Config):
    _abstract = True
    path: pathlib.Path | None = Field(
        default=None,
        desc="Location of the checkpoint.",
        hint=FieldHint.core,
    )


@config_class()
class CheckpointConfigBase(Config):
    _abstract = True
    format: CheckpointFormat = Field(
        default=CheckpointFormat.distributed,
        desc="Format of the checkpoint.",
        hint=FieldHint.core,
    )
    model_type: str | None = Field(
        default=None,
        desc="Model type for external models (ex. Huggingace model name).",
        hint=FieldHint.feature,
    )

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ):
        # TODO v0.2: Remove.
        if default.get("format", None) == "huggingface":
            warnings.warn(f"`huggingface` checkpoint format has been renamed to `external`.")
            default["format"] = CheckpointFormat.external.value
        cls._handle_renamed_field(default, "imported_type", "model_type")
        return super()._from_dict(default, strict, flat)


@config_class()
class CheckpointStateConfigBase(Config):
    _abstract = True
    model_weights: bool = Field(default=True, desc="Save/load the model weights.", hint=FieldHint.feature)
    optimizer_state: bool = Field(default=False, desc="Save/load the optimizer state.", hint=FieldHint.feature)

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ):
        cls._handle_renamed_field(default, "load_weights", "model_weights")
        cls._handle_renamed_field(default, "load_optimizer", "optimizer_state")
        return super()._from_dict(default, strict, flat)


@config_class()
class CheckpointSaveConfigBase(Config):
    _abstract = True
    parameters_per_file: int = Field(
        default=2**32,
        desc="Limit the number of parameters saved in each file.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 2**20),
    )
    data_type: DataType | None = Field(
        default=None,
        desc="Data type to save the checkpoint.",
        hint=FieldHint.feature,
    )


@config_class()
class CheckpointSaveMetadataConfig(CheckpointPathConfigBase, CheckpointConfigBase):
    _abstract = False


@config_class()
class CheckpointSaveConfig(CheckpointSaveMetadataConfig, CheckpointStateConfigBase, CheckpointSaveConfigBase):
    _abstract = False


@config_class()
class CheckpointLoadMetadataConfig(CheckpointPathConfigBase, CheckpointConfigBase):
    _abstract = False

    load_config: ModelConfigType = Field(
        default=ModelConfigType.architecture,
        desc="Configuration to save/load.",
        hint=FieldHint.core,
    )

    def _validate(self):
        super()._validate()
        if self.format == CheckpointFormat.distributed:
            assert self.load_config.load_architecture

    @property
    def compare_log_fn(self):
        return ValueError if self.load_config.load_architecture else logger.warning


@config_class()
class CheckpointLoadConfig(CheckpointLoadMetadataConfig, CheckpointStateConfigBase):
    _abstract = False

    def _validate(self):
        super()._validate()
        if self.format == CheckpointFormat.external:
            # TODO: Support optimizer?
            assert not self.optimizer_state
