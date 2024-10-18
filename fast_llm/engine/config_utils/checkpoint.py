# TODO: Use packaging.version? (Safer but extra requirement)
import enum
import logging
import pathlib
import warnings

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)

# TODO: Use packaging.version? (Safer but extra requirement)
CHECKPOINT_VERSION = "0.1"
KNOWN_CHECKPOINT_VERSIONS = ("0", "0.1")


class CheckpointType(str, enum.Enum):
    # Distributed checkpoint for fast checkpointing and resuming.
    distributed = "distributed"
    # Model state dict, for safe long-term storage in Fast-LLM format.
    state_dict = "state_dict"
    # A checkpoint format external to Fast-LLM.
    external = "external"


class LoadConfig(str, enum.Enum):
    none = "none"
    architecture = "architecture"
    model = "model"
    fast_llm = "fast_llm"

    @property
    def load_architecture(self):
        return self != LoadConfig.none

    @property
    def load_base_model(self):
        return self in (LoadConfig.model, LoadConfig.fast_llm)

    @property
    def load_fast_llm(self):
        return self == LoadConfig.fast_llm


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
    format: CheckpointType = Field(
        default=CheckpointType.distributed,
        desc="Format of the checkpoint.",
        hint=FieldHint.core,
    )
    model_type: str | None = Field(
        default=None,
        desc="Model type for external models (ex. Huggingace model name).",
        hint=FieldHint.feature,
    )
    load_config: LoadConfig = Field(
        default=LoadConfig.architecture,
        desc="Configuration to save/load.",
        hint=FieldHint.core,
    )
    fast_llm_config: bool = Field(
        default=False,
        desc="Save/load the full fast-llm model configuration, including the distributed and multi-stage configurations.",
        hint=FieldHint.feature,
    )

    @property
    def compare_log_fn(self):
        return ValueError if self.load_config.load_architecture else logger.warning

    @classmethod
    def _from_dict(
        cls,
        default: dict[str],
        strict: bool = True,
        flat: bool = False,
    ):
        # TODO v0.2: Remove.
        if default.get("format", None) == "huggingface":
            warnings.warn(f"`huggingface` checkpoint format has been renamed to `external`.")
            default["format"] = CheckpointType.external.value
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
        default: dict[str],
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
class CheckpointMetadataConfig(CheckpointPathConfigBase, CheckpointConfigBase):
    _abstract = False


@config_class()
class CheckpointSaveConfig(CheckpointMetadataConfig, CheckpointStateConfigBase, CheckpointSaveConfigBase):
    _abstract = False


@config_class()
class CheckpointLoadConfig(CheckpointMetadataConfig, CheckpointStateConfigBase):
    _abstract = False


# @config_class()
# class TrainingExportConfig(CheckpointConfigBase, CheckpointStateConfigBase, CheckpointSaveConfigBase):
#    _abstract=False
