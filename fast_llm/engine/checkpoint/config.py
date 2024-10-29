# TODO: Use packaging.version? (Safer but extra requirement)
import abc
import enum
import logging
import pathlib
import typing
import warnings

import yaml

from fast_llm.config import Config, Field, FieldHint, FieldUpdate, check_field, config_class
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.multi_stage.config import CheckpointMetadata
    from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel

logger = logging.getLogger(__name__)

# TODO: Use packaging.version? (Safer but extra requirement)
CHECKPOINT_VERSION = "0.1"
KNOWN_CHECKPOINT_VERSIONS = ("0", "0.1")


def export_safetensors_metadata(metadata):
    """
    Safetensor only accepts string entries, so we convert to string explicitly.
    We use yaml rather than json because json requires explicit quotation marks on strings, which breaks things.
    (ex. "format": "pt" becomes '"pt"' which breaks huggingface models.)
    We avoid using safe_dump for scalars because it adds junk ("\n...\n") at the end of the string
    (decoding is unaffected.)
    """
    return {
        key: str(value) if isinstance(value, (str, int, float, bool)) else yaml.safe_dump(value)
        for key, value in metadata.items()
    }


def import_safetensors_metadata(metadata):
    return {key: yaml.safe_load(value) for key, value in metadata.items()}


class CheckpointFormat(str):
    # Distributed checkpoint for fast checkpointing and resuming.
    distributed = "distributed"
    # Model state dict, for safe long-term storage in Fast-LLM format.
    state_dict = "state_dict"


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
    format: str = Field(
        default=CheckpointFormat.state_dict,
        desc="Format of the checkpoint.",
        hint=FieldHint.core,
    )

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ):
        # TODO v0.2: Remove.
        cls._handle_renamed_field(default, "imported_type", "model_type")
        if "model_type" in default:
            warnings.warn(
                "`CheckpointConfigBase.model_type` is deprecated."
                " Instead, use the model name directly as the checkpoint format."
            )
            if default.get("format", None) in ("huggingface", "external"):
                default["format"] = default.get("model_type")
                if default["format"] is None:
                    default["format"] = "auto"
            del default["model_type"]
        return super()._from_dict(default, strict, flat)


@config_class()
class CheckpointStateConfigBase(Config):
    _abstract = True
    # Defaults and descriptions are set in derived classes.
    model_weights: bool = Field(hint=FieldHint.feature)
    optimizer_state: bool | None = Field(hint=FieldHint.feature)

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
class CheckpointStateSaveConfigBase(CheckpointStateConfigBase):
    model_weights: bool = Field(desc="Save the model weights.")
    optimizer_state: bool | None = FieldUpdate(
        desc="Save the optimizer state. Default: save if supported, or as as specified by the `format`."
    )


@config_class()
class CheckpointStateLoadConfigBase(CheckpointStateConfigBase):
    # TODO: Is type override ok?
    model_weights: bool = Field(desc="Load the model weights.")
    optimizer_state: bool = FieldUpdate(default=False, desc="Load the optimizer state.")


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
class CheckpointSaveConfig(CheckpointSaveMetadataConfig, CheckpointStateSaveConfigBase, CheckpointSaveConfigBase):
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
class CheckpointLoadConfig(CheckpointLoadMetadataConfig, CheckpointStateLoadConfigBase):
    _abstract = False

    def _validate(self):
        super()._validate()
        if self.format == CheckpointFormat.distributed:
            assert self.load_config.load_architecture


class CheckpointHandler(abc.ABC):
    support_optimizer_state: typing.ClassVar[bool]

    def __init__(self, model: "FastLLMModel"):
        self._model = model

    @property
    @abc.abstractmethod
    def _include_optimizer_state(self) -> bool:
        pass

    @property
    def _num_shards(self):
        return len(self._model.state_shard_names) if self._include_optimizer_state else 1

    @property
    def _shard_names(self):
        return self._model.state_shard_names if self._include_optimizer_state else self._model.state_shard_names[:1]


class CheckpointSaver(CheckpointHandler):
    def __init__(self, model: "FastLLMModel", config: CheckpointSaveConfig):
        super().__init__(model)
        self._config = config

    # TODO: save_metadata?

    @property
    def _include_optimizer_state(self):
        if self._config.optimizer_state is None:
            return self.support_optimizer_state
        if self._config.optimizer_state:
            # TODO: This is not automatically checked in config validation.
            assert self.support_optimizer_state
        return self._config.optimizer_state

    @abc.abstractmethod
    def save(self, metadata: "CheckpointMetadata"):
        pass


class CheckpointLoader(CheckpointHandler):
    def __init__(self, model: "FastLLMModel", config: CheckpointLoadConfig):
        super().__init__(model)
        self._config = config

    @property
    def _include_optimizer_state(self):
        assert self.support_optimizer_state is not None
        if self._config.optimizer_state:
            # TODO: This is not automatically checked in config validation.
            assert self.support_optimizer_state
        return self._config.optimizer_state

    @classmethod
    @abc.abstractmethod
    def load_metadata(cls, config: CheckpointLoadMetadataConfig) -> "CheckpointMetadata":
        pass

    @abc.abstractmethod
    def load(self, metadata: "CheckpointMetadata"):
        pass
