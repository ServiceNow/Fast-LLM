# TODO: Use packaging.version? (Safer but extra requirement)
import abc
import enum
import logging
import pathlib
import typing

import yaml

from fast_llm.config import Config, Field, FieldHint, FieldUpdate, check_field, config_class, skip_valid_if_none
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.checkpoint.distributed import DistributedCheckpointHandler
    from fast_llm.engine.checkpoint.state_dict import FastLLMCheckpointHandler
    from fast_llm.engine.multi_stage.config import CheckpointMetadata, FastLLMModelConfig
    from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel

logger = logging.getLogger(__name__)


def export_safetensors_metadata(metadata: dict[str, typing.Any]) -> dict[str, str]:
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


def import_safetensors_metadata(metadata: dict[str, str]) -> dict[str, typing.Any]:
    return {key: yaml.safe_load(value) for key, value in metadata.items()}


class CheckpointFormat(abc.ABC):
    # A data structure to make information about a checkpoint format accessible during validation.
    name: typing.ClassVar[str]
    support_optimizer: typing.ClassVar[bool] = True
    support_saving: typing.ClassVar[bool] = True
    support_loading: typing.ClassVar[bool] = True
    enforce_architecture_match: typing.ClassVar[bool] = False

    @classmethod
    @abc.abstractmethod
    def get_handler_class(cls) -> type["CheckpointHandler"]:
        pass

    @classmethod
    def __fast_llm_serialize__(cls) -> str:
        return cls.name


class DistributedCheckpointFormat(CheckpointFormat):
    # TODO v0.3: Add `enforce_version_match`
    name: typing.ClassVar[str] = "distributed"
    enforce_architecture_match: typing.ClassVar[bool] = True

    @classmethod
    def get_handler_class(cls) -> type["DistributedCheckpointHandler"]:
        from fast_llm.engine.checkpoint.distributed import DistributedCheckpointHandler

        return DistributedCheckpointHandler


class FastLLMCheckpointFormat(CheckpointFormat):
    name: typing.ClassVar[str] = "fast_llm"

    @classmethod
    def get_handler_class(cls) -> type["FastLLMCheckpointHandler"]:
        from fast_llm.engine.checkpoint.state_dict import FastLLMCheckpointHandler

        return FastLLMCheckpointHandler


class ModelConfigType(str, enum.Enum):
    none = "none"
    architecture = "architecture"
    model = "model"
    fast_llm = "fast_llm"

    @property
    def load_architecture(self) -> bool:
        return self != ModelConfigType.none

    @property
    def load_base_model(self) -> bool:
        return self in (ModelConfigType.model, ModelConfigType.fast_llm)

    @property
    def load_fast_llm(self) -> bool:
        return self == ModelConfigType.fast_llm


@config_class()
class CheckpointConfigBase(Config):
    _abstract = True
    # Note: the `format` may be a str when configuring from file or cli.
    #   The actual class should be set through `setup` in a parent config validation.
    format: type[CheckpointFormat] = Field(
        default=FastLLMCheckpointFormat,
        desc="Format of the checkpoint.",
        hint=FieldHint.core,
    )

    def _validate(self) -> None:
        if not isinstance(self.format, type) or not issubclass(self.format, CheckpointFormat):
            # Would break anyway, but this makes the error more explicit.
            raise ValueError("Please call `setup` first to set the checkpoint format.")
        super()._validate()

    def setup(self, model_config: "FastLLMModelConfig| type[FastLLMModelConfig]") -> None:
        format = model_config.get_checkpoint_format(self.format)
        if self._validated:
            Assert.eq(self.format, format)
        else:
            self.format = format


@config_class()
class CheckpointStateConfigBase(CheckpointConfigBase):
    _abstract = True
    # Defaults and descriptions are set in derived classes.
    model_weights: bool = Field(default=True, hint=FieldHint.feature)
    optimizer_state: bool = Field(default=None, hint=FieldHint.feature)

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        cls._handle_renamed_field(default, "load_weights", "model_weights")
        cls._handle_renamed_field(default, "load_optimizer", "optimizer_state")
        return super()._from_dict(default, strict, flat)


@config_class()
class CheckpointSaveConfigBase(CheckpointConfigBase):
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
class CheckpointStateSaveConfigBase(CheckpointSaveConfigBase, CheckpointStateConfigBase):
    model_weights: bool = FieldUpdate(desc="Save the model weights.")
    optimizer_state: bool = FieldUpdate(desc="Save the optimizer state. Default: save if supported by the `format`.")

    def _validate(self) -> None:
        if self.optimizer_state is None:
            # TODO: Make sure it's a type
            self.optimizer_state = self.format.support_optimizer
        super()._validate()
        if self.optimizer_state:
            assert self.format.support_optimizer


@config_class()
class CheckpointPathConfigBase(CheckpointConfigBase):
    _abstract = True
    path: pathlib.Path | None = Field(
        default=None,
        desc="Location of the checkpoint.",
        hint=FieldHint.core,
    )
    timeout: float | None = Field(
        default=None,
        desc="Custom timeout for lengthy operations.",
        hint=FieldHint.optional,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )


@config_class()
class CheckpointSaveMetadataConfig(CheckpointPathConfigBase):
    _abstract = False


@config_class()
class CheckpointSaveConfig(CheckpointSaveMetadataConfig, CheckpointStateSaveConfigBase):
    _abstract = False


@config_class()
class CheckpointLoadMetadataConfig(CheckpointPathConfigBase):
    _abstract = False

    load_config: ModelConfigType = Field(
        default=ModelConfigType.architecture,
        desc="Configuration to save/load.",
        hint=FieldHint.core,
    )

    def _validate(self) -> None:
        super()._validate()
        if self.format.enforce_architecture_match:
            assert self.load_config.load_architecture

    @property
    def compare_log_fn(self):
        return ValueError if self.load_config.load_architecture else logger.warning


@config_class()
class CheckpointLoadConfig(CheckpointLoadMetadataConfig, CheckpointStateConfigBase):
    _abstract = False

    model_weights: bool = FieldUpdate(desc="Load the model weights.")
    optimizer_state: bool = FieldUpdate(default=False, desc="Load the optimizer state.")

    def _validate(self) -> None:
        super()._validate()
        if self.optimizer_state:
            assert self.format.support_optimizer


class CheckpointHandler(abc.ABC):
    format: typing.ClassVar[type[CheckpointFormat]]

    def __init__(self, model: "FastLLMModel"):
        self._model = model

    # TODO: save_metadata?

    @classmethod
    @abc.abstractmethod
    def load_metadata(cls, config: CheckpointLoadMetadataConfig) -> "CheckpointMetadata":
        pass

    @abc.abstractmethod
    def save(self, config: CheckpointSaveConfig, metadata: "CheckpointMetadata"):
        pass

    @abc.abstractmethod
    def load(self, config: CheckpointLoadConfig, metadata: "CheckpointMetadata"):
        pass

    def get_num_shards(self, config: CheckpointStateConfigBase) -> int:
        return len(self._model.state_shard_names) if config.optimizer_state else 1

    def get_shard_names(self, config: CheckpointStateConfigBase) -> tuple[str, ...]:
        return self._model.state_shard_names if config.optimizer_state else self._model.state_shard_names[:1]
