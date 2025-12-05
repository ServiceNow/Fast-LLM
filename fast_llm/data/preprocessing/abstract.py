import logging
import typing
import warnings

from fast_llm.config import Config, config_class

logger = logging.getLogger(__name__)


@config_class(registry=True)
class PreprocessingConfig(Config):
    """
    Base preprocessing configuration, with dynamic registry so configs can be saved with memmap datasets.
    """

    _abstract = True

    @classmethod
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        if cls is PreprocessingConfig and cls.get_subclass(default.get("type")) is None:
            # Default subclass, necessary for loading configs where some components could be absent.
            return NullPreprocessingConfig._from_dict(default, strict)
        return super()._from_dict(default, strict=strict)

    def check_compatibility(self, preprocessing: typing.Self) -> None:
        """
        Check whether a dataset preprocessed with `self` can produce samples for a model that requires `preprocessing`.
        """
        raise NotImplementedError()


@config_class(dynamic_type={PreprocessingConfig: "none"})
class NullPreprocessingConfig(PreprocessingConfig):
    """
    Configuration for unspecified preprocessing.
    """

    _abstract = False

    def check_compatibility(self, preprocessing: typing.Self) -> None:
        if not isinstance(preprocessing, NullPreprocessingConfig):
            warnings.warn(f"Preprocessing configuration not specified, could not check compatibility with the model.")
