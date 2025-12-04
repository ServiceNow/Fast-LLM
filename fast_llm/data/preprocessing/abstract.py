import typing

from fast_llm.config import Config, config_class


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


@config_class(dynamic_type={PreprocessingConfig: "none"})
class NullPreprocessingConfig(PreprocessingConfig):
    """
    Configuration for unspecified preprocessing.
    """

    _abstract = False
