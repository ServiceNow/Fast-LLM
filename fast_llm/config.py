import argparse
import dataclasses
import enum
import logging
import pathlib
import traceback
import types
import typing

import yaml

from fast_llm.utils import Assert, Tag, header

logger = logging.getLogger(__name__)


_AUTO_VALIDATE = True

MISSING = Tag("<MISSING>")


class NoAutoValidate:
    """
    A context for skipping config validation to allow modifications.
    The caller is responsible for the validation.
    """

    def __enter__(self):
        global _AUTO_VALIDATE
        self._old_value = _AUTO_VALIDATE
        _AUTO_VALIDATE = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _AUTO_VALIDATE
        _AUTO_VALIDATE = self._old_value


def serialize_field(value):
    if isinstance(value, enum.Enum):
        value = value.value
    elif isinstance(value, (list, tuple)):
        value = [serialize_field(x) for x in value]
    elif isinstance(value, set):
        value = list(value)
    elif isinstance(value, dict):
        value = {key: serialize_field(value_) for key, value_ in value.items()}
    elif not isinstance(value, int | float | bool | str | None):
        value = str(value)
    return value


class _ConfigDictFormat(str, enum.Enum):
    # TODO v0.2: delete class
    flat = "flat"
    nested = "nested"
    tuple = "tuple"


class FieldHint:
    """
    A label defined for each config field, to let the user and some methods know how important each field is.
    * core:
    """

    core = "core"
    optional = "optional"
    performance = "performance"
    stability = "stability"
    feature = "feature"
    expert = "expert"
    unknown = "unknown"
    logging = "logging"
    testing = "testing"
    derived = "derived"
    setup = "setup"
    deprecated = "deprecated"
    wip = "wip"


FieldHintImportance = {
    FieldHint.core: 0,
    FieldHint.optional: 10,
    FieldHint.performance: 20,
    FieldHint.stability: 20,
    FieldHint.feature: 10,
    FieldHint.expert: 40,
    FieldHint.unknown: 20,
    FieldHint.logging: 30,
    FieldHint.testing: 40,
    FieldHint.derived: 100,
    FieldHint.setup: 90,
    FieldHint.deprecated: 80,
    FieldHint.wip: 80,
}


class FieldVerboseLevel:
    nothing = -1
    core = 0
    optional = 10
    performance = 20
    debug = 50
    everything = None


FieldHintDoc = {
    FieldHint.core: "A core configuration parameter that is expected to always be provided explicitly.",
    FieldHint.optional: "An optional parameter that may be ignored as the default tends to be good enough.",
    FieldHint.performance: "An optional parameter related to computational performance.",
    FieldHint.stability: "An optional parameter related to numerical precision and computational stability.",
    FieldHint.feature: "An parameter related to an optional feature, that should only be defined if that feature is enabled.",
    FieldHint.expert: "An advanced parameter that needs some additional expertise to be handled.",
    FieldHint.unknown: "No hint has been provided for this parameter.",
    FieldHint.logging: "An optional parameter related to logging or debug logs",
    FieldHint.testing: "A rarely defined parameter that is only meant for testing and debugging.",
    FieldHint.derived: "A parameter that is typically calculated from others.",
    FieldHint.setup: "An external parameter that must be provided in `setup` after initialization.",
    FieldHint.deprecated: "The feature is deprecated and may be removed renamed or replaced soon.",
    FieldHint.wip: "The parameter is not fully implemented yet.",
}


class Field(dataclasses.Field):
    __slots__ = (
        "desc",
        "doc",
        "hint",
        "valid",
    )

    def __init__(
        self,
        *,
        desc: str | None = None,
        doc: str | None = None,
        hint: str = FieldHint.unknown,
        # Validation function on the field to satisfy.
        # Should raise an Exception in case of failure, and return the validated value.
        # Run before the default validation (type check).
        valid: typing.Optional[typing.Callable[[typing.Any], typing.Any]] = None,
        default=dataclasses.MISSING,
        default_factory=dataclasses.MISSING,
        init: bool = True,
        repr: bool = True,
        hash=None,
        compare: bool = True,
        metadata=None,
        kw_only=dataclasses.MISSING,
    ):
        if default is not dataclasses.MISSING and default_factory is not dataclasses.MISSING:
            raise ValueError("cannot specify both default and default_factory")
        if isinstance(default_factory, type) and issubclass(default_factory, Config):
            default_factory = _ConfigFactory(default_factory)
        super().__init__(
            default=default,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
            metadata=metadata,
            kw_only=kw_only,
        )
        self.desc = desc
        self.doc = doc
        self.hint = hint
        self.valid = valid


def check_field(fn, *args, **kwargs):
    """
    Helper function to define a condition that a config field should satisfy,
    in the form of a method that may raise an exception.
    """

    def valid(x):
        fn(x, *args, **kwargs)
        return x

    return valid


def test_field(fn, *args, **kwargs):
    """
    Helper function to define a condition that a config field should satisfy,
    in the form of a function that returns a boolean.
    """

    def valid(x):
        if not fn(x, *args, **kwargs):
            raise ValueError(fn, x, args, kwargs)
        return x

    return valid


def process_field(fn, *args, **kwargs):
    """
    Helper function to apply non-standard processing during validation,
    in the form of a function that returns the processed value,
    and may raise an exception in case of an unexpected input.
    """

    def valid(x):
        return fn(x, *args, **kwargs)

    return valid


def skip_valid_if_none(fn, *args, **kwargs):
    """
    Field validation wrapper that skips validation if the field is None.
    """

    def valid(x):
        return None if x is None else fn(x, *args, **kwargs)

    return valid


class _ConfigFactory:
    """
    A dataclass default factory that prevents early validation.
    Validation is still done through the parent config if needed.
    """

    def __init__(self, factory: typing.Callable[[], "Config"] | type["Config"]):
        self._factory = factory

    def __call__(self):
        with NoAutoValidate():
            return self._factory()


class ValidationError(ValueError):
    pass


def _process_config_class(cls: type["Config"]):
    for _, field in cls.fields():
        if field._field_type is dataclasses._FIELD:
            Assert.custom(isinstance, field, Field)
    cls.__class_validated__ = True
    return cls


def config_class(cls=None):
    """
    Fast-LLM replacement for the default dataclass wrapper. Performs additional verifications.
    """

    def wrap(cls):
        Assert.custom(issubclass, cls, Config)
        return _process_config_class(dataclasses.dataclass(cls))

    # See if we're being called as @config_class or @config_class().
    if cls is None:
        # We're called with parens.
        return wrap

    # We're called as @config_class without parens.
    return wrap(cls)


@dataclasses.dataclass()
class Config:
    """
    An advanced `dataclass` with basic type checking, validation and argparse support.
    Typically, a subclass will:
    * Add some dataclass parameters.
    * Implement `_validate` which post-processes and validates a config.
    * Add new functionality.
    """

    __argparse_map__: typing.ClassVar[dict] = {}
    # We can't use @config_class on this one because it needs this class to be defined, so we assume this one is OK.
    __class_validated__: typing.ClassVar[bool] = True
    _abstract: typing.ClassVar[bool] = False
    _validated: bool = Field(init=False)

    def __post_init__(self):
        """
        Perform validation unless prevented with `NoAutoValidate`.
        In general this should not be overridden in derived classes,
        and all post-processing should be done in `_validate`
        """
        self._check_abstract()
        self._validated = False
        if _AUTO_VALIDATE:
            self.validate()

    def validate(self, *, _is_validating=False):
        """
        Validate a class and mark it as read-only
        This should not be overridden in derived classes.
        """
        if not self._validated:
            try:
                self._validate()
            except ValidationError as e:
                if _is_validating:
                    raise
                else:
                    raise ValueError("\n".join(e.args))
            self._validated = True
        return self

    def _validate(self):
        """
        Verify that the type hints are respected,
        and fix some know entries compatible with the type hint (ex. `int -> float`, `str -> pathlib.Path`)

        Can be extended to add custom post-processing (typically before the super() call)
        and validation (typically after)
        """
        errors = []
        for name, field in self.fields():
            if not field.init or field._field_type == dataclasses._FIELD_CLASSVAR:  # noqa
                continue
            value = getattr(self, name)
            try:
                new_value = value if field.valid is None else field.valid(value)
                new_value, valid = self._post_process_field(new_value, field.type, field, errors)
                if new_value is not value:
                    setattr(self, name, new_value)
                if not valid:
                    raise ValidationError(f"Invalid type `{type(value)}` (expected `{field.type}`)")
            except ValidationError as e:
                errors.append(f"Validation failed for field `{name}` in class {self.__class__.__name__}:)")
                errors.extend(["  " + arg for arg in e.args])
            except Exception as e:
                errors.append(
                    f"Validation failed for field `{name}` in class {self.__class__.__name__}: {', '.join(e.args)}"
                    f"\n\n====================== stack trace ========================\n"
                    + traceback.format_exc()
                    + "===========================================================\n"
                )

        if errors:
            # TODO: Option to show traceback for errors.
            raise ValidationError(*errors)

    def __setattr__(self, key, value):
        """
        Make the class read-only after validation.
        """
        # `_validated` may not be set yet.
        if getattr(self, "_validated", False):
            if value is getattr(self, key):
                # Allow setting the exact same object to facilitate setup of cross-dependencies.
                # Ex. allow re-setting cross-dependencies of already validated sub-configs.
                return
            raise RuntimeError()
        super().__setattr__(key, value)

    def __delattr__(self, key):
        """
        Make the class read-only after validation.
        """
        if getattr(self, "_validated", False):
            raise RuntimeError()
        super().__delattr__(key)

    @classmethod
    def _post_process_field(cls, x, type_, field: Field, errors: list[str]):
        if type_ is typing.Any:
            valid = True
        elif type_ is types.NoneType:
            if x == "":
                x = None
            valid = x is None
        elif isinstance(type_, types.UnionType):
            # Takes care of Optional too
            valid = False
            if types.NoneType in type_.__args__ and x == "":
                x = None
            for subtype in type_.__args__:
                x_, valid = cls._post_process_field(x, subtype, field, errors)
                if valid:
                    x = x_
                    break
        elif hasattr(type_, "__origin__"):
            origin = type_.__origin__
            if origin is typing.Union:
                raise NotImplementedError(f"Use python 3.10 format instead ({type_})")
            elif origin in (list, set, tuple):
                valid = isinstance(x, (origin, list, tuple))
                if valid and hasattr(type_, "__args__"):
                    args = type_.__args__
                    if origin is tuple and not (len(args) == 2 and args[1] is ...):
                        if len(x) == len(args):
                            x, valid_ = zip(
                                *[cls._post_process_field(y, arg, field, errors) for y, arg in zip(x, args)]
                            )
                            valid = all(valid_)
                        else:
                            valid = False
                    else:
                        if origin is not tuple:
                            Assert.eq(len(args), 1)
                        arg = args[0]
                        x, valid_ = zip(*[cls._post_process_field(y, arg, field, errors) for y in x])
                        x = origin(x)
                        valid = all(valid_)
            else:
                raise NotImplementedError(origin)
        elif not isinstance(type_, type):
            raise NotImplementedError(type_)
        elif issubclass(type_, Config):
            if isinstance(x, dict):
                raise ValueError(f"Nested config not properly initialized. Use `{type_.__name__}.from_dict` instead.")
            valid = isinstance(x, type_)
            x.validate(_is_validating=True)
        else:
            if hasattr(type_, "__fast_llm_validator__"):
                x = type_.__fast_llm_validator__(x)
            elif type_ is float and isinstance(x, int):
                # Ints are ok too.
                x = float(x)
            elif issubclass(type_, enum.Enum) and not isinstance(x, type_) and issubclass(type_, type(x)):
                # Enum values are ok too.
                x = type_(x)
            elif issubclass(type_, pathlib.PurePath) and isinstance(x, str):
                # Str paths are ok too.
                x = type_(x)
            valid = isinstance(x, type_)
        return x, valid

    @classmethod
    def fields(cls) -> typing.Iterable[tuple[str, Field]]:
        """
        An iterable for the field definitions of a `Config` class.
        """
        return cls.__dataclass_fields__.items()  # noqa

    @classmethod
    def get_field(cls, name) -> Field:
        return cls.__dataclass_fields__[name]  # noqa

    def _to_dict(
        self,
        verbose: int | None = None,
        all_fields: bool = False,
        format_: _ConfigDictFormat = _ConfigDictFormat.nested,
        serializable: bool = False,
    ):
        """
        Serialize the config to a dict that can (generally) be used to reconstruct an identical `Config`.
        When not flat, the dict includes a `__class__` entry which allows support for derived classes.

        Args:
            all_fields: Include the derived fields, with `init=False`.
            format_: The config format used to represent nested configs. Options:
              * `ConfigDictFormat.flat`: Flatten nested configs into a flat dict, keep only the innermost config keys.
                The user is responsible for preventing name clashes.
                Legacy format, used mainly for argparse configuration.
              * `ConfigDictFormat.nested`: Preserve the nested config structure by returning nested dicts.
                Also save a `__class__` entry to support derived classes. Standard format.
              * `ConfigDictFormat.tuple`: Preserve the nested config structure by returning tuples of keys.
                Used for config updates.
            serializable: Ensure the dict is serializable to json or yaml. Information may be lost.
        """
        arg_dict = {}
        for name, field in self.fields():
            value = getattr(self, name, MISSING)
            if (not field.init or field._field_type == dataclasses._FIELD_CLASSVAR) and not (all_fields):
                # Exclude class variables and derived fields unless requested explicitly.
                continue
            elif isinstance(value, Config):
                field_dict = value._to_dict(
                    verbose=verbose,
                    all_fields=all_fields,
                    format_=format_,
                    serializable=serializable,
                )
                if format_ == _ConfigDictFormat.flat:
                    arg_dict.update(field_dict)
                elif format_ == _ConfigDictFormat.tuple:
                    arg_dict.update({(name,) + name_: value_ for name_, value_ in field_dict.items()})
                elif format_ == _ConfigDictFormat.nested:
                    if len(field_dict) > 0 or all_fields:
                        arg_dict[name] = field_dict
                else:
                    raise NotImplementedError(format_)
            elif verbose is None or FieldHintImportance[field.hint] <= verbose or value != field.default:
                arg_dict[(name,) if format_ == _ConfigDictFormat.tuple else name] = (
                    serialize_field(value) if serializable else value
                )
        return arg_dict

    def to_flat_dict(self, verbose: int | None = FieldVerboseLevel.core):
        # TODO v0.2: Remove flat format
        return self._to_dict(verbose=verbose, format_=_ConfigDictFormat.flat, serializable=True)

    def to_copy(
        self,
        *updates: typing.Union["Config", dict[str | tuple[str, ...], typing.Any]],
        strict: bool = True,
    ):
        return self.from_dict(self, *updates, strict=strict)

    def to_serialized(self, verbose: int | None = FieldVerboseLevel.core):
        return self._to_dict(verbose=verbose, format_=_ConfigDictFormat.nested, serializable=True)

    def to_logs(
        self,
        verbose: int | None = FieldVerboseLevel.core,
        log_fn=logger.info,
        title: str | None = None,
        width: int = 80,
        fill_char: str = "-",
    ):
        arg_dict = self.to_serialized(verbose=verbose)
        if title is None:
            title = self._class_name()
        return log_fn(
            f"\n{header(title, width, fill_char)}"
            f"\n{yaml.safe_dump(arg_dict, sort_keys=False)}"
            f"{header('end', width, fill_char)}"
        )

    @classmethod
    def _class_name(cls):
        return f"{cls.__module__}.{cls.__name__}"

    @classmethod
    def _to_argparse(cls, parser: argparse.ArgumentParser):
        """
        Add arguments for the config and its sub-configs to an existing parser.
        The whole config hierarchy is flattened (see `to_dict`),
        and the user is responsible for preventing name clashes.
        """
        cls._check_abstract()
        field: Field
        for name, field in cls.fields():
            try:
                if not field.init or field._field_type == dataclasses._FIELD_CLASSVAR:  # noqa
                    assert name not in cls.__argparse_map__
                    continue
                if name in cls.__argparse_map__:
                    if cls.__argparse_map__[name] is None:
                        continue
                    argparse_kwargs = cls.__argparse_map__[name].copy()
                else:
                    argparse_kwargs = {}
                if "type" in argparse_kwargs:
                    is_list = False
                else:
                    type_ = field.type
                    is_list = isinstance(type_, types.GenericAlias) and type_.__origin__ in (list, set)
                    if is_list:
                        Assert.eq(len(type_.__args__), 1)
                        Assert.eq(type_.__origin__, field.default_factory)
                        type_ = type_.__args__[0]
                    elif isinstance(field.default_factory, _ConfigFactory):
                        Assert.custom(issubclass, type_, Config)
                    elif field.default_factory is not dataclasses.MISSING:
                        raise NotImplementedError(name)
                    if (
                        isinstance(type_, types.UnionType)
                        and len(type_.__args__) == 2
                        and type_.__args__[1] is type(None)
                    ):
                        # Optional
                        type_ = type_.__args__[0]
                        # Would break other things.
                        Assert.not_custom(issubclass, type_, Config)
                    Assert.custom(isinstance, type_, type)
                    if issubclass(type_, Config):
                        type_._to_argparse(parser)
                        continue
                    if hasattr(type_, "__fast_llm_argparse_type__"):
                        type_ = type_.__fast_llm_argparse_type__
                    elif type_ is bool:
                        type_ = lambda x: bool(int(x))
                    elif not issubclass(type_, (str, int, float, pathlib.Path)):
                        raise NotImplementedError(name, type_)
                    argparse_kwargs["type"] = type_
                if "required" not in argparse_kwargs:
                    argparse_kwargs["required"] = (
                        field.default is dataclasses.MISSING and "default" not in argparse_kwargs and not is_list
                    )
                if "default" not in argparse_kwargs and not is_list and not argparse_kwargs["required"]:
                    argparse_kwargs["default"] = field.default
                if "nargs" not in argparse_kwargs and is_list:
                    argparse_kwargs["nargs"] = "*"
                if "help" not in argparse_kwargs:
                    argparse_kwargs["help"] = field.desc
                parser.add_argument(
                    f"--{name}",
                    f"--{name.replace('_', '-')}",
                    **argparse_kwargs,
                )
            except Exception:
                raise ValueError(f"Failed to add argparse argument for field {name}")
        return parser

    @classmethod
    def from_dict(
        cls,
        default: typing.Union["Config", dict],
        *updates: typing.Union["Config", dict[str | tuple[str, ...], typing.Any]],
        strict: bool = True,
    ):
        if isinstance(default, Config):
            default = default._to_dict()
        for update in updates:
            if isinstance(update, Config):
                update = update._to_dict(format_=_ConfigDictFormat.tuple)
            for keys, value in update.items():
                if isinstance(keys, str):
                    default[keys] = value
                else:
                    dict_to_update = default
                    for key in keys[:-1]:
                        if key not in dict_to_update:
                            dict_to_update[key] = {}
                        dict_to_update = dict_to_update[key]
                    dict_to_update[keys[-1]] = value

        return cls._from_dict(default, strict)

    @classmethod
    def from_flat_dict(
        cls,
        default: dict,
        strict: bool = True,
    ):
        # TODO v0.2: Separate dict only needed for flat format
        return cls._from_dict(default, strict, True)

    @classmethod
    def _from_dict(
        cls,
        default: dict,
        strict: bool = True,
        flat: bool = False,
    ):
        cls._check_abstract()
        # TODO v0.2: Separate dict only needed for flat format
        out_arg_dict = {}

        # TODO v0.2: Remove backward compatibility fix
        if "__class__" in default:
            del default["__class__"]

        for name, field in cls.fields():
            if not field.init or field._field_type == dataclasses._FIELD_CLASSVAR:  # noqa
                continue
            if isinstance(field.type, type) and issubclass(field.type, Config):
                # Do not validate yet in case the root class sets cross-dependencies in validation.
                with NoAutoValidate():
                    if flat:
                        out_arg_dict[name] = field.type._from_dict(default, False, True)
                    else:
                        out_arg_dict[name] = field.type._from_dict(default.pop(name, {}), strict)
            elif name in default:
                out_arg_dict[name] = default.pop(name)
        if strict and default:
            raise ValueError(cls, list(default))
        out = cls(**out_arg_dict)  # noqa
        if _AUTO_VALIDATE:
            out.validate()
        return out

    @classmethod
    def from_flat_args(cls, args: list[str] | None = None):
        """
        TODO v0.2: Remove flat format
        Make an argument parser for the config and its sub-configs,
        parse the provided args (or `sys.argv`),
        and create a config from the resulting namespace.
        """
        return cls.from_flat_dict(cls._to_argparse(argparse.ArgumentParser()).parse_args(args).__dict__.copy())

    @classmethod
    def _check_abstract(cls):
        if cls._abstract:
            raise RuntimeError(f"{cls.__name__} is abstract")
        if not cls.__class_validated__:
            raise RuntimeError(f"{cls.__name__} hasn't been validated. Make sure to use the @config_class decorator.")

    def __init_subclass__(cls, **kwargs):
        """
        We need to postpone validation until the class has been processed by the dataclass wrapper.
        """
        assert (
            cls.__class_validated__
        ), f"Parent class of config class {cls.__name__} has not been validated. Make sure to use the @config_class decorator."
        cls.__class_validated__ = False
