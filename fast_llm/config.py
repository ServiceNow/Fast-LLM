import argparse
import dataclasses
import enum
import logging
import pathlib
import types
import typing

import requests
import yaml

from fast_llm.utils import Assert, Tag

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
    elif isinstance(value, dict):
        value = {key: serialize_field(value_) for key, value_ in value.items()}
    elif not isinstance(value, int | float | bool | str | None):
        value = str(value)
    return value


class ConfigDictFormat(str, enum.Enum):
    flat = "flat"
    nested = "nested"
    tuple = "tuple"


class FieldHint:
    """
    A label defined for each config field, to let the user and some methods know how important each field is.
    * core:
    """

    core = "core"
    architecture = "architecture"
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


FieldHintDoc = {
    FieldHint.core: "A core configuration parameter that is expected to always be provided explicitly.",
    FieldHint.architecture: "A core configuration parameter that defines a base model architecture, i.e., that cannot be changed without breaking the model.",
    FieldHint.optional: "An optional parameter that may be ignored as the default tends to be good enough.",
    FieldHint.performance: "An optional parameter related to computational performance.",
    FieldHint.stability: "An optional parameter related to numerical precision and computational stability.",
    FieldHint.feature: "An parameter related to an optional feature, that should only be defined if that feature is enabled.",
    FieldHint.expert: "An advanced parameter that needs some additional expertise to be handled.",
    FieldHint.unknown: "No hint has been provided for this parameter.",
    FieldHint.logging: "An optional parameter related logging or debug logs",
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


def config_class(
    cls=None,
    /,
    *,
    init=True,
    repr=True,
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False,
    match_args=True,
    kw_only=False,
    slots=False,
):
    """
    Fast-LLM replacement for the default dataclass wrapper. Performs additional verifications.
    """

    def wrap(cls):
        Assert.custom(issubclass, cls, Config)
        return _process_config_class(
            dataclasses._process_class(cls, init, repr, eq, order, unsafe_hash, frozen, match_args, kw_only, slots)
        )

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
        self.check_abstract()
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
                    raise TypeError(f"Invalid type `{type(value)}` (expected `{field.type}`)")
            except ValidationError as e:
                errors.append(f"Validation failed for field `{name}` in class {self.__class__.__name__}:)")
                errors.extend(["  " + arg for arg in e.args])
            except Exception as e:
                errors.append(
                    f"Validation failed for field `{name}` in class {self.__class__.__name__}: {', '.join(e.args)}"
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
            # TODO: Make a dict of special classes instead?
            if type_ is float and isinstance(x, int):
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

    def items(self, all_fields=False):
        """
        A generator for the field values of a `Config` instance.
        Optionally include the derived fields, with `init=False`.
        """
        return (
            (key, getattr(self, key, MISSING))
            for key, field in self.fields()
            if all_fields or (field.init and field._field_type != dataclasses._FIELD_CLASSVAR)  # noqa
        )

    def to_dict(
        self, all_fields: bool = False, format_: ConfigDictFormat = ConfigDictFormat.flat, serializable: bool = False
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
        for name, value in self.items(all_fields=all_fields):
            if isinstance(value, Config):
                field_dict = value.to_dict(all_fields=all_fields, format_=format_, serializable=serializable)
                if format_ == ConfigDictFormat.flat:
                    arg_dict.update(field_dict)
                elif format_ == ConfigDictFormat.nested:
                    arg_dict[name] = field_dict
                elif format_ == ConfigDictFormat.tuple:
                    arg_dict.update({(name,) + name_: value_ for name_, value_ in field_dict.items()})
                else:
                    raise NotImplementedError(format_)
            else:
                arg_dict[(name,) if format_ == ConfigDictFormat.tuple else name] = (
                    serialize_field(value) if serializable else value
                )
        if format_ == ConfigDictFormat.nested:
            arg_dict["__class__"] = self.__class__.__name__ if serializable else self.__class__
        return arg_dict

    @classmethod
    def to_argparse(cls, parser: argparse.ArgumentParser):
        """
        Add arguments for the config and its sub-configs to an existing parser.
        The whole config hierarchy is flattened (see `to_dict`),
        and the user is responsible for preventing name clashes.
        """
        cls.check_abstract()
        field: Field
        for name, field in cls.fields():
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
                if isinstance(type_, types.UnionType) and len(type_.__args__) == 2 and type_.__args__[1] is type(None):
                    # Optional
                    type_ = type_.__args__[0]
                    # Would break other things.
                    Assert.not_custom(issubclass, type_, Config)
                Assert.custom(isinstance, type_, type)
                if issubclass(type_, Config):
                    type_.to_argparse(parser)
                    continue
                if type_ is bool:
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

    @classmethod
    def from_dict(
        cls,
        arg_dict: dict,
        *,
        format_: ConfigDictFormat = ConfigDictFormat.flat,
        strict: bool = True,
        strict_cls: bool = False,
    ):
        """
        Create a `Config` from a config dict (see `to_dict`).

        Args:
            arg_dict: A config dict. Will be modified in-place.
            format_: The config format used to represent nested configs. See `to_dict`
            strict: Ensure that all fields are used.
            strict_cls: Ignore the `__class__` field and use the default values. Applied recursively to nested configs.
        """
        cls.check_abstract()
        out_arg_dict = {}

        if format_ == ConfigDictFormat.tuple:
            arg_dict = cls.update_config_dict({}, arg_dict)
            format_ = ConfigDictFormat.nested

        if format_ == ConfigDictFormat.nested:
            arg_dict_cls = arg_dict.pop("__class__", cls)
            if not strict_cls:
                if isinstance(arg_dict_cls, str):
                    if arg_dict_cls != cls.__name__:
                        # TODO: Support serialized class names?
                        raise NotImplementedError()
                else:
                    Assert.custom(issubclass, arg_dict_cls, cls)
                    cls = arg_dict_cls  # noqa
        for name, field in cls.fields():
            if not field.init or field._field_type == dataclasses._FIELD_CLASSVAR:  # noqa
                continue
            if isinstance(field.type, type) and issubclass(field.type, Config):
                # Do not validate yet in case the root class sets cross-dependencies in validation.
                with NoAutoValidate():
                    if format_ == ConfigDictFormat.flat:
                        out_arg_dict[name] = field.type.from_dict(
                            arg_dict, format_=format_, strict=False, strict_cls=strict_cls
                        )
                    else:
                        out_arg_dict[name] = field.type.from_dict(
                            arg_dict.pop(name, {}), format_=format_, strict=strict, strict_cls=strict_cls
                        )
            elif name in arg_dict:
                out_arg_dict[name] = arg_dict.pop(name)
        if strict and arg_dict:
            raise ValueError(cls, list(arg_dict))
        out = cls(**out_arg_dict)  # noqa
        if _AUTO_VALIDATE:
            out.validate()
        return out

    @classmethod
    def from_namespace(cls, value: argparse.Namespace, strict: bool = True):
        """
        Create a `Config` from a flat argparse namespace.

        Args:
            value: A config namespace.
            strict: Ensure that all fields are used.
        """
        return cls.from_dict(value.__dict__.copy(), strict=strict)

    @classmethod
    def from_other(
        cls,
        other: "Config",
        updates: dict[str | tuple[str, ...], typing.Any] | None = None,
        strict: bool = True,
        strict_cls: bool = False,
    ):
        """
        Create a `Config` from another one, copying and possibly updating the fields.

        Args:
            other: The config to copy from. May have a different class.
            updates: Change the value of some fields.
            strict: Ensure that all fields are used (may need to disable if the class is different).
            strict_cls: Ignore the `__class__` field and use the default values. Applied recursively to nested configs.
        """
        config_dict = other.to_dict(format_=ConfigDictFormat.nested)
        cls.update_config_dict(config_dict, updates)
        return cls.from_dict(config_dict, format_=ConfigDictFormat.nested, strict=strict, strict_cls=strict_cls)

    @classmethod
    def update_config_dict(cls, config: dict, updates: dict[str | tuple[str, ...], typing.Any] | None = None):
        if updates is not None:
            for keys, value in updates.items():
                if isinstance(keys, str):
                    config[keys] = value
                else:
                    dict_to_update = config
                    for key in keys[:-1]:
                        if key not in dict_to_update:
                            dict_to_update[key] = {}
                        dict_to_update = dict_to_update[key]
                    dict_to_update[keys[-1]] = value
        return config

    @classmethod
    def get_parser(cls):
        """
        Make an argument parser for the config and its sub-configs.
        The whole config hierarchy is flattened (see `to_dict`),
        and the user is responsible for preventing name clashes.
        """
        parser = argparse.ArgumentParser()
        cls.to_argparse(parser)
        return parser

    @classmethod
    def from_url(cls, config_url: str, config_auth_token_file: None | str = None):
        """
        Read a config from a URL, typically a config file hosted on GitHub.
        """

        headers = {"Accept": "application/vnd.github.v3.raw"}
        if config_auth_token_file:
            with open(config_auth_token_file) as f:
                headers["Authorization"] = f"token {f.read().strip()}"
        response = requests.get(config_url, headers=headers)
        if response.status_code == 200:
            arg_dict = yaml.safe_load(response.text)
            return cls.from_dict(arg_dict, format_=ConfigDictFormat.nested, strict=True, strict_cls=True)
        else:
            if isinstance(response.reason, bytes):
                try:
                    reason = response.reason.decode("utf-8")
                except UnicodeDecodeError:
                    reason = response.reason.decode("iso-8859-1")
            else:
                reason = response.reason
            raise ValueError(f"Failed to fetch config from {config_url}: {response.status_code}, {reason}")

    @classmethod
    def from_args(cls, args: list[str] | None = None):
        """
        Make an argument parser for the config and its sub-configs,
        parse the provided args (or `sys.argv`),
        and create a config from the resulting namespace.
        """
        cls.check_abstract()
        initial_parser = argparse.ArgumentParser(add_help=False)
        initial_parser.add_argument("--config_url")
        initial_parser.add_argument("--config_auth_token_file")
        initial_args, remaining_args = initial_parser.parse_known_args(args)
        if initial_args.config_url:
            Assert.empty(remaining_args)
            return cls.from_url(initial_args.config_url, initial_args.config_auth_token_file)
        else:
            return cls.from_namespace(cls.get_parser().parse_args(remaining_args))

    def show(self, all_fields=False, log_fn=logger.info):
        """
        Print all config and sub-config arguments in alphabetical order, following the Megatron-LM format.
        TODO: Avoid flattening.

        Args:
            all_fields: Include the derived fields, with `init=False`.
            log_fn: The logging function to use.
        """
        args = self.to_dict(all_fields=all_fields)
        log_fn(
            "------------------------ arguments ------------------------"
            + "".join([f"\n  {arg} {'.' * (48 - len(arg))} {args[arg]}" for arg in sorted(args)])
            + "\n-------------------- end of arguments ---------------------"
        )

    @classmethod
    def check_abstract(cls):
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
