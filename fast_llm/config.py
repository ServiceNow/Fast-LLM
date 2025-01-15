import dataclasses
import enum
import logging
import pathlib
import traceback
import types
import typing
import warnings

import yaml

from fast_llm.utils import Assert, Tag, get_type_name, header, log, pop_nested_dict_value, set_nested_dict_value

logger = logging.getLogger(__name__)


_AUTO_VALIDATE = True

MISSING = Tag("<MISSING>")
DEFAULT = Tag("<DEFAULT>")


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


class _ConfigDictFormat(str, enum.Enum):
    # TODO v0.3: delete class
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


class FieldUpdate(dict):
    """
    Specify some entries in the field that should be updated from the base class.
    Useful for changing the default or description in a derived class.
    Processed in `__init_subclass__`.
    """


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


class NestedValidationError(ValidationError):
    pass


class FieldTypeError(ValueError):
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

    # We can't use @config_class on this one because it needs this class to be defined, so we assume this one is OK.
    __class_validated__: typing.ClassVar[bool] = True
    _abstract: typing.ClassVar[bool] = False
    _validated: bool = Field(init=False, repr=False)
    _unknown_fields: dict[str, typing.Any] = Field(init=False, repr=False)

    def __post_init__(self):
        """
        Perform validation unless prevented with `NoAutoValidate`.
        In general this should not be overridden in derived classes,
        and all post-processing should be done in `_validate`
        """
        self._validated = False
        if _AUTO_VALIDATE:
            self.validate()

    def __setattr__(self, key: str, value: typing.Any) -> None:
        """
        Make the class read-only after validation.
        """
        # `_validated` may not be set yet.
        if getattr(self, "_validated", False):
            if value is getattr(self, key):
                # Allow setting the exact same object to facilitate setup of cross-dependencies.
                # Ex. allow re-setting cross-dependencies of already validated sub-configs.
                return
            raise RuntimeError(
                f"Cannot set attribute `{key}`"
                f" in configuration class `{get_type_name(type(self))}` after validation."
            )
        super().__setattr__(key, value)

    def __delattr__(self, key: str) -> None:
        """
        Make the class read-only after validation.
        """
        if getattr(self, "_validated", False):
            raise RuntimeError(
                f"Cannot delete attribute `{key}`"
                f" in configuration class `{get_type_name(type(self))}` after validation."
            )
        super().__delattr__(key)

    def validate[T](self: T, *, _is_validating: bool = False) -> T:
        """
        Validate a class and mark it as read-only
        This should not be overridden in derived classes.
        """
        if not self._validated:
            try:
                self._validate()
            except (ValidationError, FieldTypeError) as e:
                if _is_validating:
                    raise
                else:
                    raise type(e)("\n".join(e.args)) from None
            self._validated = True
        return self

    def _validate(self) -> None:
        """
        Verify that the type hints are respected,
        and fix some know entries compatible with the type hint (ex. `int -> float`, `str -> pathlib.Path`)

        Can be extended to add custom post-processing (typically before the super() call)
        and validation (typically after)
        """
        self._check_abstract()
        errors = []
        for name, field in self.fields():
            if not field.init or field._field_type == dataclasses._FIELD_CLASSVAR:  # noqa
                continue
            value = getattr(self, name)
            if value is DEFAULT:
                # Replace the value with its default.
                # We still need to validate because some fields have invalid defaults.
                value = field.default
            new_value = self._validate_nested(value, field.type, field.name, field.valid, errors, False)
            setattr(self, name, new_value)
        for name in getattr(self, "_unknown_fields", {}):
            errors.append(f"Unknown field `{name}` in class {self._get_class_name()}")
        if errors:
            # TODO: Option to show traceback for errors.
            raise NestedValidationError(*errors)

    @classmethod
    def _validate_nested(cls, value, type_, name: str, valid_fn: typing.Optional[typing.Callable], errors, nested):
        try:
            value = value if valid_fn is None else valid_fn(value)
            value = cls._validate_element(value, type_, name)
        except FieldTypeError as e:
            # There is a problem with the config class itself, no point in continuing.
            raise FieldTypeError(
                f"Invalid field type `{get_type_name(type_)}` in class {cls._get_class_name()}:",
                *["  " + arg for arg in e.args],
            )
        except ValidationError as e:
            # This is a known error, `e.args` should have all the required information.
            message = f"Validation failed for field `{name}`" + (
                ":" if nested else f" of type `{get_type_name(type_)}` in class {cls._get_class_name()}:"
            )
            if len(e.args) > 1 or isinstance(e, NestedValidationError):
                errors.extend([message] + ["  " + arg for arg in e.args])
            else:
                # No need to have the error description on a separate line.
                errors.append(f"{message} {e.args[0]}")
        except Exception as e:
            # This is an unknown error, so we need to provide the stack trace so the user can tell what the problem is.
            errors.append(
                f"Validation failed for field `{name}` in class {cls._get_class_name()}: {', '.join(e.args)}"
                f"\n\n====================== stack trace ========================\n"
                + traceback.format_exc()
                + "===========================================================\n"
            )
        return value

    @classmethod
    def _validate_element(cls, value, type_, name: str):
        if type_ is typing.Any:
            # TODO: Check if x is or contains a config?
            pass
        elif type_ is types.NoneType:
            if value == "":
                value = None
            if value is not None:
                raise ValidationError(f"Unexpected type `{get_type_name(type(value))}`")
        elif isinstance(type_, types.UnionType):
            # Takes care of Optional too
            value = cls._validate_union(value, type_, name)
        elif hasattr(type_, "__origin__"):
            # TODO: Improve error messages for nested entries.
            origin = type_.__origin__
            if origin in (list, set, tuple):
                value = cls._validate_array(value, type_, name)
            elif issubclass(origin, dict):
                value = cls._validate_dict(value, type_, name)
            elif origin is type:
                cls._validate_type(value, type_, name)
            else:
                raise FieldTypeError(f"Unsupported __origin__ `{origin}`")
        elif not isinstance(type_, type):
            raise FieldTypeError(f"Not a type.")
        elif issubclass(type_, Config):
            cls._validate_element_type(value, type_, name)
            value.validate(_is_validating=True)
        else:
            value = cls._validate_simple(value, type_, name)
        return value

    @classmethod
    def _validate_union(cls, value, type_, name: str):
        errors = []
        for subtype in type_.__args__:
            errors_ = []
            x_ = cls._validate_nested(value, subtype, f"{name}[{get_type_name(subtype)}]", None, errors_, True)
            if errors_:
                errors.extend(errors_)
            else:
                # Only need one valid subtype, we return the first one.
                return x_
        # If none of the subtype works, we provide information for all of them.
        raise ValidationError(*errors)

    @classmethod
    def _validate_array(cls, value, type_, name: str):
        origin = type_.__origin__
        cls._validate_element_type(value, (origin, list, tuple), name)
        args = getattr(type_, "__args__", [typing.Any, ...] if origin is tuple else [typing.Any])
        errors = []
        if issubclass(origin, tuple) and not (len(args) == 2 and args[1] is ...):
            if len(value) != len(args):
                raise ValidationError(f"Invalid length {len(value)} (expected {len(args)})")
            new_value = origin(
                cls._validate_nested(value_, arg, f"{name}[{i}]", None, errors, True)
                for i, (value_, arg) in enumerate(zip(value, args))
            )
        else:
            if not issubclass(origin, tuple) and len(args) != 1:
                FieldTypeError(f"Invalid array specification")
            new_value = origin(
                cls._validate_nested(value_, args[0], f"{name}[{i}]", None, errors, True)
                for i, value_ in enumerate(value)
            )
        if errors:
            raise ValidationError(*errors)
        return new_value

    @classmethod
    def _validate_dict(cls, value, type_, name: str):
        args = list(getattr(type_, "__args__", []))
        if len(args) > 2:
            raise FieldTypeError(f"Invalid dict specification `{get_type_name(type_)}` for field `{name}`")
        args.extend([typing.Any for _ in range(2 - len(args))])
        cls._validate_element_type(value, type_.__origin__, name)
        errors = []
        new_value = {}
        old_keys = {}
        for key, value_ in value.items():
            new_key = cls._validate_nested(key, args[0], f"{name}(key {key})", None, errors, True)
            new_value_ = cls._validate_nested(value_, args[1], f"{name}[{key}]", None, errors, True)
            if key in new_value:
                errors.append(f"Duplicate key `{new_key}` after validation (from `{old_keys[new_key]}`, `{key}`)")
            old_keys[new_key] = key
            new_value[new_key] = new_value_
        if errors:
            raise ValidationError(*errors)
        return new_value

    @classmethod
    def _validate_simple(cls, value, type_, name: str):
        if hasattr(type_, "__fast_llm_validator__"):
            value = type_.__fast_llm_validator__(value)
        elif type_ is float and isinstance(value, int):
            # Ints are ok too.
            value = float(value)
        elif issubclass(type_, enum.Enum) and not isinstance(value, type_) and issubclass(type_, type(value)):
            # Enum values are ok too.
            value = type_(value)
        elif issubclass(type_, pathlib.PurePath) and isinstance(value, str):
            # Str paths are ok too.
            value = type_(value)
        cls._validate_element_type(value, type_, name)
        return value

    @classmethod
    def _validate_type(cls, value, type_: type | tuple[type, ...], name):
        args = list(getattr(type_, "__args__", []))
        if len(args) != 1:
            raise FieldTypeError(f"Invalid type specification `{get_type_name(type_)}` for field `{name}`")
        if not isinstance(value, type):
            raise ValidationError(f"Unexpected type `{get_type_name(type(value))}`")
        if not issubclass(value, args[0]):
            raise ValidationError(f"Field value `{value} is not a subclass of `{get_type_name(type_)}`")

    @classmethod
    def _validate_element_type(cls, value, type_: type | tuple[type, ...], name):
        if not isinstance(value, type_):
            raise ValidationError(f"Unexpected type `{get_type_name(type(value))}`")

    @classmethod
    def fields(cls) -> typing.Iterable[tuple[str, Field]]:
        """
        An iterable for the field definitions of a `Config` class.
        """
        return cls.__dataclass_fields__.items()  # noqa

    @classmethod
    def get_field(cls, name: str) -> Field:
        return cls.__dataclass_fields__[name]  # noqa

    def _to_dict(
        self,
        verbose: int | None = None,
        all_fields: bool = False,
        format_: _ConfigDictFormat = _ConfigDictFormat.nested,
        serializable: bool = False,
    ) -> dict[str, typing.Any]:
        """
        Serialize the config to a dict that can (generally) be used to reconstruct an identical `Config`.
        When not flat, the dict includes a `__class__` entry which allows support for derived classes.

        Args:
            all_fields: Include the derived fields, with `init=False`.
            format_: The config format used to represent nested configs. Options:
              * `ConfigDictFormat.nested`: Preserve the nested config structure by returning nested dicts.
                Also save a `__class__` entry to support derived classes. Standard format.
              * `ConfigDictFormat.tuple`: Preserve the nested config structure by returning tuples of keys.
                Used for config updates.
            serializable: Ensure the dict is serializable to json or yaml. Information may be lost.
        """
        arg_dict = {}
        for name, field in self.fields():
            value = getattr(self, name, MISSING)
            self._add_field_to_args(arg_dict, name, field, value, verbose, all_fields, format_, serializable)
        if hasattr(self, "_unknown_fields"):
            for name, value in self._unknown_fields.items():
                self._add_field_to_args(arg_dict, f"!!! {name}", None, value, None, all_fields, format_, serializable)

        return arg_dict

    @classmethod
    def _add_field_to_args(
        cls,
        args: dict | list,
        name: str | None,
        field: Field | None,
        value: typing.Any,
        verbose: int | None = None,
        all_fields: bool = False,
        format_: _ConfigDictFormat = _ConfigDictFormat.nested,
        serializable: bool = False,
    ) -> None:
        if (
            field is not None
            and (not field.init or field._field_type == dataclasses._FIELD_CLASSVAR)
            and not (all_fields)
        ):
            # Exclude class variables and derived fields unless requested explicitly.
            return
        elif isinstance(value, Config):
            field_value = value._to_dict(
                verbose=verbose,
                all_fields=all_fields,
                format_=format_,
                serializable=serializable,
            )
        elif isinstance(value, (list, tuple, set)):
            field_value = {} if format_ == _ConfigDictFormat.tuple else []
            for i, list_value in enumerate(value):
                cls._add_field_to_args(
                    field_value, str(i), None, list_value, verbose, all_fields, format_, serializable
                )
        elif isinstance(value, dict):
            field_value = {}
            for dict_name, dict_value in value.items():
                cls._add_field_to_args(
                    field_value, dict_name, None, dict_value, verbose, all_fields, format_, serializable
                )
        elif (
            verbose is not None
            and field is not None
            and FieldHintImportance[field.hint] > verbose
            and value == field.default
        ):
            # Exclude unimportant default values.
            return
        else:
            field_value = value
            if serializable:
                field_value = cls._serialize_value(value)
            if format_ == _ConfigDictFormat.tuple:
                field_value = {(): field_value}

        if serializable:
            name = cls._serialize_value(name)
        if format_ == _ConfigDictFormat.tuple:
            args.update({(name,) + name_: value_ for name_, value_ in field_value.items()})
        elif format_ == _ConfigDictFormat.nested:
            if not isinstance(field_value, (dict, list)) or len(field_value) > 0 or all_fields:
                if isinstance(args, dict):
                    args[name] = field_value
                else:
                    args.append(field_value)
        else:
            raise NotImplementedError(format_)

    @classmethod
    def _serialize_value(cls, value: typing.Any) -> int | float | bool | str | None:
        value = value
        if hasattr(value, "__fast_llm_serialize__"):
            value = value.__fast_llm_serialize__()
        if isinstance(value, enum.Enum):
            value = value.value
        # Tag is not actually serializable, but needs to be kept as-is for config processing,
        # and should be absent for valid configs.
        elif not isinstance(value, int | float | bool | str | Tag | None):
            value = str(value)
        return value

    def to_copy[
        T
    ](self: T, *updates: typing.Union["Config", dict[str | tuple[str, ...], typing.Any]], strict: bool = True,) -> T:
        return self.from_dict(self, *updates, strict=strict)

    def to_serialized(self, verbose: int | None = FieldVerboseLevel.core) -> dict[str, typing.Any]:
        return self._to_dict(verbose=verbose, format_=_ConfigDictFormat.nested, serializable=True)

    def to_logs[
        T
    ](
        self,
        verbose: int | None = FieldVerboseLevel.core,
        log_fn: typing.Callable[[str], T] = logger.info,
        title: str | None = None,
        width: int = 80,
        fill_char: str = "-",
    ) -> T:
        arg_dict = self.to_serialized(verbose=verbose)
        if title is None:
            title = self._get_class_name()
        return log_fn(
            f"\n{header(title, width, fill_char)}"
            f"\n{yaml.safe_dump(arg_dict, sort_keys=False)}"
            f"{header('end', width, fill_char)}"
        )

    @classmethod
    def _get_class_name(cls) -> str:
        return get_type_name(cls)

    @classmethod
    def from_dict(
        cls,
        default: typing.Union["Config", dict[str, typing.Any]],
        *updates: typing.Union["Config", dict[str | tuple[str, ...], typing.Any]],
        strict: bool = True,
    ) -> typing.Self:
        if isinstance(default, Config):
            default = default._to_dict()
        for update in updates:
            if isinstance(update, Config):
                update = update._to_dict(format_=_ConfigDictFormat.tuple)
            for keys, value in update.items():
                set_nested_dict_value(default, keys, value)

        return cls._from_dict(default, strict)

    @classmethod
    def from_flat_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
    ) -> typing.Self:
        # TODO v0.3: Remove flat format
        return cls._from_dict(default, strict, True)

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        # TODO v0.3: Remove flat format
        out_arg_dict = {}

        # TODO v0.3: Remove backward compatibility fix
        if "__class__" in default:
            del default["__class__"]

        # Do not validate yet in case the root class sets cross-dependencies in validation.
        with NoAutoValidate():
            for name, field in cls.fields():
                if not field.init or field._field_type == dataclasses._FIELD_CLASSVAR:  # noqa
                    continue
                if flat:
                    if isinstance(field.type, type) and issubclass(field.type, Config):
                        if flat:
                            out_arg_dict[name] = field.type._from_dict(default, False, True)
                        else:
                            out_arg_dict[name] = field.type._from_dict(default.pop(name, {}), strict)
                    elif name in default:
                        out_arg_dict[name] = default.pop(name)
                else:
                    # Check for nested configs to instantiate.
                    try:
                        value = cls._from_dict_nested(default.pop(name, MISSING), field.type, strict)
                        if value is not MISSING:
                            out_arg_dict[name] = value
                    except FieldTypeError as e:
                        raise FieldTypeError(
                            f"Invalid field type `{get_type_name(field.type)}` in class {cls._get_class_name()}: "
                            + ", ".join(e.args)
                        )
            out = cls(**out_arg_dict)  # noqa
            if strict and default:
                out._unknown_fields = default.copy()
        if _AUTO_VALIDATE:
            out.validate()
        return out

    @classmethod
    def _from_dict_nested(cls, value, type_, strict: bool):
        if type_ in (typing.Any, types.NoneType):
            pass
        elif isinstance(type_, types.UnionType):
            # Takes care of Optional too
            value = cls._from_dict_union(value, type_, strict)
        elif hasattr(type_, "__origin__"):
            # TODO: Improve error messages for nested entries.
            origin = type_.__origin__
            if origin in (list, set, tuple):
                value = cls._from_dict_array(value, type_, strict)
            elif issubclass(origin, dict):
                value = cls._from_dict_dict(value, type_, strict)
            elif origin is type:
                pass
            else:
                raise FieldTypeError(f"Unsupported __origin__ `{origin}`")
        elif not isinstance(type_, type):
            raise FieldTypeError(f"Not a type: {type_}.")
        elif issubclass(type_, Config):
            if value is MISSING:
                value = {}
            if isinstance(value, dict):
                value = type_._from_dict(value, strict)
        return value

    @classmethod
    def _from_dict_union(cls, value, type_, strict: bool):
        new_value = value
        for subtype in type_.__args__:
            new_value_ = cls._from_dict_nested(value, subtype, strict)
            if new_value_ is not value:
                if new_value is not value:
                    # Happens if the union contains more than one Config class (or dict)
                    raise FieldTypeError(f"Ambiguous config class in union type {get_type_name(type_)}")
                new_value = new_value_
        return new_value

    @classmethod
    def _from_dict_array(cls, value, type_, strict: bool):
        origin = type_.__origin__
        if not isinstance(value, (list, set, tuple)):
            # This case will be handled during validation.
            return value
        args = getattr(type_, "__args__", [typing.Any, ...] if origin is tuple else [typing.Any])
        if issubclass(origin, tuple) and not (len(args) == 2 and args[1] is ...):
            new_value = origin(
                cls._from_dict_nested(value_, arg, strict) for i, (value_, arg) in enumerate(zip(value, args))
            )
            if len(new_value) < len(value):
                # We keep this for validation.
                new_value += value[len(value) - len(new_value) :]
        else:
            if not issubclass(origin, tuple) and len(args) != 1:
                FieldTypeError(f"Invalid array specification")
            new_value = origin(cls._from_dict_nested(value_, args[0], strict) for i, value_ in enumerate(value))
        return new_value

    @classmethod
    def _from_dict_dict(cls, value, type_, strict: bool):
        args = list(getattr(type_, "__args__", []))
        if len(args) > 2:
            raise FieldTypeError(f"Invalid dict specification `{get_type_name(type_)}`")
        if not isinstance(value, dict):
            # This case will be handled during validation.
            return value
        args.extend([typing.Any for _ in range(2 - len(args))])
        # Keys can't include configs so we only recurse on values.
        return {key: cls._from_dict_nested(value_, args[1], strict) for key, value_ in value.items()}

    @classmethod
    def _handle_renamed_field(
        cls,
        default: dict[str, typing.Any],
        old_name: str | tuple[str, ...],
        new_name: str | tuple[str, ...],
        fn: typing.Callable | None = None,
    ) -> None:
        if old_name in default:
            warnings.warn(f"Field `{old_name}` is deprecated in class {get_type_name(cls)}, use `{new_name}` instead.")
            value = pop_nested_dict_value(default, old_name)
            if fn is not None:
                value = fn(value)
            set_nested_dict_value(default, new_name, value)

    def compare(self, other: "Config", log_fn: typing.Union[type[BaseException], typing.Callable] = ValueError):
        # TODO: Check classes?
        self_dict = self._to_dict(format_=_ConfigDictFormat.tuple, serializable=True)
        other_dict = other._to_dict(format_=_ConfigDictFormat.tuple, serializable=True)
        compare = {
            key: (self_dict.get(key, MISSING), other_dict.get(key, MISSING))
            for key in self_dict.keys() | other_dict.keys()
        }
        diff = {
            key: (self_value, other_value)
            for key, (self_value, other_value) in compare.items()
            if self_value != other_value
        }
        if diff:
            log(
                f"Config diff:\n  "
                + "\n  ".join(
                    f"{'.'.join(key)}`: `{self_value}` != `{other_value}`"
                    for key, (self_value, other_value) in diff.items()
                ),
                log_fn=log_fn,
            )

    @classmethod
    def _check_abstract(cls) -> None:
        if cls._abstract:
            raise ValidationError(f"{cls.__name__} is abstract")
        if not cls.__class_validated__:
            raise ValidationError(
                f"{cls.__name__} hasn't been validated. Make sure to use the @config_class decorator."
            )

    def __init_subclass__(cls):
        """
        We need to postpone validation until the class has been processed by the dataclass wrapper.
        """
        for base_class in cls.__mro__:
            if issubclass(base_class, Config):
                assert cls.__class_validated__, (
                    f"Parent class {get_type_name(base_class)} of config class {get_type_name(cls)} has not been validated."
                    f" Make sure to use the @config_class decorator."
                )
        cls.__class_validated__ = False
        for name in list(cls.__dict__):
            value = getattr(cls, name)
            if isinstance(value, FieldUpdate):
                # In case of multiple inheritance, the base class field may not appear in `cls.__dataclass_fields__`.
                # so we iterate over superclasses following mro and use the first match.
                base_class_field = None
                for base_class in cls.__mro__:
                    base_class_fields = getattr(base_class, "__dataclass_fields__", {})
                    if name in base_class_fields:
                        base_class_field = base_class_fields[name]
                        break
                if base_class_field is None:
                    raise RuntimeError(f"Trying to update the non-existent field {name} in class {get_type_name(cls)}")
                setattr(
                    cls,
                    name,
                    Field(
                        desc=value.pop("desc", base_class_field.desc),
                        doc=value.pop("doc", base_class_field.doc),
                        hint=value.pop("hint", base_class_field.hint),
                        valid=value.pop("valid", base_class_field.valid),
                        default=value.pop("default", base_class_field.default),
                        default_factory=value.pop("default_factory", base_class_field.default_factory),
                        repr=value.pop("repr", base_class_field.repr),
                        hash=value.pop("hash", base_class_field.hash),
                        compare=value.pop("compare", base_class_field.compare),
                        metadata=value.pop("metadata", base_class_field.metadata),
                        kw_only=value.pop("kw_only", base_class_field.kw_only),
                    ),
                )
                if name in cls.__annotations__:
                    # TODO: Generalize to other type hints.
                    if isinstance(cls.__annotations__[name], type) and isinstance(base_class_field.type, type):
                        Assert.custom(issubclass, cls.__annotations__[name], base_class_field.type)
                else:
                    # dataclasses expects an annotation, so we use the one from the base class.
                    cls.__annotations__[name] = base_class_field.type


class Configurable[ConfigType: Config]:
    config_class: typing.ClassVar[type[Config]] = Config

    def __init__(self, config: ConfigType, *args, **kwargs):
        Assert.custom(isinstance, config, self.config_class)
        self._config = config
        # Handle multiple inheritance.
        super().__init__(*args, **kwargs)

    @property
    def config(self) -> ConfigType:
        return self._config
