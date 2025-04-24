---
title: Configuration Reference
---

Robust configuration is a fundamental aspect of Fast-LLM:
the library comes with hundreds of options that can be specified to set up an experiment exactly how you want it.
These options are all handled by Fast-LLM's configuration system.
This section explains this configuration system from a developer's perspective,
and shows how it can be extended with new configuration parameters.

TODO: Belongs elsewhere? Mixing purposes, benefits and principles?
Fast-LLM's configuration scheme serves multiple purposes, most importantly:

* **Modularity**: Sort out the various parameters into a hierarchy of modular components, or *configs*.
* **Parsing**: Parse yaml files, cli arguments and python dictionaries into such configurations.
* **Resolution**: Set sensible defaults, including ones that may depend on other fields, in the same config or a different one in the hierarchy.
* **Validation**: Verify that inputs have an appropriate type, and adjust automatically for compatible (ex. `int` -> `float`) or non-serializable types (ex. `str` -> `enum.StrEnum`). Enforce constrains between config parameters, whether they lie in the same config or a different one in the hierarchy.
* **Immutability**: Configs are protected from accidental changes after resolution.
* **Serialization**: Allow serialization and deserialization of any config, independently of its place in the hierarchy. Keep the number of serialized fields to a minimum.
* **Derivation**: Allow for derived config fields, i.e. values that are not provided by the user but rather computed from others.
* **Functionality**: Let configs be full-fledged classes, supporting any kind of derived functionality.
* **Portability**: Configs can be created and validated with a minimalistic installation of Fast-LLM, (almost) without loading any of its dependencies. This allows resolving and validating configs from anywhere, ex. locally before launching an experiment on a remote server.

## Prerequisites

Before reading this guide, we expect you to be familiar with:

* **Dataclasses**: Fast-LLM's config system is an extension of Python's `dataclasses` module,
    and much of its functionality is borrowed from it.
* **Type hints**: Configuration are strongly typed, so knowledge of type hints is essential.

!!! note
    Fast-LLM generally follows the same principles as existing Python-based configuration mechanisms
    such as `omegaconf`, `hydra` and `pydantic`, so much of Fast-LLM's configuration system may already look familiar.
    However, note that Fast-LLM differs from those in several aspects, most importantly in the config validation system.

## Basic usage

### Declare a config class with fields

Fast-LLM configs are an extension of dataclasses, but are declared in a slightly different way:

1. Config classes use Fast-LLM's `@config_class` decorator (rather than `@dataclasses.dataclass`)
2. Config classes inherit from Fast-LLM's `Config` base class, either directly or through another config class.
3. Fields are always typed and described with Fast-LLM's `Field` class (see below).

Here is an example describing a simple training interval,
i.e., some feature happening once every `interval` training iterations.

```python
from fast_llm.config import config_class, Config, Field, FieldHint, test_field

@config_class()
class IntervalConfig(Config):
    interval: int | None = Field(
        default=None,
        desc="The number of training iterations between each interval. Setting to None will disable.",
        hint=FieldHint.core,
        valid=test_field(lambda x: x is None or x>=0),
    )
    offset: int = Field(
        default=0,
        desc="Offset for the first interval.",
        hint=FieldHint.feature,
        valid=test_field(lambda x: x>=0),
    )
```

#### Describe a field

Fast-LLM's `Field` are similar to `dataclasses.field`
and support attributes such as `default`, `default_factory` and `init`,
but comes with additional ones:

**desc** (required) and **doc** (optional) describe the field and its usage to the user and other developers.
They are also used to generate documentation for the config class.
**desc** Should contain a short description (max 2-3 lines),
and may be supplemented with **doc** for fields that deserve a more lengthy documentation.

!!! tip "Best practices"
    Keep in mind that config fields are user-facing,
    and that the typical user may not be familiar with the feature you are implementing.
    Therefore, it is important to use clear and accessible field names and descriptions,
    preferably free of abbreviations or technical jargon.
    Don't hesitate to add references and/or links to tutorials!
    Also note that field names are difficult to change due to the need for backward compatibility.

**hint** (required) helps sort out the fields in large config classes by importance,
i.e., it helps distinguish commonly used fields from the more obscure ones.
Most fields are labelled as `core` (essential to the config class),
`optional` (important but may be omitted) or `feature` (only relevant when using an optional feature).

### Config validation

### Derived fields

### Custom functionality

## Nested and derived config classes

## Configurable classes

## Runnable classes

## Advanced validation

## Backward compatibility

## Reference

### Field types

Fast-LLM supports most common python data types:

* Basic python types (`int`, `float`, `bool`, `None`, `str`). Note that `int` values are accepted as `float`.
* Dictionaries (`dict[KeyType, ValueType]`). These may be arbitrarily nested, i.e. `ValueType` may be of any supported type.
* Other config classes. These are specified and serialized as dictionaries, and similarly may be arbitrarily nested.
* Arrays (`tuple[ValueType]`, `list[ValueType]`, `set[ValueType]`). `ValueType` may be of any supported type, but nested arrays (ex. containing dictionaries, configs or other arrays) are not recommended as they are more difficult to configure.
* Enums: most enums types are supported, though we recommend `enum.StrEnum` as a base class for simpler usage and meaningful serialized values. Enums are specified and serialized as their value.
* Paths (`pathlib.Path`). These are specified and serialized as strings.
* Unions (`TypeA | TypeB`). Unions of any supported type are allowed (including `None` for optional fields), though other configs are not recommended. Note that resolution stops at the first match, ex. a `float|int` field will systematically recognize integers as floats and convert them, but `int|float` will keep them as-is.
* Types (`type[BaseType]`). Allowed, but serialization is not supported.
* Any (`typing.Any`). Technically supported, but specification and serialization is restricted to yaml-serializable type. Not recommended.

!!! warning
    Older type hints from the `typing` module (ex. `typing.List`, `typing.Union` and `typing.Optional`) are not supported.
    Use the newer syntax instead.

### Type hints

The full list of type hints is as follows:

* `core`: A core configuration parameter that is expected to always be provided explicitly.
* `optional`: An optional parameter that may be ignored as the default tends to be good enough.
* `feature`: An parameter related to an optional feature, that should only be defined if that feature is enabled.
* `performance`: An optional parameter related to computational performance.
* `stability`: An optional parameter related to numerical precision and computational stability.
* `expert`: An advanced parameter that needs some additional expertise to be handled.
* `unknown`: No hint has been provided for this parameter.
* `logging`: An optional parameter related to logging or debug logs
* `testing`: A rarely defined parameter that is only meant for testing and debugging.
* `derived`: A parameter that is typically calculated from others.
* `setup`: An external parameter that must be provided in `setup` after initialization.
* `deprecated`: The feature is deprecated and may be removed renamed or replaced soon.
* `wip`: The parameter is not fully implemented yet.

**valid** (optional) may be used to modify (ex. standardize) or check a condition on the input value.
It has signature `typing.Callable[[typing.Any], typing.Any] | None`,
i.e. it takes one argument, the input field, and returns the possibly modified field.
If the fields is invalid (i.e., the test condition fails), it should raise an `Exception`.
In most cases `valid` only verifies a condition, which can be simplified with a wrapper (as in the example above):

* `check_field`: Wraps a condition with no return value, and return the unmodified field.
* `test_field`: Wraps a condition that returns a boolean indicating whether the field is valid.
