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

Before reading this guide, we espect you to be familiar with:

* **Dataclasses**: Fast-LLM's config system is an extension of Python's `dataclasses` module,
    and much of its functionality is borrowed from it.
* **Type hints**: Configuration are strongly typed, so knowledge of type hints is essential.

!!! note
    Fast-LLM generally follows the same principles as existing Python-based configuration mechanisms
    such as `omegaconf`, `hydra` and `pydantic`, so much of Fast-LLM's configuration system may already look familiar.
    However, note that Fast-LLM differs from those in several aspects, most importantly in the config validation system.

## Basic usage

```python
import pathlib
from fast_llm.config import config_class, Config, Field, FieldHint

@config_class()
class TokenizerConfig(Config):

    name: str = Field(
        default="Tokenizer",
        desc="A name for the tokenizer.",
        hint=FieldHint.deprecated,
    )
    path: pathlib.Path | None = Field(
        default=None,
        desc="Path to the tokenizer file.",
        hint=FieldHint.core,
    )
```
