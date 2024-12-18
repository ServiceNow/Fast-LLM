---
title: Style Guide
---

This section collects general coding style guidelines used in Fast-LLM.
Following these will ensure a swift reviewing process and will help maintain consistency and readability.
Note that while we try to enforce these principles,
exceptions may be allowed on a case-by-case basis, for example if they noticeably improve readability.

As a general principle, **Fast-LLM prioritizes code readability and maintainability** over conciseness,
coding speed or individual programmer's preferences.
Most of the style choices below are based on this principle.

## 🎯 Basic Style

Unless otherwise specified, Fast-LLM follows the [PEP 8](https://peps.python.org/pep-0008/) coding style.
This style (and many other conventions) is enforced with automatic formatting through a [pre-commit](https://pre-commit.com/) git hook.

Make sure these git hooks are installed by running

```bash
pip install pre-commit
pre-commit install
```

!!! note "More on automated formatting"

    Fast-LLM's automated formatting includes [Black](https://black.readthedocs.io/en/stable/),
    [isort](https://pycqa.github.io/isort/), [autoflake](https://github.com/PyCQA/autoflake), and a few other packages.
    See Fast-LLM's [pre-commit configuration](https://github.com/ServiceNow/Fast-LLM/blob/main/.pre-commit-config.yaml) for more details.

## 📚 Naming Conventions

Names should be as self-explanatory as possible, within reason.
This includes python identifiers (classes, variables, methods, modules, etc.), file names and configuration parameters.
For example:

* Use meaningful, self-descriptive identifier names (ex. `x -> loss`). This gives other users a better chance to understand the code.
Abstract variable names such as `x` are however OK for generic methods where more descriptive names aren't appropriate (ex. `add(x, y)`).
* Avoid abbreviations, especially domain-specific ones. Ex. `bs -> batch_size`.
This gives everyone a chance to understand the code, regardless of their prior knowledge.
* Avoid redundancies especially for configuration parameters, ex. `data.data_type` -> `data.type`.
* Avoid name parts that refer to the data type, ex. `num`. Use type hints instead.

Note that these conventions are enforced more strictly on user-facing names since they are more difficult to change,
for example configuration parameters and the public interface of core classes and modules.

## 🛬 Imports

We use the following conventions for imports (other than those enforced by isort):

* Import standard library and third party modules by module (ex. `import package.module`, not `from package.module import method`).
In addition to keeping the code consistent, this keeps identifier's origin explicit so anyone can tell where it came from with just a quick glance at the code. This is especially useful for identifiers that with otherwise ambiguous source (ex. `float32` may come from torch, numpy, triton, etc.; Fast-LLM's configuration scheme has many identifiers in common with `dataclasses`, `omegaconf` and `pydantic`)
* Avoid renaming with `as`, except for some (arbitrarily chosen) common ones: `numpy as np`, `triton.language as tl`.
* Import first-party modules through specific identifiers (ex. `from fast_llm.module import method`, not `import fast_llm.module`). This keeps Fast-LLM identifiers to a manageable length and makes it easier to track what is used in a given file.
* Always use absolute imports (ex. no `from .module import method`)
* Include all explicitly-imported third-party module to `setup.cfg`.
Only add new requirements if they provide a substantial benefit,
as we try to keep the requirements to a minimum.
* Prefer file-level imports over imports inside methods, unless they significantly slow down the import process
or concern an optional dependency that should not be absolutely required to import the module (ex. `transformers`).
If an offending import is only required for a type hint, include it in a `if typing.TYPE_CHECKING:` block.

!!! warning "Configuration modules"

    Fast-LLM supports instantiation and validation of configurations with a barebone installation.
    Because of this, modules that contain configuration classes (usually named `config.py`)
    should not include any top-level third-party import (except for those installed in the [barebone install](https://github.com/ServiceNow/Fast-LLM/blob/main/setup.cfg)),
    and the same applies for configuration initialization and validation methods.
    Third-party import may be included in other methods if needed.

## 🔓 Public and Private Interface

Although good practices of object-oriented programming are generally ignored in python,
Fast-LLM attempts to follow them to an extent, while avoiding unnecessary bloat:

* Mark private and protected variables with an underscore `_` prefix.
As is customary in python, we make no distinction between the two and avoid the double-underscore `__` notation.
* Keep public interfaces (methods and variables without underscore prefix) as lean as possible,
i.e. mark everything as private/protected unless there is a clear need to make it public.
We can always add to the public interface later, but removing from it is difficult.
* Use accessors sparingly through the `@property` decorator or equivalent,
usually to define read-only public variables.

## 💡 Type Hints

Fast-LLM uses type hints for several reasons, including code readability, type checking in IDEs,
and type validation for configurations:

* Always use type hints for the public interface of a classes and modules.
Type hints for method outputs may be omitted if they can be easily inferred.
* Prefer using type hints in private interfaces, especially if it improves readability and/or static type checking.
* Use newer type hint formats when possible, ex. `typing.List -> list`, `typing.Union(A,B) -> A | B`.

## 🗑️ Misc

* Please add descriptions and comments as needed, especially for parts that would otherwise be difficult to understand.
* Use `pathlib` rather than `os.path`.
* We encourage the use of modern python features when beneficial, up to the minimum python version (3.12).
