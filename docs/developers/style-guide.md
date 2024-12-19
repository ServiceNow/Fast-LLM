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

Please make sure these git hooks are installed by running

```bash
pip install pre-commit
pre-commit install
```

!!! note "More on automated formatting"
    Fast-LLM's automated formatting includes [Black](https://black.readthedocs.io/en/stable/),
    [isort](https://pycqa.github.io/isort/), [autoflake](https://github.com/PyCQA/autoflake), and a few other packages.
    See Fast-LLM's [pre-commit configuration](https://github.com/ServiceNow/Fast-LLM/blob/main/.pre-commit-config.yaml) for more details.

## 📚 Naming Conventions

In addition to PEP 8, we use the following naming conventions for python identifiers (classes, variables, methods, modules, etc.),
file names and configuration parameters.
For example:

*   Use meaningful, self-descriptive identifier names (ex. `x -> loss`).
Abstract variable names such as `x` are however OK for generic methods where more descriptive names aren't appropriate (ex. `add(x, y)`).
*   Please avoid abbreviations, especially domain-specific ones.
This gives everyone a chance to understand the code, regardless of their prior knowledge. Ex. `bs -> batch_size`.
*   Try to keep names concise, for example by eliminating redundancies
and avoiding data type qualifiers such as `num` (covered by the type hint).
This is especially important for configuration parameters as the fully qualified names can get very long.
For example, `transformer.num_transformers_heads` can be simplified to `transformer.heads` without sacrificing clarity.

Note that these conventions are especially important on user-facing names which are more difficult to change,
for example configuration parameters and the public interface of core classes and modules.

!!! note "Why this matters"
    Using explicit, self-explanatory names gives other users a better chance to understand the code,
    regardless of their prior knowledge, which facilitates collaboration and maintenance.
    Our conventions follow this principle, while attempting to avoid excessively long names.

## 🛬 Imports

We use the following conventions for imports (other than those enforced by isort):

*   Import standard library and third party modules by module (ex. `import package.module`, not `from package.module import method`).
In addition to keeping the code consistent, this keeps identifier's origin explicit so anyone can tell where it came from with just a quick glance at the code.
*   Avoid renaming with `as`, except for some (arbitrarily chosen) common ones: `numpy as np`, `triton.language as tl`.
*   Import first-party modules through specific identifiers (ex. `from fast_llm.module import method`, not `import fast_llm.module`). This keeps Fast-LLM identifiers to a manageable length and makes it easier to track what is used in a given file.
*   Always use absolute imports (ex. no `from .module import method`)
*   Include all explicitly-imported third-party module to `setup.cfg`.
Only add new requirements if they provide a substantial benefit,
as we try to keep the requirements to a minimum.
*   Prefer file-level imports over imports inside methods, unless they significantly slow down the import process
or concern an optional dependency that should not be absolutely required to import the module (ex. `transformers`).
If an offending import is only required for a type hint, include it in a `if typing.TYPE_CHECKING:` block.

!!! note "Why this matters"
    Most python conventions make no clear recommendation concerning imports,
    which can easily lead to inconsistent import formats across a repo, and can make it harder to understand.
    Our conventions aim to avoid these arbitrary choices by providing an explicit prescription,
    which should be good enough nearly everywhere. Our choice is justified as follows:

    * For third-party and standard library packages, fully qualified identifiers are typically relatively short,
    so it makes sense to keep them.
    This also keeps identifier's origin explicit so anyone can tell where it came from with just a quick glance at the code.
    This is especially useful for identifiers that with otherwise ambiguous source (ex. `float32` may come from torch, numpy, triton, etc.; Fast-LLM's configuration scheme has many identifiers in common with `dataclasses`, `omegaconf` and `pydantic`)
    * For first-package, fully qualified names are generally too long to use in code,
    since they include the entire directory structure to the Fast-LLM,
    so first-party identifiers need to be imported by name.
    There should be very little ambiguity, because name clashes are uncommon within Fast-LLM,
    and external identifiers are already clearly marked as such.

!!! warning "Configuration modules"
    Fast-LLM supports instantiation and validation of configurations with a barebone installation.
    Because of this, modules that contain configuration classes (usually named `config.py`)
    should not include any top-level third-party import (except for those installed in the [barebone install](https://github.com/ServiceNow/Fast-LLM/blob/main/setup.cfg)),
    and the same applies for configuration initialization and validation methods.
    Third-party import may be included in other methods if needed.

## 🔓 Public and Private Interface

We use the following conventions for class and module interfaces:

*   Mark private and protected variables with an underscore `_` prefix.
As is customary in python, we make no distinction between the two and avoid the double-underscore `__` notation.
*   Keep public interfaces (methods and variables without underscore prefix) as lean as possible,
i.e. mark everything as private/protected unless there is a clear need to make it public.
We can always add to the public interface later, but removing from it is difficult.
*   Use accessors sparingly through the `@property` decorator or equivalent,
usually to define read-only public variables.

!!! note "Why this matters"
    Although good practices of object-oriented programming are generally ignored in python,
    Fast-LLM attempts to follow them to an extent, while avoiding unnecessary bloat.
    Public interfaces are expected to be stable,
    which make further modifications difficult as they could break external code.
    On the other hand, private interface are freely modifiable,
    which provides more freedom for fixes, improvement, refactoring, etc.
    Therefore, having lean public interfaces is critical for us to keep maintaining and improving Fast-LLM.

## 💡 Type Hints

Fast-LLM uses type hints for several reasons, including code readability, type checking in IDEs,
and type validation for configurations:

*   Always use type hints for the public interface of a classes and modules.
Type hints for method outputs may be omitted if they can be trivially inferred,
ex. if they return the input, an explicitly typed variable or nothing.
*   Prefer using type hints in private interfaces, especially if it improves readability and/or static type checking.
*   Prefer newer type hint formats over older ones, ex. `typing.List -> list`, `typing.Union(A,B) -> A | B`.

!!! note "Why this matters"
    We use type hints for various reasons. In addition to making the code more understandable,
    they are used by IDEs such as VS Code or PyCharm to perform static type checking,
    which speeds up development and is essential to keeping the code bug-free.

## 🗑️ Misc

*   Please add descriptions and comments as needed, especially for parts that would otherwise be difficult to understand.
*   Please favor `pathlib` over `os.path` for file path operations because it offers a cleaner and more modern API.
*   We encourage the use of modern python features when beneficial, up to the minimum python version (3.12).
