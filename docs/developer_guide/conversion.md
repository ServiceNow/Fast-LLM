---
title: Model Conversion Reference
---

!!! warning

    This reference is a work in progress. Stay tuned for new updates!

This reference guide describes all there is to know about Fast-LLM's checkpoint conversion system.
After reading this, you should be able to create your own `External` converter, in Hugging Face format or other.
And if you are familiar with the rest of Fast-LLM, you will also be able to create an entirely custom converter.

## Conversion and checkpointing basics

Fast-LLM provides a simple and [fully customizable interface](https://github.com/ServiceNow/Fast-LLM/blob/main/fast_llm/engine/checkpoint/config.py) to save/load checkpoints and configurations.
This same interface is used both by Fast-LLM official checkpoint formats (`distributed` and `fast-llm`),
and by the checkpoint conversion interface.
It can also be used to define entirely new checkpoint formats, though this is generally not recommended.

In this guide we focus on the checkpoint conversion interface, in particular for Hugging Face formats,
since this is the most common use case.

## Checkpoint format metadata

When creating a new checkpoint format, the first step is to subclass `CheckpointFormat`.
This data structure holds important properties of the format, and makes them accessible at the configuration level.
Some important entries include:
*   `name`: A name for the format, as will appear for example in configuration files
*   `support_optimizer`: Whether the optimizer state can be included in a checkpoint.
*   `support_saving`, `support_loading`: This can be used to create read-only or write-only formats.
*   `get_handler_class()`: Return the actual checkpoint conversion class, as we'll soon describe.
The class should be imported lazily so the `CheckpointFormat` remains accessible by configurations.

Here is a simple example:
```python
class AwesomeCheckpointFormat(CheckpointFormat):
    name = "awesome_checkpoint"
    support_optimizer = False

    @classmethod
    def get_handler_class(cls):
        from package.module import AwesomeCheckpointHandler

        return AwesomeCheckpointHandler
```

Once the metadata class is created, we want to let the model know about it.
We do this by adding it to the `checkpoint_formats` property of the model configuration class. For example:
```python
@config_class()
class AwesomeModelConfig(FastLLMModelConfig):
    checkpoint_formats = FastLLMModelConfig.checkpoint_formats + (AwesomeCheckpointFormat,)
    # ...
```

## External checkpoint handler

Now that we've defined a format, we're ready to tackle the actual implementation of an [external checkpoint handler](https://github.com/ServiceNow/Fast-LLM/blob/main/fast_llm/engine/checkpoint/external.py).
External handlers define a list of converters that can be used to convert configurations and state tensors automatically.
They also require an implementation of checkpoint reading and writing,
although we already provide such [implementation](https://github.com/ServiceNow/Fast-LLM/blob/main/fast_llm/engine/checkpoint/huggingface.py) for Hugging Face formats.

### Configuration conversion

The configuration conversion utility interfaces between two configurations in the form of nested dictionaries:
a serialized Fast-LLM configuration and an external configuration.
The `_load_config` method is expected to read the configuration on disk, as expected by the checkpoint format,
and return the same configurstion in the forma of a nested dictionary,
with `_save_config` handling the reverse operation.
See the [Hugging Face implementation](https://github.com/ServiceNow/Fast-LLM/blob/main/fast_llm/engine/checkpoint/huggingface.py) for an example.

To perform the conversion, the checkpoint handler relies on a list of `ParamConverter` objects,
which describe how individual parameters (or in some case multiple ones) should be converted.
The `ParamConverter` base interface consists of four entries:
*   `fast_llm_names: tuple[tuple[str, ...], ...]`:
*   `export_names: tuple[tuple[str, ...], ...]`:
*   `export_params(self, fast_llm_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]`:
*   `import_params(self, export_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]`:


### State conversion
