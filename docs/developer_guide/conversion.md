---
title: Model Conversion Reference
---

This reference guide describes all there is to know about Fast-LLM's checkpoint conversion system.
After reading this, you should be able to create your own `External` converter, in Hugging Face format or other.
And if you are familiar with the rest of Fast-LLM, you will also be able to create an entirely custom converter.

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

You can see a more complete example in the [GPT model source code](https://github.com/ServiceNow/Fast-LLM/blob/main/fast_llm/models/gpt/config.py).

## External checkpoint handler

Now that we've defined a format, we're ready to tackle the actual implementation of an [external checkpoint handler](https://github.com/ServiceNow/Fast-LLM/blob/main/fast_llm/engine/checkpoint/external.py).
External handlers define a list of converters that can be used to convert configurations and state tensors automatically.
They also require an implementation of checkpoint reading and writing,
although we already provide such [implementation](https://github.com/ServiceNow/Fast-LLM/blob/main/fast_llm/engine/checkpoint/huggingface.py) for Hugging Face formats.

!!! note "Supported formats"
    The external checkpoint conversion is principally designed for checkpoint formats that store state tensors in a variable list of `Savetensor` files.
    It comes with default saving and loading that handles lazy loading, management of memory usage, safety checks.
    It is possible to use a more generic format by overriding the `save` and (in some cases) `load` methods, but this requires significant effort.
    Note that we may provide better generalization options at some point in the future.

Let's begin an example where we convert our `AwesomeModel` to its Hugging Face counterpart.
The first step is to define a handler class and let it know about our model class:

```python
class AwesomeHuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    _model: AwesomeModel
    _model_class= AwesomeModelConfig
```

### Configuration conversion

The configuration conversion utility interfaces between two configurations in the form of nested dictionaries:
a serialized Fast-LLM configuration and an external configuration.
The `_load_config` method is expected to read the configuration on disk, as expected by the checkpoint format,
and return the same configuration in the forma of a nested dictionary,
with `_save_config` handling the reverse operation.
See the [Hugging Face implementation](https://github.com/ServiceNow/Fast-LLM/blob/main/fast_llm/engine/checkpoint/huggingface.py) for an example.

To perform the conversion, the checkpoint handler relies on a list of `ParamConverter` objects,
which describe how individual parameters (or in some case multiple ones) should be converted.
The `ParamConverter` base interface is a dataclass consisting of two variables and two methods:

*   `fast_llm_names: tuple[tuple[str, ...], ...]`: An array of entry names on the Fast-LLM side, in tuple format.
For example, `((transformer, head_groups),)` refers to the single entry `config["transformer"]["head_groups"]`.
*   `export_names: tuple[tuple[str, ...], ...]`: An array of entry names on the external side, in the same tuple format.
*   `export_params(self, fast_llm_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]`:
This method takes the configuration parameters corresponding to `fast_llm_names` (in the same order),
and returns converted parameters corresponding to `export_names`.
*   `import_params(self, export_values: tuple[typing.Any, ...]) -> tuple[typing.Any, ...]`:
The converse of`export_params`, converting parameters corresponding to `export_names` into those corresponding to `fast_llm_names`.

While not strictly part of the interface, it may also be useful to define a dataclass `__post_init__`,
for example to restrict the number of parameters in `fast_llm_names` and `export_names`.

Fast-LLM offers several generic configuration converter classes, including:

*   `RenameParamConverter`: A simple 1-1 mapping between parameters, with optional renaming but identical value.
Typically, most converters are of this type.
*   `ConstantImportParamConverter`: A 1-0 mapping for Fast-LLM parameters that without an equivalent in the external format,
that must take a specific value `fast_llm_value` for conversion to make sense (i.e., they take a hard-coded value in the external format).
This type of converter is common for Hugging Face converters, as Hugging Face models support much fewer configuration parameters.
*   `ConstantExportParamConverter`: A 0-1 mapping, the converse of `ConstantImportParamConverter`
*   `MappedConfigParamConverter`: A 1-1 mapping similar to `RenameParamConverter`, but with a non-trivial relation between values.

In addition to those, you may need to implement your own custom converter.
Here is an example that associates several Fast-LLM variables with a tuple.

```python
@dataclasses.dataclass(kw_only=True)
class PackingParamConverter(ParamConverter):
    def __post_init__(self):
        # There may be any number of Fast-LLM variables, but only one external one
        Assert.eq(len(self.export_names), 1)

    def export_params(self, fast_llm_values):
        # Pack the values into a single tuple.
        return (fast_llm_values,)

    def import_params(self, export_values):
        # Unpack the values. We can safely assume `export_values` has length one because of the assertion in `__post_init__`
        return export_values[0]
```

Now that we've seen how parameter converters work, we're ready to add them to our handler class.
We do so by creating a list of converters in the `_create_config_converters` class method.
Continuing our `AwesomeModel` handler example, we define:

```python
    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        # For Hugging Face handlers, we need to call the superclass method.
        return super()._create_config_converters() + [
            # A trivial example where both the name and value are the same on both sides.
            RenameParamConverter(
                fast_llm_names=(("vocab_size",),),
                export_names=(("vocab_size",),),
            ),
            # A non-trivial example of `RenameParamConverter` with renaming and handling of nested dictionaries.
            RenameParamConverter(
                fast_llm_names=(("transformer", "rotary", "theta"),), export_names=(("rope_theta",),)
            ),
            # A constant import example indicating that the external format does not support absolute positional embeddings.
            ConstantImportParamConverter(fast_llm_names=(("use_position_embeddings",),), fast_llm_value=False),
            # The `architectures` parameter is a common use case for `ConstantExportParamConverter` in Hugging Face models.
            ConstantExportParamConverter(export_names=(("architectures",),), export_value=["AwesomeModelForCausalLM"]),
            # A value mapping example, where we match Fast-LLM activation types with their Hugging Face equivalents.
            MappedConfigParamConverter(
                fast_llm_names=(("transformer", "activation_type"),),
                export_names=(("hidden_act",),),
                fast_llm_value=ActivationType.from_hf_name,
                export_value=lambda activation_type: activation_type.hf_name,
            ),
            # A more hypothetical example using `PackingParamConverter` to pack two parameters `epsilon_1`, `epsilon_2` into a tuple `eps`.
            PackingParamConverter(
                fast_llm_names=(("epsilon_1",),("epsilon_2",)),
                export_names=(("eps",),),
            ),
        ]
```

!!! note "How conversion works"
    The once the converters are defined, the conversion utility takes it from there.
    Exporting works as follows (importing work similarly):
    *The handler creates an empty export config dict, then loops over its list of converters. For each converter, it:
    *   Reads the value of each parameter defined in `fast_llm_names`, and gathers them in a tuple.
    *Calls `converter.export_params`, providing the set of read values as argument.
    *   Ensure that the returned value has the correct length (that of `export_names`)
    *   Set the respective values in the export config dict.

!!! note "About `MISSING` and `DEFAULT`"
    If a value is not found during import, it will be replaced by the `MISSING` tag.
    The converter's `import_params` has the opportunity to handle this missing value,
    and if a `MISSING`, the handler will throw an error because it does not know what value to set on the Fast-LLM side.

    The `MISSING` tag is also supported during export,
    but has a different meaning as the value is always expected to be found in the Fast-LLM configuration.
    Instead, `export_params` may return a `MISSING` tag indicating that no value should not be added to the Fast-LLM config.
    It may also return `DEFAULT`, which will be replaced by the default value for the configuration parameter.

    Note that the handling of `MISSING` and `DEFAULT` is experimental and may be improved in the future.

### State conversion

State conversion follows the same principle as configuration conversion, but acts on flat dictionaries of state tensors.
Converters are defined by subclassing `WeightConverter`, with the interface:

*   `fast_llm_name: str | tuple[str, ...]`: An entry name or array of entry names on the Fast-LLM side.
For example, `((transformer, head_groups),)` refers to the single entry `config["transformer"]["head_groups"]`.
*   `export_name: str | tuple[str, ...]`: An entry name or array of entry names on the external side.
*   `export_weight(self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]) -> tuple[torch.Tensor | SafeTensorSlice, ...]`:
This method takes the state dict entries corresponding to `fast_llm_name` (in the same order),
and returns converted entries corresponding to `export_name`.
*   `import_weight(self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]) -> tuple[torch.Tensor | SafeTensorSlice, ...]`:
The converse of`export_weight`, converting state dict entries corresponding to `export_name` into those corresponding to `fast_llm_name`.

Fast-LLM offers several generic state dict converter classes, including:

*   `WeightConverter`: The base class allows for a simple 1-1 mapping between parameters with optional renaming, similar to `RenameParamConverter`.
*   `SplitWeightConverter`: A 1-N mapping, where a Fast-LLM parameter corresponds to multiple equally-sized chunks in the external side.
This happens for example in the MLP, where Hugging Fast keeps the `gate` and `up` parts separate,
while Fast-LLM combines those in a single tensor to improve performance (and similarly for the multiple experts in the case of MoE).

Since different libraries tend to hold weights in different formats, it is often necessary to define custom converters.
Here is an example where a weight needs to be transposed during conversion:

```python
class TransposeWeightConverter(WeightConverter):
    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        Assert.eq(len(weight), 1)
        return (weight[0][:].transpose().contiguous(),)

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        Assert.eq(len(weight), 1)
        return (weight[0][:].transpose().contiguous(),)
```

We define the list of weight converters in the `_create_weight_converters` method.
Continuing our `AwesomeModel` handler example, we define:

```python
    def _create_weight_converters(self) -> list[WeightConverter]:
        converters = []
        # The set of converters may depend on the base model configuration, which is accessible through `self._model.base_model_config`.
        num_layers = len(self._model.config.base_model.decoder)

        # A simple renaming example, for the word embeddings.
        converters.append(WeightConverter("layers.0.word_embeddings_weight", "model.embed_tokens.weight"))

        # We usually want to loop dynamically over layers
        for i in range(num_layers):
            # A `SplitWeightConverter` example, splitting a weight in two.
            converters.append(SplitWeightConverter(
                f"layers.{i + 1}.weight",
                (f"model.layers.{i}.weight_1", f"model.layers.{i}.weight_2"),
            ))
        return converters
```

And that's it! We're ready to use the new checkpoint format in Fast-LLM.
For example, we may set the pretrained and export format in a configuration using

```yaml
training:
  export:
    format: awesome_checkpoint
pretrained:
  format: awesome_checkpoint
```

### External converters beyond Hugging Face

!!! warning
    Coming soon. Stay tuned for new updates!

## Creating a custom checkpoint format

!!! warning
    Coming soon. Stay tuned for new updates!
