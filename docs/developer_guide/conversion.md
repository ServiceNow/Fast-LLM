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

Configuration conversion is handled by a `HuggingFaceBaseModelConverter` subclass,
which is linked to the handler via a `base_model_converter_class` class variable.
The converter implements three class methods:

*   `import_config(cls, config: dict) -> dict`:
Reads the external (e.g., Hugging Face) configuration dict and returns a Fast-LLM `base_model` config dict.
*   `export_config(cls, config: BaseModelConfig) -> dict`:
Takes a Fast-LLM `BaseModelConfig` object and returns the corresponding external configuration dict.
*   `get_converters(cls, config: BaseModelConfig, exported_config: dict) -> list[WeightConverter]`:
Returns the list of weight converters for this model (described in the next section).

The `_load_config` and `_save_config` methods on the handler read and write the external configuration file.
See the [Hugging Face implementation](https://github.com/ServiceNow/Fast-LLM/blob/main/fast_llm/engine/checkpoint/huggingface.py) for their default implementation.

Continuing our `AwesomeModel` example, the base model converter class could look like:

```python
class AwesomeBaseModelConverter(HuggingFaceBaseModelConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        # Build and return a Fast-LLM base_model config dict from the external config.
        return {
            "hidden_size": config["hidden_size"],
            "embeddings": {"vocab_size": config["vocab_size"]},
            "decoder": {
                "num_blocks": config["num_hidden_layers"],
                "block": {
                    "mixer": {
                        "heads": config["num_attention_heads"],
                        "head_groups": config.get("num_key_value_heads", config["num_attention_heads"]),
                        "rotary": {"type": "default", "theta": config.get("rope_theta", 10000)},
                        "add_linear_biases": False,
                    },
                    "mlp": {
                        "intermediate_size": config["intermediate_size"],
                        "gated": True,
                        "activation": ActivationType.from_hf_name(config["hidden_act"]),
                        "add_linear_biases": False,
                    },
                    "normalization": {"type": "rms_norm", "epsilon": config["rms_norm_eps"]},
                },
            },
            "head": {"normalization": {"type": "rms_norm", "epsilon": config["rms_norm_eps"]}},
            "tied_embedding_weight": config.get("tie_word_embeddings", False),
        }

    @classmethod
    def export_config(cls, config: AwesomeBaseModelConfig) -> dict:
        # Build and return the external config dict from the Fast-LLM config object.
        decoder_block = config.decoder.block
        return {
            "model_type": "awesome_model",
            "architectures": ["AwesomeModelForCausalLM"],
            "hidden_size": config.hidden_size,
            "vocab_size": config.embeddings.vocab_size,
            "num_hidden_layers": config.decoder.num_blocks,
            "num_attention_heads": decoder_block.mixer.heads,
            "num_key_value_heads": decoder_block.mixer.head_groups,
            "rope_theta": decoder_block.mixer.rotary.theta,
            "intermediate_size": decoder_block.mlp.intermediate_size,
            "hidden_act": decoder_block.mlp.activation.hf_name,
            "rms_norm_eps": decoder_block.normalization.epsilon,
            "tie_word_embeddings": config.tied_embedding_weight,
        }

    @classmethod
    def get_converters(cls, config: AwesomeBaseModelConfig, exported_config: dict) -> list[WeightConverter]:
        # Described in the next section.
        ...
```

Then wire the converter into the handler via `base_model_converter_class`:

```python
class AwesomeHuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    _model_class = AwesomeModelConfig
    architecture = "AwesomeModelForCausalLM"
    base_model_converter_class = AwesomeBaseModelConverter

    @classmethod
    def get_transformers_configuration_class(cls):
        from transformers import AutoConfig
        return AutoConfig
```

### State conversion

State conversion follows the same principle as configuration conversion, but acts on flat dictionaries of state tensors.
Converters are defined by subclassing `WeightConverter`, with the interface:

*   `fast_llm_name: str | tuple[str, ...]`: A state dict key, or tuple of keys, on the Fast-LLM side.
For example, `"layers.0.mixer.weight"` or `("layers.0.weight_1", "layers.0.weight_2")`.
*   `export_name: str | tuple[str, ...]`: A state dict key, or tuple of keys, on the external side.
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

We define the list of weight converters in the `get_converters` class method of the base model converter.
Continuing our `AwesomeModel` example, we define:

```python
    @classmethod
    def get_converters(cls, config: AwesomeBaseModelConfig, exported_config: dict) -> list[WeightConverter]:
        converters = []
        # The set of converters may depend on the base model configuration.
        num_layers = config.decoder.num_blocks

        # A simple renaming example, for the word embeddings.
        converters.append(WeightConverter("layers.0.word_embeddings_weight", "model.embed_tokens.weight"))

        # We usually want to loop dynamically over layers.
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
