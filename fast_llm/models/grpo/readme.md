# Custom model template

The "custom" model is a template for customized training of a GPT-style model,
for example to fine-tune it for a particular class.
This is typically done as follows:

1. Create a copy of the `custom` model, and rename it appropriately, ex. `my_model`, `MyModelTrainer`, etc.
2. If necessary, adjust the base classes to inherit from more abstract classes or another model.
ex. `MyModelData(AbstractData)` to re-implement data processing from scratch.
3. Add custom configuration fields in `config.py`.
4. Adapt or re-implement the data loading scheme in `MyModelData`.
5. Adapt or re-implement the preprocessing scheme in `MyModelBaseModel`.
6. Adapt or re-implement the model head, ex. change the task and/or add a custom loss.
7. If needed, adapt the huggingface interface to return outputs for the desired task.
8. Apply other changes as needed.
9. Add the new model to the registry (`models.auto.py`) so it can be used through the cli.
10. Run training with the new model, ex. `fast-llm train my_model [...]`.


## Preprocessing variables and kwargs

To pass additional parameters to the model during preprocessing, ex. a target for the loss or a runtime parameter,
simply add them to the returned `kwargs`.
Those kwargs will be passed directly to the `forward` method of each layer and can be used as needed.

In some cases, it may be desirable to modify the `kwargs` inside a layer,
for example to pass additional data to other layers or to the backward pass.
This possible with certain caveats:
* There is no direct support for autograd. Detaching tensors is recommended to prevent memory losses.
* Such modifications may be incompatible with pipeline parallelism,
as the data will not be transferred to pipeline-parallel devices.


## Disclaimer

Model customization is a work in progress.
Some abstractions may be missing or poorly implemented,
and some methods and variables may be hard-coded or very difficult to override.
We intend to address these issues in the future, but it will most likely incur some breaking changes in the interface.
