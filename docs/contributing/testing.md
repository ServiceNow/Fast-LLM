---
title: Writing and running tests
---

## Debugging with tests

### Selecting tests

When debugging, it is often practical to target specific tests that will run quickly. While Pytest supports targeting specific directory, files or tests, the complex parameterization and dependencies of our tests often makes explicit targeting tedious and/or impractical. We provide several options for selecting tests:

* `--skip-slow`: This will run a subset of "fast" tests that cover the majority of our codebase. This is useful for quickly checking that changes did not break Fast-LLM too badly before running the full test suite. Note that parallel testing (`-n`) is not needed (and may be counter-productive) with this argument.
* `--run-extra-slow`: Some tests are disabled by default because they take too long to run (ex. complex integration tests) and/or are not particularly important. This argument re-enables them.
* `--models MODEL0 MODEL1 ...`: This allows targeting one or more specific models from the model tests (see below), and is particularly useful when debugging a model. For example, `pytest tests/models/test_models/test_checkpoint.py -v -ra --models llama` will test checkpoints specifically for the llama model. (Note that `-n` may not be needed here as model tests for a given model are only partly distributed dure to dependency constraints.)

### Monitoring distributed tests

`--no-distributed-capture`

### Other options

* `--show-gpu-memory N`: Our testing suite monitors GPU memory usage and reports the highest users. Use this option to adjust the number of reported tests (10 by default). Note that this option is mainly intended to make sure tests don't use too much memory (which could cause crashes with lots of parallel tests) and may not be an accurate measurement.
* `--show-skipped`: Many tests skipped for obvious reasons (ex. marked as slow or extra slow, skipped model testing groups (see below)) are removed entirely from the report to reduce clutter. This option may be used to show them explicitly.

## Best practices

## Testing models

[Model integration tests](https://github.com/ServiceNow/Fast-LLM/blob/main/tests/models) are the most important part of our testing suite, ensuring that Fast-LLM works and yields consistent results for a variety of models, training configurations, optimizations, etc.

For each tested model, we run a series of tests divided into several groups. Much of these tests consist of running a short Fast-LLM training run, then comparing intermediate tensors (ex. parameter initialization, layer outputs and gradients, parameter gradients) against a baseline.

### What is being tested

Coming soon.

!!! warning "Don't forget about unit tests!"

    While adding a model is a quick and efficient way to increase coverage, it is **not a replacement for unit tests**.
    The model testing suite performs intensive consistency checks, but does little to make sure those results are correct to begin with. See [functional tests](https://github.com/ServiceNow/Fast-LLM/blob/main/tests/functional) and [test_lm_head](https://github.com/ServiceNow/Fast-LLM/blob/main/tests/layers/test_lm_head.py) for good examples of unit tests for individual components and an entire layer.

### Adding a model

When adding support for a new model that comes with additional features, the simplest option to increase coverage is to add an example configuration to the [tested modelsl](https://github.com/ServiceNow/Fast-LLM/blob/main/tests/utils/model_configs.py).
In general, adding a model means calling `_update_and_add_testing_config` with:

* A reference model configuration (ex. `llama`).
* A name for the new model (ex. `my_custom_llama`)
* A list of arguments that make the model unique (ex. `["model.base_model.my_feature=True"]`)
* An optional checkpoint format to use for conversion tests (ex. `MyCustomLlamaCheckpointFormat`)
* A `groups` argument detailing the action to take for each testing group.

Here is a detailed example and reference:

```python
_update_and_add_testing_config(
    "llama",
    "my_custom_llama",
    model_type="custom_llama_gpt",
    extra_args=["model.base_model.my_feature=True"],
    megatron_args=None, # A list of arguments for the associated Megatron run, if applicable.
    checkpoint_format=MyCustomLlamaCheckpointFormat,
    groups={
        ModelTestingGroup.basic: ModelTestingGroupAction.normal,
        ModelTestingGroup.checkpoint: ModelTestingGroupAction.normal,
        ModelTestingGroup.convert: ModelTestingGroupAction.normal,
        ModelTestingGroup.generate: ModelTestingGroupAction.normal,
        ModelTestingGroup.megatron: ModelTestingGroupAction.not_implemented,
        ModelTestingGroup.distributed: ModelTestingGroupAction.unimportant,
    },
    compare_factor=2.0, # Optionally adjust comparison thresholds, ex. for a model that comes with unusually high variances.
)
```

#### Reference for groups

Fast-LLM currently supports the following testing groups:

* `basic`: Run Fast-LLM training with a baseline configuration to make sure the model can be run. Run a variety of single-gpu configuration (ex. different data type, gradient accumulation) and check that the results are consistent with the baseline. Typically set to `normal` (runs for all models).
* `checkpoint`: Test basic checkpoint saving and loading. Typically set to `normal` (runs for all models).
* `convert`: Test more advanced checkpoint manipulation that involve conversion between different formats. Typically set to `normal` (runs for all models that support external checkpoint formats).
* `generate`: Test generative inference through the Hugging Face model wrapper. Typically set to `normal` (runs for all models that support it).
* `megatron`: Compare against an equivalent training run in Megatron-LM. Typically set to `not_implemented` or `unimportant`, (slow and only a small selection of models support it).
* `distributed`: Run a variety of multi-gpu configurations and compare against the baseline. Typically set to `unimportant`, as it is very slow and resource-intensive.

Each testing group may be associated with one of the following options:

* `main`: Indicate that the group will always run, even when testing with `--skip-slow`. Typically not used for new models.
* `normal`: Indicate that this group is part of the standard testing suite.
* `not_implemented`: Indicate that the testing group is not supported and should never run.
* `broken` Indicate that the testing group is supported, but expected to fail due to a known issue. This test will be disabled unless using with `--test-extra-slow` so the testing suite isn't flooded with known failures. If using, please make sure that the problem is tracked through a github issue.
* `unimportant`: Indicate that the test is not worth running as part of the standard testing suite, ex. if all associated features already sufficiently tested through other models and/or groups. It will still run with `--test-extra-slow`.
