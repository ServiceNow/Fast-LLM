---
title: Configuring Data for Training
---

In this section we show how to configure datasets through a series of examples

We already saw an example dataset configuration in the [quick-start guide](../quick-start.md), where we prepared a simple dataset and split it into training and validation sub-datasets, and used these to train a small model. This was done by:

1. Defining a dataset preparation configuration.
2. Running `fast-llm prepare` with said configuration. This generated some binary files along with two fast-llm configuration files, `fast-llm-tutorial/dataset/fast_llm_config_training.yaml` and `fast-llm-tutorial/dataset/fast_llm_config_validation.yaml`.
3. Defining a fast-llm data configuration that use those datasets:

    ```yaml
    data:
      datasets:
        Training:
          type: file
          path: fast-llm-tutorial/dataset/fast_llm_config_training.yaml
        Validation:
          type: file
          path: fast-llm-tutorial/dataset/fast_llm_config_validation.yaml
    ```

4. Running `fast-llm training` with said configuration.

In this section we are interested in generalizing step 3. For more details on steps 1 and 2, please refer to the quick-start guide or [this example](data-configuration.md).

## Example 1: Blending multiple datasets

In this example, we have three datasets and want to sample from each of them during training with probabilities 0.70, 0.25 and 0.05. For this, we use the `blended` type which takes other datasets as arguments:

```yaml
data:
  datasets:
    Training:
      type: blended
      datasets:
        - type: file
          path: path/to/dataset_0.yaml
        - type: file
          path: path/to/dataset_1.yaml
        - type: file
          path: path/to/dataset_2.yaml
      weights: [0.70, 0.25, 0.05]
```

!!! note "Dataset wrappers"
    The `blended` dataset wrapper is one example of the many dataset wrappers available in fast-llm. Such wrappers may be nested (almost) arbitrarily to generate the dataset scheme that fits your needs. Fast-LLM will use the `type` argument to dynamically select the appropriate configuration class(es). With some effort you can even create your own wrapper!

## Example 2: Configure shuffling

In this example, we have a large dataset that comes pre-shuffled, so shuffling in unnecessary for the first epoch.

```yaml
data:
  datasets:
    Training:
      type: file
      path: path/to/dataset.yaml
  sampling:
    shuffle: skip_first_epoch
```

## Example 3: Disable shuffling for validation

In this example, we want to disable shuffling entirely, but only for the validation dataset. We can do this with the `sampled` dataset wrapper:

```yaml
data:
  datasets:
    Training:
      type: file
      path: path/to/training_dataset.yaml
    Validation:
      type: sampled
      dataset:
        type: file
        path: path/to/validation_dataset.yaml

      sampling:
        shuffle: disabled
```

!!! note "More about sampling configuration"
    Sampling parameters may be globally defined through data configuration (example 2), dataset wrapper(s) (examples 3, 4), or both (example 5). In the case where a dataset sampling is configured with both methods (or multiple nested wrappers), (innermost) wrapper overrides the data (or next-to-innermost wrapper) for the explicitly defined fields (and only those).

## Example 4: Set sampling seed for individual datasets

In this example, we have a blend of datasets as in example 1, but we wish to set the seed for each dataset individually for reproducibility reasons. For this, we use the `seed` field of the `sampling` wrapper:

```yaml
data:
  datasets:
    Training:
      type: blended
      datasets:
        - type: sampled
          dataset:
            type: file
            path: path/to/dataset_0.yaml
          sampling:
            seed:1234
        - type: sampled
          dataset:
            type: file
            path: path/to/dataset_0.yaml
          sampling:
            seed:2345
        - type: sampled
          dataset:
            type: file
            path: path/to/dataset_0.yaml
          sampling:
            seed:3456
      weights: [0.70, 0.25, 0.05]
```

!!! note "Default seed"
    In the absence of explicit seed, Fast-LLM uses a default seed (`data.sampling`'s default) instead, and uses seed shifts to ensure different seeds for each phase and for the various blended datasets.

## Example 5: Advanced scenario

In this example, we combine everything we learned so far to create a complex scenario, where:

* The training dataset is a blend consists of two datasets, one of them being itself a blend of three datasets.
* All datasets except for one come pre-shuffled, so can skip shuffling for the first epoch.
* We want to set the seed explicitly for the validation and innermost blended datasets, but keep the default seed for the others.

```yaml
data:
  datasets:
    Training:
      type: blended
      datasets:
        - type: sampled
          dataset:
            type: blended
            datasets:
              - type: file
                # Seed = 1234
                path: path/to/dataset_0.yaml
              - type: file
                # Seed = 1234 + blend_shift, shuffle = skip_first_epoch
                path: path/to/dataset_1.yaml
              - type: sampled
                dataset:
                  type: file
                  # Seed = 1234 + 2 * blend_shift, shuffle = epoch
                  path: path/to/dataset_2.yaml
                sampling:
                  # Shuffle each epoch independently (default shuffling)
                  shuffle: epoch
          sampling:
            seed: 1234
        - type: file
          # Seed = default + train_shift + 2 * blend_shift, shuffle = skip_first_epoch
          path: path/to/dataset_3.yaml
      weights: [0.70, 0.25, 0.05]
    Validation:
        type: sampled
        dataset:
          type: file
          # Seed = 2345, shuffle = skip_first_epoch
          path: path/to/validation_dataset.yaml
        sampling:
          seed: 2345
  sampling:
    shuffle: skip_first_epoch
```

!!! note "Configure from file"
    If a dataset configuration is especially complex and makes the dataset configuration excessively big, or is reused across many experiments, you may want to save it to a yaml file and refer to it un the config using a `file` dataset. This can be used to reduce the present example to
    
    ```yaml
    data:
      datasets:
        Training:
          type: file
          path: path/to/training_dataset_config.yaml
        Validation:
          type: file
          path: path/to/validation_dataset_config.yaml
      sampling:
        shuffle: skip_first_epoch
    ```

    In fact, all the elementary datasets from file we've been using so far are of this format, and consist of more elementary `memmap` datasets optionally wrapped with `blended` and/or `slice` wrappers.
