---
title: Training Llama 3.1 8B
---

Follow this guide to train a Llama-3.1 or Qwen 2.5 7B like model from scratch!

## Preliminary steps

- [Quick Start](../quick-start.md)
- [Data preparation](data-preparation.md)

## Training configuration

In this guide, we show you how to configure a model architecture and train a model from scratch.
Let's start from the following training configuration:
=== "Llama 3.1 8B"
    ```yaml
    training:
      train_iters: 100_000
      logs:
        interval: 10
      evaluators:
          interval: 100
          evaluator:
            type: loss
            iterations: 25
            dataset_name: validation
      checkpoint:
        interval: 1000
        keep: 5
      export:
        format: llama
        interval: 20_000
    data:
      micro_batch_size: 4096
      maximum_document_length: 4096
      datasets:
        training:
          type: file
          path: path/to/training_dataset_config.yaml
        validation:
          type: file
          path: path/to/validation_dataset_config.yaml
    optimizer:
      weight_decay: 0.1
      beta_1: 0.9
      beta_2: 0.95
      learning_rate:
        base: 6.0e-04
        minimum: 6.0e-05
        decay_style: cosine
        decay_iterations: 100_000
        warmup_iterations: 2000
    model:
      multi_stage:
        zero_stage: 2
      distributed:
        compute_dtype: bf16
    run:
      experiment_dir: fast-llm-tutorial/experiment
    ```
=== "Qwen 2.5 7B"
    ```yaml
    training:
      train_iters: 100_000
      logs:
        interval: 10
      evaluators:
        validation:
          interval: 100
          evaluator:
            type: loss
            iterations: 25
            dataset_name: validation
      checkpoint:
        interval: 1000
        keep: 5
      export:
        format: qwen2
        interval: 20_000
    data:
      micro_batch_size: 8192
      maximum_document_length: 8192
      datasets:
        training:
          type: file
          path: path/to/training_dataset_config.yaml
        validation:
          type: file
          path: path/to/validation_dataset_config.yaml
    optimizer:
      weight_decay: 0.1
      beta_1: 0.9
      beta_2: 0.95
      learning_rate:
        base: 6.0e-04
        minimum: 6.0e-05
        decay_style: cosine
        decay_iterations: 100_000
        warmup_iterations: 2000
    model:
      multi_stage:
        zero_stage: 2
      distributed:
        compute_dtype: bf16
    run:
      experiment_dir: fast-llm-tutorial/experiment
    ```

This configuration will not work because it misses important arguments to define model architecture.
There are 2 ways of instantiating our a model.

We could use a pretrained model config. This step is similar to what is done in the [Quick Start guide](../quick-start.md).
First download the model configuration:
=== "Llama 3.1 8B"
    ```bash
    git lfs install
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/meta-llama/Llama-3.1-8B ./fast-llm-tutorial/pretrained-model
    ```
=== "Qwen 2.5 7B"
    ```bash
    git lfs install
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen2.5-7B ./fast-llm-tutorial/pretrained-model
    ```

By specifying a pretrained model from the HuggingFace hub, Fast-LLM automatically converts the config to load the model.
    **Only the configuration is loaded, not the weights**, because of `model_weights: no`.
=== "Llama 3.1 8B"
    ```yaml
    pretrained:
      format: llama
      path: fast-llm-tutorial/pretrained_model
      model_weights: no
    ```
=== "Qwen 2.5 7B"
    ```yaml
    pretrained:
      format: qwen2
      path: fast-llm-tutorial/pretrained_model
      model_weights: no
    ```

Alternatively, we define the model architecture ourselves as follows:
=== "Llama 3.1 8B"
      ```yaml
      model:
        base_model:
          tied_embedding_weight: false
          hidden_size: 4096  # (1)!
          embeddings:
            vocab_size: 128256
          decoder:
            num_blocks: 32
            block:
              mixer:
                heads: 32
                head_groups: 8
                head_size: 128
                add_linear_biases: false
                rotary:
                  type: llama3
                  theta: 500_000
              mlp:
                intermediate_size: 14336
                gated: true
                activation: silu
                add_linear_biases: false
              normalization:
                type: rms_norm
          head:
            normalization:
              type: rms_norm
      ```
=== "Qwen 2.5 7B"
      ```yaml
      model:
        base_model:
          tied_embedding_weight: false
          hidden_size: 3584  # (1)!
          embeddings:
            vocab_size: 152064
          decoder:
            num_blocks: 28
            block:
              mixer:
                heads: 28
                head_groups: 4
                head_size: 128
                add_linear_biases: false
                query_layer:
                  bias:
                    enabled: true
                key_layer:
                  bias:
                    enabled: true
                value_layer:
                  bias:
                    enabled: true
                rotary:
                  type: default
                  theta: 1_000_000
              mlp:
                intermediate_size: 18944
                gated: true
                activation: silu
                add_linear_biases: false
              normalization:
                type: rms_norm
                epsilon: 1e-06
          head:
            normalization:
              type: rms_norm
              epsilon: 1e-06
      ```

1.  Hidden-size/num-layers will be used to provide good defaults for weight initialization std.

Configuring the model this way is a bit more verbose than using the pretrained configuration, but gives an idea of how to configure a the model with Fast-LLM.
