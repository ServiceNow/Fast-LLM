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
      test_iters: 0
      export:
        format: llama
        interval: 20_000
    batch:
      micro_batch_size: 2
      sequence_length: 4096
      batch_size: 256
    data:
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
      base_model:
        cross_entropy_impl: fused
      multi_stage:
        zero_stage: 2
      distributed:
        training_dtype: bf16
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
      test_iters: 0
      export:
        format: qwen2
        interval: 20_000
    batch:
      micro_batch_size: 1
      sequence_length: 8192
      batch_size: 256
    data:
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
      base_model:
        cross_entropy_impl: fused
      multi_stage:
        zero_stage: 2
      distributed:
        training_dtype: bf16
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
          tie_word_embeddings: false
          use_position_embeddings: false
          vocab_size: 128256
          transformer:
            activation_type: silu
            add_linear_biases: false
            ffn_hidden_size: 14336
            gated: true
            head_groups: 8
            hidden_size: 4096  # (1)!
            kv_channels: 128
            normalization:
              type: rms_norm
            num_attention_heads: 32
            num_layers: 32
            rotary:
              type: llama3
              theta: 500_000
      ```
=== "Qwen 2.5 7B"
      ```yaml
      model:
        base_model:
          tie_word_embeddings: false
          use_position_embeddings: false
          vocab_size: 152064
          transformer:
            activation_type: silu
            add_linear_biases: only_attn_qkv
            ffn_hidden_size: 18944
            gated: true
            head_groups: 4
            hidden_size: 3584  # (1)!
            normalization:
              type: rms_norm
              epsilon: 1e-06
            num_attention_heads: 28
            num_layers: 28
            rotary:
              type: default
              theta: 1_000_000
      ```

1.  Hidden-size/num-layers will be used to provide good defaults for weight initialization std.

Configuring the model this way is a bit more verbose than using the pretrained configuration, but gives an idea of how to configure a the model with Fast-LLM.
