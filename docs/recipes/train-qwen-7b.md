---
title: Training Qwen 2.5 7B
---

Follow this guide to train a Qwen-2.5 like model from scratch!


# Preliminary steps
- [Quick Start](quick-start.md)
- [Data preparation](data-preparation.md)


# Training configuration
In this guide, we show you how to configure a model architecture and train a model from scratch.
Let's start from the following training configuration:

    ```yaml
    training:
      train_iters: 100_000
      logs:
        interval: 10
      validation:
        iterations: 25
        interval: 1000
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
      format: file
      path: fast-llm-tutorial/dataset/fast_llm_dataset.json
      split: [99, 1, 0]
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
There are 2 ways of instantiating our Qwen-2.5-7B model. We could use a pretrained model config, or define the model architecture ourselves.

=== "Pretrained configuration"
    This step is similar to what is done in the [Quick Start guide](quick-start.md).
    First download the model configuration:
    ```bash
    git lfs install
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen2.5-7B ./fast-llm-tutorial/pretrained-model
    ```
    By specifying a pretrained model from the HuggingFace hub, Fast-LLM automatically converts the config to load the model.
    **Only the configuration is loaded, not the weights**, because of `model_weights: no`.

    ```yaml
    pretrained:
      format: qwen2  
      path: fast-llm-tutorial/pretrained_model
      model_weights: no 
    ```

=== "From-scratch configuration"
      In this step, we specify the model architecture as follows:
      
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

      Configuring the model this way is a bit more verbose than using the pretrained configuration, but gives an idea of how to configure a Qwen-2.5-7B-like model with Fast-LLM.

