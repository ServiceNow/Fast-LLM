---
title: Training Llama 3.1 8B
---

Follow this guide to train a Llama-3.1 like model from scratch!


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
        format: llama
        interval: 20_000
    batch:
      micro_batch_size: 4
      sequence_length: 4096
      batch_size: 480
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
There are 2 ways of instantiating our Llama-3.1-8B model. We could use a pretrained model config, or define the model architecture ourselves.

=== "Pretrained configuration"
    This step is similar to what is done in the [Quick Start guide](quick-start.md).
    First download the model configuration:
    ```bash
    git lfs install
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/meta-llama/Llama-3.1-8B ./fast-llm-tutorial/pretrained-model
    ```
    By specifying a pretrained model from the HuggingFace hub, Fast-LLM automatically converts the config to load the model.
    **Only the configuration is loaded, not the weights**, because of `model_weights: no`.

    ```yaml
    pretrained:
      format: llama  
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
              scaling_type: llama3
            rotary_embedding_scale: -13.122363377404328  # (2)!
            use_rotary_embeddings: true
      ```

      1.  Hidden-size/num-layers will be used to provide good defaults for weight initialization std.
      2.  -ln(500_000)

      Configuring the model this way is a bit more verbose than using the pretrained configuration, but gives you an idea of how you configure a Llama-3.1-8B-like model with Fast-LLM.

