---
title: Continual Pretraining of Llama 3.1 8B or Qwen 2.5 7B
---


In this guide, we provide step-by-step instructions to do continued pretraining on The Stack with Llama 3.1 8B  or Qwen 2.5 7B models.

## Preliminary steps

- [Quick Start](../quick-start.md)
- [Data preparation](data-preparation.md)

## Download the Pretrained Model

Let's download the model first:
=== "Llama 3.1 8B"
    ```bash
    git lfs install
    git clone https://huggingface.co/meta-llama/Llama-3.1-8B ./fast-llm-tutorial/pretrained-model
    ```
=== "Qwen 2.5 7B"
    ```bash
    git lfs install
    git clone https://huggingface.co/Qwen/Qwen2.5-7B ./fast-llm-tutorial/pretrained-model
    ```

## Training

This is not much different from a pretraining config. We will:

- specify the the model checkpoint to load and its format. Fast-LLM will automatically infer the corresponding model architecture.
- adapt some of the training parameters for our needs.
- and that's it!
=== "Llama 3.1 8B"

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
      export:  # (1)!
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
          path: fast-llm-tutorial/dataset/fast_llm_config_training.yaml  # (2)!
        validation:
          type: file
          path: fast-llm-tutorial/dataset/fast_llm_config_validation.yaml  # (2)!
    optimizer:
      weight_decay: 0.1
      beta_1: 0.9
      beta_2: 0.95
      learning_rate:
        base: 1.0e-04  # (3)!
        minimum: 1.0e-05
        decay_style: cosine
        decay_iterations: 100_000
        warmup_iterations: 2000
    pretrained:  # (4)!
      format: llama
      path: fast-llm-tutorial/pretrained-model
      model_weights: yes  # (5)!
    model:
      base_model:
        transformer:
          use_flash_attention: yes
        cross_entropy_impl: fused
      multi_stage:
        zero_stage: 2
      distributed:
        training_dtype: bf16
    run:
      experiment_dir: fast-llm-tutorial/Llama-3.1-8B-cpt
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
      export:  # (1)!
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
          path: fast-llm-tutorial/dataset/fast_llm_config_training.yaml  # (6)!
        validation:
          type: file
          path: fast-llm-tutorial/dataset/fast_llm_config_validation.yaml  # (6)!
    optimizer:
      weight_decay: 0.1
      beta_1: 0.9
      beta_2: 0.95
      learning_rate:
        base: 1.0e-04  # (3)!
        minimum: 1.0e-05
        decay_style: cosine
        decay_iterations: 100_000
        warmup_iterations: 2000
    pretrained:  # (4)!
      format: qwen2
      path: fast-llm-tutorial/pretrained-model
      model_weights: yes  # (5)!
    model:
      base_model:
        transformer:
          use_flash_attention: yes
        cross_entropy_impl: fused
      multi_stage:
        zero_stage: 2
      distributed:
        training_dtype: bf16
    run:
      experiment_dir: fast-llm-tutorial/qwen-2.5-7B-cpt
    ```

1.  A the model will be saved in Hugging Face format to `~/results` directory every 20,000 iterations.
2.  Location of the dataset metadata file generated in Step 4 of quick start guide.
3.  The learning-rate can be used to trade-off between learning and forgetting. A higher learning-rate will learn quickly on our new dataset but will cause forgetting. A lower learning-rate will instead retain more of the pretrained model's knowledge, but will slow down adapting to the new domain.
4.  Config of the pretrained model. We load the model downloaded from the repository earlier.
5.  This tells Fast-LLM to load the weights of the pretrained model. If we wanted to use the model's configuration, but train from scratch, we could use the same config but set this to `no`.

## Checkpoint usage

Checkpoints will be saved regularly, and every 20k steps a checkpoint will be exported in the HF format.
You can use it in `transformers` as you would use the pretrained  model, except this one should be stronger on programming languages!
=== "Llama 3.1 8B"
    ```python
    from transformers import pipeline, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("fast-llm-tutorial/pretrained-model")
    pipe = pipeline("text-generation", model="fast-llm-tutorial/Llama-3.1-8B-cpt/export/llama/20000/", tokenizer=tokenizer)
    ```
=== "Qwen 2.5 7B"
    ```python
    from transformers import pipeline, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("fast-llm-tutorial/pretrained-model")
    pipe = pipeline("text-generation", model="fast-llm-tutorial/qwen-2.5-7B-cpt/export/qwen2/20000/", tokenizer=tokenizer)
    ```
