---
title: Training Llama 3.1 8B
---

In this guide, we provide step-by-step instructions to do continued pretraining on The Stack with Llama 3.1-8B 🦙.

# Preliminary steps
- [Quick Start](quick-start.md)
- [Data preparation](data-preparation.md)

# Download the Pretrained Model
Let's download Llama-3.1-8B:
```bash
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.1-8B ./fast-llm-tutorial/pretrained-model
```

# Training
This is not much different from a pretraining config. We will:
- specify the Llama3.1 checkpoint to load. Fast-LLm will automatically infer the corresponding model architecture.
- adapt some of the training parameters for our needs.
- and that's it!

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
    export:  # (1)!
      format: llama
      interval: 20_000
  batch:
    micro_batch_size: 2
    sequence_length: 4096
    batch_size: 256
  data:
    format: file
    path: fast-llm-tutorial/dataset.json  # (2)!
    split: [99, 1, 0]  
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

    1.  A Llama model will be saved in Hugging Face format to `~/results` directory every 20,000 iterations.
    2.  Location of the dataset metadata file generated in Step 4.
    3.  The learning-rate can be used to trade-off between learning and forgetting. A higher learning-rate will learn quickly on our new dataset but will cause forgetting. A lower learning-rate will instead retain more of the pretrained model's knowledge, but will slow down adapting to the new domain.
    4.  Config of the pretrained model. We load a `llama` model from the repository downloaded earlier.
    5.  This tells Fast-LLM to load the weights of the pretrained model. If we wanted to use Llama-3.1's configuration, but train from scratch, we could use the same config but set this to `no`.

# Checkpoint usage
Checkpoints will be saved regularly, and every 20k steps a checkpoint will be exported in the HF format.
You can use it in `transformers` as you would use the pretrained Llama 3.1 model, except this one will be much stronger at {lang}!

```python
from transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("fast-llm-tutorial/pretrained-model")
pipe = pipeline("text-generation", model="fast-llm-tutorial/Llama-3.1-8B-cpt/export/llama/20000/", tokenizer=tokenizer)

```
