---
title: "Quick Start 🚀"
---

This guide will get you up and running with Fast-LLM on a single machine. Let's train a model and see some results!

You'll need:

- At least one NVIDIA GPU on your machine. We recommend 8 A100s or higher for this tutorial 🤑
- Docker (installed and running)
- Some patience for the initial setup and training 😊

## Step 1: Pull the Fast-LLM Docker Image

To start, grab the pre-built Fast-LLM Docker image:

```bash
docker pull ghcr.io/servicenow/fast-llm:latest
```

## Step 2: Set Up Directories for Your Inputs and Outputs

Let's create folders to store our input data and output results:

```bash
mkdir ~/inputs ~/results
```

## Step 3: Preparing the Training Data

For this tutorial, we'll use 9B tokens of text from the [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset. This dataset is a free approximation of the WebText data OpenAI used for GPT-2, and it's perfect for our setup!

We've got a script that'll download and preprocess the dataset for you. Run it like this:

!!! info inline end "What's Happening Here?"

    This will grab the OpenWebText data, tokenize it with the GPT-2 tokenizer, and save it in 91 shards of 100M tokens each. Expect around 2 hours for the whole thing to finish, mainly due to tokenization. If you've got more CPU cores, try upping `num_processes_*` to speed things up.

```bash
python tools/prepare_dataset.py \                          
    tokenizer_path_or_name="gpt2" \             
    dataset_name_or_path="openwebtext" \                                       
    dataset_split="train" \
    dataset_field="text" \
    output_dir="inputs" \ 
    num_processes_load=4 \
    num_processes_map=4 \
    num_processes_save=4 \
    num_tokens_per_shard=100000000
```

## Step 4: Choose Your Model

Fast-LLM supports many GPT variants, including (but not limited to) GPT-2, Llama, Mistral, and Qwen. For this tutorial, let's train the GPT-2 model from scratch with Fully Sharded Data Parallelism (FSDP). We'll grab a configuration file from Huggingface Hub and save it as `~/inputs/config.json`:

=== "GPT-2 (124M)"

    ```bash
    curl -O https://huggingface.co/openai-community/gpt2/resolve/main/config.json
    ```

=== "GPT-2 XL (1558M)"

    ```bash
    curl -O https://huggingface.co/openai-community/gpt2-xl/resolve/main/config.json
    ```

=== "Llama-3.2-3B-Instruct"

    ```bash
    curl -O https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/config.json
    ```

=== "Qwen2.5-3B-Instruct"

    ```bash
    curl -O https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/main/config.json
    ```

!!! tip "Model Size Matters"

    Smaller models like GPT-2 (124M) will train relatively quickly, especially if you've only got a few GPUs. But if you're feeling adventurous (and patient), give the larger models a shot!

## Step 5: Set Up Your Training Configuration

Next, we'll create a configuration file for Fast-LLM. Save the following as `~/inputs/fast-llm-config.yaml`:

```yaml
training:
  train_iters: 600_000  # (1)!
  logs:
    interval: 10
  validation:
    iterations: 25
    interval: 1000
  checkpoint:
    interval: 1000
    keep_latest: 5
  test_iters: 0
  export:
    format: huggingface
    interval: 20_000
  wandb:
    project_name: fast-llm
    entity_name: servicenow  # (2)!
    tags: quick-start
    alert:
      interval: 1000
batch:
  micro_batch_size: 1  # (3)!
  sequence_length: 1024  # (4)!
  batch_size: 480  # (5)!
data:
  format: file
  split: [998, 2, 0]  # (6)!
optimizer:
  weight_decay: 0.1  # (7)!
  beta_1: 0.9  # (8)!
  beta_2: 0.95  # (9)!
  learning_rate:
    base: 6.0e-04  # (10)!
    minimum: 6.0e-05  # (11)!
    decay_style: cosine  # (12)!
    decay_iterations: 600_000  # (13)!
    warmup_iterations: 2000  # (14)!
pretrained:
  format: huggingface
  path: /app/inputs  # (15)!
  load_weights: False  # (16)!
model:
  multi_stage:
    zero_stage: 2
  distributed:
    training_dtype: bf16
run:
  experiment_dir: /app/results
```

1. Total number of tokens will be ~300B.
2. Replace `servicenow` with your own W&B entity name.
3. Adjust based on GPU memory. For GPT-2 and an A100-80GB, a `micro_batch_size` of 1 should work well.
4. Should be a power of 2 and divisible by 8. For an A100-80GB, 1024 is a good starting point.
5. Must be divisible by number of GPUs. At 1024 tokens per sequence, 480 corresponds to about ~500k tokens per batch.
6. 99.8% train, 0.2% validation, 0% test.
7. L2 regularization penalty.
8. 1st Adam optimizer parameter.
9. 2nd Adam optimizer parameter.
10. Peak learning rate.
11. Should be 1/10th of base per Chinchilla.
12. Cosine decay starting at `base` after warmup and ending at `minimum` after `decay_iterations`.
13. Usually the same as `train_iters`.
14. Number of steps of linear warmup.
15. Location of the `config.json` file downloaded in Step 4.
16. Set to `False` to train from scratch.

## Step 6: Add Your Weights & Biases API Key

Save your Weights & Biases API key to `~/inputs/.wandb_api_key` so Fast-LLM can track your training progress there. You can create a free W&B account if you don't already have one.

## Step 7: Launch Training

Alright, the big moment! If you're on an 8-GPU machine, run the following to kick off training:

```bash
docker run --gpus all -it --rm ghcr.io/servicenow/fast-llm:latest \
    -v ~/inputs:/app/inputs \
    -v ~/results:/app/results \
    -e PYTHONHASHSEED=0 \
    -e WANDB_API_KEY_PATH=/app/inputs/.wandb_api_key \
    torchrun --nproc_per_node=8 --no_python fast-llm train gpt --config /app/inputs/fast-llm-config.yaml
```

!!! note

    Setting the Python hash seed to 0 ensures consistent, reproducible ordering in hash-dependent operations across processes, which is crucial for parallel computations.

Expect training to run for a few days (for a full 300B tokens). Keep an eye on the validation loss. You should see it drop as the model learns.

## Tracking Your Progress with W&B 📊

With Weights & Biases, you'll see the loss curve, training metrics, and more. If you follow this whole training setup, you should see the validation loss approaching the ballpark of ~2.85 (similar to the original GPT-2 model finetuned on OpenWebText).

## Troubleshooting Basics 🛠️

Here are some common issues you might encounter and how to address them:

- **CUDA Out of Memory**: Try lowering the `micro_batch_size` or `sequence_length` in your configuration to fit within available memory.

- **Underutilized GPU or Low Memory Usage**: If memory usage is low or GPU utilization isn't maxed out, try increasing `micro_batch_size` (to 4, 8, or 16 if memory allows) or extending `sequence_length` (up to 2048, 3072, or 4096, as memory permits). Larger batches and longer sequences help keep GPUs engaged and reduce idle time.

- **Docker Permission Issues**: If you encounter Docker permission errors, confirm that Docker has permission to access your GPUs. Use the `--gpus all` flag in your Docker run command and ensure your user has access to the `docker` and `nvidia-docker` groups.

## Final Thoughts

And that's it! You've set up, prepped data, chosen a model, configured training, and launched a full training run with Fast-LLM. From here, feel free to tweak the model, try out larger datasets, or scale things up to a multi-node setup if you're on a cluster.
We have guides for Slurm and Kubernetes setups if distributed training is your jam. Happy training! 🚀
