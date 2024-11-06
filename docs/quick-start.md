---
title: "Quick Start 🚀"
---

This guide will get you up and running with Fast-LLM on a single machine. Let's train a model and see some results!

You'll need:

-   At least one NVIDIA GPU on your machine. We recommend 8 A100s or higher for this tutorial 🤑
-   Docker (installed and running). You can run without without docker but we don't recommend it. 🐳
-   Some patience for the initial setup and training 😊

## Step 1: Pull the Fast-LLM Docker Image 🐳

To start, grab the pre-built Fast-LLM Docker image:

```bash
docker pull ghcr.io/servicenow/fast-llm:latest
```

This image contains everything you need to train LLMs with Fast-LLM.

!!! info

    Installing Fast-LLM from source is also an option:

    ```sh
    pip install --no-build-isolation "git+https://github.com/ServiceNow/Fast-LLM.git#egg=fast_llm[CORE,OPTIONAL,DEV]"
    ```

    However, we recommend the Docker image for simplicity and reproducibility.

## Step 2: Set Up Directories for Your Inputs and Outputs

Let's create folders to store our input data and output results:

```bash
mkdir ~/inputs ~/results
```

## Step 3: Choose Your Model 🤖

Fast-LLM supports many GPT variants, including (but not limited to) Llama, Mistral, and Mixtral. For this tutorial, let's train a Llama model with data parallelism. You can choose from two models:

=== "SmolLM-135M"

    SmolLM is a smaller, more manageable model with 135M parameters. It's perfect for testing and getting familiar with Fast-LLM. We'll grab its configuration file from Huggingface Hub and save it as `~/inputs/config.json`:

    ```bash
    curl -O https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/config.json
    mv config.json ~/inputs
    ```

=== "Llama-3.2-1B"

    Llama is a larger model with 1B parameters. It's more powerful but requires more resources to train. We'll grab the model from Huggingface Hub and save it to `~/inputs`:

    ```bash
    git lfs install
    git clone https://huggingface.co/meta-llama/Llama-3.2-1B ~/inputs
    ```

!!! tip "Model Size Matters"

    Smaller models like SmolLM-135M will train relatively quickly, especially if you've only got a few GPUs. But if you're feeling adventurous (and patient), give the larger Llama-3.2-1B a shot!

## Step 4: Preparing the Training Data 📚

For this tutorial, we'll use 9B tokens of text from the [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset. This dataset is a free approximation of the WebText data OpenAI used for GPT-2, and it's perfect for our test run!

We've got a script that'll download and preprocess the dataset for you. Run it like this:

=== "SmolLM-135M"

    ```bash
    docker run -it --rm ghcr.io/servicenow/fast-llm:latest \
        -v ~/inputs:/app/inputs \
        python tools/prepare_dataset.py \
        tokenizer_path_or_name="HuggingFaceTB/SmolLM-135M" \
        dataset_name_or_path="openwebtext" \
        dataset_split="train" \
        output_dir="inputs" \
        num_processes_load=4 \
        num_processes_map=4 \
        num_processes_save=4 \
        num_tokens_per_shard=100000000
    ```

=== "Llama-3.2-1B"

    ```bash
    docker run -it --rm ghcr.io/servicenow/fast-llm:latest \
        -v ~/inputs:/app/inputs \
        python tools/prepare_dataset.py \
        tokenizer_path_or_name="meta-llama/Llama-3.2-1B" \
        dataset_name_or_path="openwebtext" \
        dataset_split="train" \
        output_dir="inputs" \
        num_processes_load=4 \
        num_processes_map=4 \
        num_processes_save=4 \
        num_tokens_per_shard=100000000
    ```

!!! info "What's Happening Here?"

    This will grab the OpenWebText data, tokenize it, and save it in 91 shards of 100M tokens each. Expect around 2 hours for the whole thing to finish, mainly due to tokenization. If you've got more CPU cores, try upping `num_processes_*` to speed things up.

!!! tip "Use a Smaller Dataset for Testing"

    Since we're just testing things out, we can also use a smaller dataset. Replace `openwebtext` with `stas/openwebtext-10k` to use a small subset representing the first 10K records from the original dataset. This will speed up the process and let you see how things work without waiting for hours.

## Step 5: Set Up Your Training Configuration ⚙️

Next, we'll create a configuration file for Fast-LLM. Save the following as `~/inputs/fast-llm-config.yaml`:

=== "SmolLM-135M"

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
        keep: 5
      test_iters: 0
      export: # (2)!
        format: llama
        interval: 20_000
      wandb: # (3)!
        project_name: fast-llm
        entity_name: servicenow
        tags: quick-start
    batch:
      micro_batch_size: 1  # (4)!
      sequence_length: 1024
      batch_size: 480  # (5)!
    data:
      format: file
      path: /app/inputs/fast_llm_dataset.json  # (6)!
      split: [99, 1, 0]  # (7)!
    optimizer: # (8)!
      weight_decay: 0.1
      beta_1: 0.9
      beta_2: 0.95
      learning_rate: # (9)!
        base: 6.0e-04
        minimum: 6.0e-05
        decay_style: cosine
        decay_iterations: 600_000
        warmup_iterations: 2000
    pretrained:
      format: llama  # (10)!
      path: /app/inputs
      load_weights: no  # (11)!
    model:
      multi_stage:
        zero_stage: null  # (12)!
      distributed:
        training_dtype: bf16  # (13)!
    run:
      experiment_dir: /app/results
    ```

    1.  Total number of training tokens will be approximately 300B.
    2.  A Llama model will be saved in Hugging Face format to `~/results` directory every 20,000 iterations.
    3.  Entirely optional, but it's a good idea to track your training progress with Weights & Biases. Replace `servicenow` with your own W&B entity name. If you don't want to use W&B, just remove this section.
    4.  Adjust the number of sequences per GPU based on GPU memory. For SmolLM-135M and an A100-80GB, a `micro_batch_size` of 1 should work well.
    5.  Must be divisible by the number of GPUs and the `micro_batch_size`. At 1024 tokens per sequence, 480 corresponds to about 500,000 tokens per batch.
    6.  Location of the dataset metadata file generated in Step 4.
    7.  99% train, 1% validation, 0% test. These settings need to be adjusted based on the size of your dataset. If you're using a smaller dataset, you need to increase the validation split.
    8.  These are good default optimizer settings for training models.
    9.  We are using a cosine decay schedule with linear warmup. After reaching the peak learning rate `base` at `warmup_iterations`, the learning rate will decay to `minimum` at `decay_iterations`, following a cosine curve. The minimum learning rate should be 1/10th of the base learning rate per Chinchilla.
    10.  Format of the pretrained model. Since SmolLM is a Llama model, we set this to `llama`.
    11.  We'll train SmolLM-135M from scratch. You can set to `yes` to continue training from a checkpoint (if you put one in `~/inputs`).
    12.  We're not using ZeRO for this tutorial, so we set `zero_stage` to `null`. You can set this to `1`, `2`, or `3` for ZeRO-1, ZeRO-2, or ZeRO-3, respectively.
    13.  `bf16` is supported on Ampere GPUs and higher. Fast-LLM also supports `fp16`.

=== "Llama-3.2-1B"

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
        keep: 5
      test_iters: 0
      export:  # (2)!
        format: llama
        interval: 20_000
      wandb:  # (3)!
        project_name: fast-llm
        entity_name: servicenow
        tags: quick-start
    batch:
      micro_batch_size: 1  # (4)!
      sequence_length: 1024
      batch_size: 480  # (5)!
    data:
      format: file
      path: /app/inputs/fast_llm_dataset.json  # (6)!
      split: [99, 1, 0]  # (7)!
    optimizer: # (8)!
      weight_decay: 0.1
      beta_1: 0.9
      beta_2: 0.95
      learning_rate:  # (9)!
        base: 6.0e-04
        minimum: 6.0e-05
        decay_style: cosine
        decay_iterations: 600_000
        warmup_iterations: 2000
    pretrained:
      format: llama  # (10)!
      path: /app/inputs
      load_weights: yes  # (11)!
    model:
      multi_stage:
        zero_stage: null  # (12)!
      distributed:
        training_dtype: bf16  # (13)!
    run:
      experiment_dir: /app/results
    ```

    1.  Total number of training tokens will be approximately 300B.
    2.  A Llama model will be saved in Hugging Face format to `~/results` directory every 20,000 iterations.
    3.  Entirely optional, but it's a good idea to track your training progress with Weights & Biases. Replace `servicenow` with your own W&B entity name. If you don't want to use W&B, just remove this section.
    4.  Adjust the number of sequences per GPU based on GPU memory. For Llama-3.2-1B and an A100-80GB, a `micro_batch_size` of 1 should work well.
    5.  Must be divisible by the number of GPUs and the `micro_batch_size`. At 1024 tokens per sequence, 480 corresponds to about 500,000 tokens per batch.
    6.  Location of the dataset metadata file generated in Step 4.
    7.  99% train, 1% validation, 0% test. These settings need to be adjusted based on the size of your dataset. If you're using a smaller dataset, you need to increase the validation split.
    8.  These are good default optimizer settings for training models.
    9.  We are using a cosine decay schedule with linear warmup. After reaching the peak learning rate `base` at `warmup_iterations`, the learning rate will decay to `minimum` at `decay_iterations`, following a cosine curve. The minimum learning rate should be 1/10th of the base learning rate per Chinchilla.
    10.  Format of the pretrained model. Since it's a Llama model, we set this to `llama`.
    11.  We want to continue training Llama-3.2-1B from a checkpoint. If you're training from scratch, set this to `no`.
    12.  We're not using ZeRO for this tutorial, so we set `zero_stage` to `null`. You can set this to `1`, `2`, or `3` for ZeRO-1, ZeRO-2, or ZeRO-3, respectively.
    13.  `bf16` is supported on Ampere GPUs and higher. Fast-LLM also supports `fp16`.

## (Optional) Step 6: Add Your Weights & Biases API Key 🔑

If you included the W&B section in your configuration, you'll need to add your API key. Save your W&B API key to `~/inputs/.wandb_api_key` so Fast-LLM can track your training progress there. You can create a free W&B account if you don't already have one.

## Step 7: Launch Training 🚀

Alright, the big moment! If you're on an 8-GPU machine, run the following to kick off training:

```bash
docker run --gpus all -it --rm ghcr.io/servicenow/fast-llm:latest \
    -v ~/inputs:/app/inputs \
    -v ~/results:/app/results \
    -e PYTHONHASHSEED=0 \
    -e WANDB_API_KEY_PATH=/app/inputs/.wandb_api_key \
    torchrun --nproc_per_node=8 --no_python \
    fast-llm train gpt --config /app/inputs/fast-llm-config.yaml
```

!!! warning "Python Hash Seed"

    Setting the Python hash seed to 0 ensures consistent, reproducible ordering in hash-dependent operations across processes. Training will fail if this isn't set.

## Step 8. Track Training Progress 📊

Fast-LLM will log training progress to the console every 10 iterations. You can expect to see the following throughput:

=== "SmolLM-135M"

    | Metric              | A100         | H100         |
    |---------------------|-------------:|-------------:|
    | Tokens/s            | 1,234,567    | 1,456,789    |
    | TFLOPS              | 312          | 512          |

=== "Llama-3.2-1B"

    | Metric              | A100         | H100         |
    |---------------------|-------------:|-------------:|
    | Tokens/s            | 1,234,567    | 1,456,789    |
    | TFLOPS              | 312          | 512          |

If you included the W&B section in your configuration, you can also track your training progress on the Weights & Biases dashboard as well. Follow the link in the console output to view your training run.

## Troubleshooting Basics 🛠️

Here are some common issues you might encounter and how to address them:

-   **CUDA Out of Memory**: Try lowering the `micro_batch_size` or `sequence_length` in your configuration to fit within available memory.

-   **Underutilized GPU or Low Memory Usage**: If memory usage is low or GPU utilization isn't maxed out, try increasing `micro_batch_size` (to 4, 8, or 16 if memory allows) or extending `sequence_length` (up to 2048, 3072, or 4096, as memory permits). Larger batches and longer sequences help keep GPUs engaged and reduce idle time.

-   **Docker Permission Issues**: If you encounter Docker permission errors, confirm that Docker has permission to access your GPUs. Use the `--gpus all` flag in your Docker run command and ensure your user has access to the `docker` and `nvidia-docker` groups.

## Final Thoughts

And that's it! You've set up, prepped data, chosen a model, configured training, and launched a full training run with Fast-LLM. From here, feel free to tweak the model, try out larger datasets, or scale things up to a multi-node setup if you're on a cluster.
We have guides for Slurm and Kubernetes setups if distributed training is your jam. Happy training! 🚀
