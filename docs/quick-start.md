---
title: "Quick Start"
---



- **Purpose:** This section should provide an easy entry point for users who want to quickly get up and running with Fast-LLM. A single-node setup is a reasonable assumption for most users, as it doesn’t require specialized hardware or admin-level permissions on large clusters.
- **Content Ideas:**
  - **Single Node Setup Guide** (with GPU access, Docker, and root/privileged access assumed): Walk through the installation, setting up Docker, configuring the environment, and launching a simple training or inference task.
  - **Running Your First Model**: Include a basic example with a small dataset to show how to use Fast-LLM on a local machine.
  - **Troubleshooting Basics**: Common issues that users might run into when setting up on a single node.
- **Why It Makes Sense:** Most users will likely start with a local environment to experiment with Fast-LLM, so guiding them through a single-node setup as the "Getting Started" entry point makes it more approachable.

---

we want to asume that the user has at least one NVIDIA GPU available in one machine, and that they have Docker installed.

in that case, it's really straightforward to get started with Fast-LLM.

first, let's download a pre-built Docker image with Fast-LLM:

```bash
docker pull ghcr.io/servicenow/fast-llm:latest
```

let's make two directories to store our inputs and outputs:

```bash
mkdir ~/inputs ~/results
```

then let's download a huggingface config file for training a model and save this as `~/inputs/config.json`:

=== "Llama-3.2-3B-Instruct"

    ```bash
    curl -O https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/config.json
    ```

=== "Qwen2.5-3B-Instruct"

    ```bash
    curl -O https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/main/config.json
    ```

Now let's use this config in our Fast-LLM training configuration file:

```yaml
training:
  train_iters: 100
  logs:
    interval: 10
  validation:
    iterations: null
  test_iters: 0
batch:
  micro_batch_size: 1
  batch_size: 32
data:
  format: random
  split: [1, 0, 0]
optimizer:
  learning_rate:
    base: 1.0e-05
pretrained:
  format: huggingface
  path: /app/inputs
model:
  multi_stage:
    zero_stage: 2
  distributed:
    training_dtype: bf16
run:
  experiment_dir: /app/results
```

save this to a file called `~/inputs/fast-llm-config.yaml`.
this will be mounted into the Docker container when we run it.

then, run the following command:

```bash
docker run --gpus all -it --rm ghcr.io/servicenow/fast-llm:latest
        -v ~/inputs:/app/inputs
        -v ~/results:/app/results
        torchrun --nproc_per_node=8 --no_python fast-llm train gpt --config /app/inputs/fast-llm-config.yaml
```

[^^^ this may be incorrect, I'm not sure how to run the command]

[I want the number of GPUs be auto-detected, but I'm not sure how to do that]
