---
title: "Quick Start"
---

This guide will get you up and running with Fast-LLM on a single machine. Let's train a model and see some results!

## Prerequisites

To follow this guide, you'll need:

-   **Hardware**: At least one NVIDIA GPU with Volta architecture or newer. We wrote this guide with an 8-GPU machine of Ampere or Hopper architecture in mind.
-   **Software**:
    -   **Docker** (if using the Docker setup), or
    -   **Local Environment**: PyTorch 2.2 or later, CUDA 12.1 or later, and APEX AMP (if building from source), or
    -   **Cluster Setup**: Access to a Kubernetes or Docker-enabled Slurm cluster.
-   **Time**: The initial setup and training process requires a little patience. 😊

## 🏗 Step 1: Initial Setup

First, choose your environment. You can use Docker, your local environment, Slurm, or Kubernetes.

=== "Docker"

    You selected Docker for this tutorial. We'll use the Fast-LLM Docker image to train our model, which includes all the necessary dependencies. Grab the [pre-built Fast-LLM Docker image](https://github.com/ServiceNow/Fast-LLM/pkgs/container/fast-llm) from GitHub's container registry (GHCR).

    ```bash
    docker pull ghcr.io/servicenow/fast-llm:latest
    ```

    Let's also create folders to store our input data and output results:

    ```bash
    mkdir ~/inputs ~/results
    ```

=== "Local Environment"

    You're setting up Fast-LLM in your machine's local environment. This means you'll need to install Fast-LLM and its dependencies. For simplicity and reproducibility, we recommend using the Fast-LLM Docker image instead. It's preconfigured with everything you need. But if you're set on a local installation, follow the steps below.

    Fast-LLM depends on [CUDA](https://developer.nvidia.com/about-cuda) 12.1 or later, [PyTorch](https://pytorch.org) 2.2 or later, [APEX](https://github.com/NVIDIA/apex?tab=readme-ov-file#installation), and [OpenAI Triton](https://github.com/triton-lang/triton). Follow the instructions on their respective websites to install them. If you use [conda](https://docs.conda.io/projects/conda/en/latest/index.html), you can create a new environment and install these dependencies in it.
    
    Now, make sure PyTorch can access your GPU by running the following command:

    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```

    If APEX is correctly installed, the following command should run without errors:

    ```bash
    python -c "from amp_C import *"
    ```

    For Triton, you can verify the installation by running:

    ```bash
    python -c "import triton; print(triton.__version__)"
    ```
    
    Fast-LLM also depends on [FlashAttention-2](https://github.com/Dao-AILab/flash-attention), which will be installed automatically when you install Fast-LLM:

    ```bash
    pip install --no-build-isolation "git+https://github.com/ServiceNow/Fast-LLM.git#egg=fast_llm[CORE,OPTIONAL,DEV]"
    ```

    You can verify the installation by running:

    ```bash
    python -C "import flash_attn; print(flash_attn.__version__)"
    ```

    and

    ```bash
    python -c "import fast_llm; print(fast_llm.__version__)"
    ```

    At this point, you should be ready to run Fast-LLM on your local environment.

    Before we continue, let's create folders to store our input data and output results:

    ```bash
    mkdir /mnt/inputs /mnt/results
    ```

    If this location isn't writable, you can create the folders in your home directory:

    ```bash
    mkdir ~/inputs ~/results
    ```

    Make sure to update the paths in the following commands accordingly.

=== "Slurm"

    You've chosen Docker-enabled [Slurm](https://slurm.schedmd.com/) for this tutorial. Slurm will pull the `ghcr.io/servicenow/fast-llm:latest` Docker image to train the model. Just make sure there's a shared file system for both input data and output results. We'll assume your home directory is accessible across all nodes.

    Let's create a folder to store our input data and output results in the shared home directory:

    ```bash
    mkdir ~/inputs ~/results
    ```

=== "Kubernetes"

    You selected to use [Kubernetes](https://kubernetes.io/) with [KubeFlow](https://www.kubeflow.org/) for this tutorial. We will use a `PyTorchJob` resource to train our model with the `ghcr.io/servicenow/fast-llm:latest` Docker image and store our input data and output results in shared [persistent volume claims](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) (PVCs).

    Let's now create two PVCs named `pvc-fast-llm-inputs` and `pvc-fast-llm-results` to store our input data and output results, respectively.

    Create a file named `pvc-fast-llm-inputs.yaml` with the following content:

    ```yaml
    # Persistent volume claim for Fast-LLM inputs
    apiVersion: "v1"
    kind: "PersistentVolumeClaim"
    metadata:
      name: "pvc-fast-llm-inputs"
    spec:
      storageClassName: local-path
      accessModes:
        - ReadWriteMany
      resources:
        requests:
          storage: 1000Gi
    ```

    Then, create a second file named `pvc-fast-llm-results.yaml` with these contents:

    ```yaml
    # Persistent volume claim for Fast-LLM results
    apiVersion: "v1"
    kind: "PersistentVolumeClaim"
    metadata:
      name: "pvc-fast-llm-results"
    spec:
      storageClassName: local-path
      accessModes:
        - ReadWriteMany
      resources:
        requests:
          storage: 1000Gi
    ```

    Apply both PVCs to your Kubernetes cluster:

    ```bash
    kubectl apply -f pvc-fast-llm-inputs.yaml
    kubectl apply -f pvc-fast-llm-results.yaml
    ```

    We also need to create a temporary pod that mounts the inputs PVC and allows us to copy files there. Here's a basic YAML configuration for such a pod:

    ```yaml
    # Temporary pod to manage input data and results
    apiVersion: v1
    kind: Pod
    metadata:
      name: fast-llm-data-management
    spec:
      containers:
        - name: fast-llm-data-management-container
          image: ubuntu
          command: ["sleep", "infinity"]
          volumeMounts:
            - mountPath: /mnt/inputs
              name: inputs
            - mountPath: /mnt/results
              name: results
      volumes:
        - name: inputs
          persistentVolumeClaim:
            claimName: pvc-fast-llm-inputs
        - name: results
          persistentVolumeClaim:
            claimName: pvc-fast-llm-results
    ```

    Save this configuration to a file named `pod-fast-llm-data-management.yaml`. Next, apply this configuration to your Kubernetes cluster to create the pod:

    ```bash
    kubectl apply -f pod-fast-llm-data-management.yaml
    ```

    The pod will allow you to copy files to and from the inputs and results PVCs. You can access it by running:

    ```bash
    kubectl exec -it fast-llm-data-management -- /bin/bash
    ```

    !!! note "Cleaning up unused resources"
    
        At the very end of this guide, you should clean up the data management pod to avoid unnecessary resource consumption by running

        ```bash
        kubectl delete pod fast-llm-data-management
        ```

        Don't run this just yet, though. You'll need the pod throughout the guide.

## 🤖 Step 2: Choose Your Model

Fast-LLM supports many GPT variants, including (but not limited to) Llama, Mistral, and Mixtral. For this tutorial, you can choose from two models:

=== "SmolLM2-135M"

    SmolLM2 is a smaller, more manageable model with 135M parameters. It is similar to GPT-2 but with a few improvements. A perfect choice for testing and getting familiar with Fast-LLM. We'll grab the model from Huggingface Hub and save it to our inputs folder.

    === "Docker"

        ```bash
        git lfs install
        git clone https://huggingface.co/HuggingFaceTB/SmolLM2-135M ~/inputs/SmolLM2-135M
        ```

    === "Local Environment"

        ```bash
        git lfs install
        git clone https://huggingface.co/HuggingFaceTB/SmolLM2-135M /mnt/inputs/SmolLM2-135M
        ```

    === "Slurm"

        ```bash
        git lfs install
        git clone https://huggingface.co/HuggingFaceTB/SmolLM2-135M ~/inputs/SmolLM2-135M
        ```

    === "Kubernetes"

        ```bash
        kubectl exec -it fast-llm-data-management -- /bin/bash
        git lfs install
        git clone https://huggingface.co/HuggingFaceTB/SmolLM2-135M /mnt/inputs/SmolLM2-135M
        ```

=== "Llama-3.2-1B"

    Llama is a larger model with 1B parameters. It's more powerful but requires more resources to train. We'll grab the model from the Huggingface Hub and save it to our inputs folder.

    !!! note "Access Required"
    
        Meta gates access to their Llama models. You need to request access to the model from Meta before you can download it at https://huggingface.co/meta-llama/Llama-3.2-1B.

    === "Docker"

        First, sign in to your Hugging Face account:

        ```bash
        pip install huggingface_hub
        huggingface-cli login
        ```

        Then, clone the model:

        ```bash
        git lfs install
        git clone https://huggingface.co/meta-llama/Llama-3.2-1B ~/inputs/Llama-3.2-1B
        ```

    === "Local Environment"

        First, sign in to your Hugging Face account:

        ```bash
        pip install huggingface_hub
        huggingface-cli login
        ```

        Then, clone the model:

        ```bash
        git lfs install
        git clone https://huggingface.co/meta-llama/Llama-3.2-1B /mnt/inputs/Llama-3.2-1B
        ```
    
    === "Slurm"

        First, sign in to your Hugging Face account:

        ```bash
        pip install huggingface_hub
        huggingface-cli login
        ```

        Then, clone the model:

        ```bash
        git lfs install
        git clone https://huggingface.co/meta-llama/Llama-3.2-1B ~/inputs/Llama-3.2-1B
        ```
    
    === "Kubernetes"
    
        First, sign in to your Hugging Face account:

        ```bash
        kubectl exec -it fast-llm-data-management -- /bin/bash
        pip install huggingface_hub
        huggingface-cli login
        ```
        
        Then, clone the model:

        ```bash
        git lfs install
        git clone https://huggingface.co/meta-llama/Llama-3.2-1B /mnt/inputs/Llama-3.2-1B
        ```

!!! tip "Model Size Matters"

    Smaller models like SmolLM2-135M will train relatively quickly, especially if you've only got a few GPUs. But if you're feeling adventurous (and patient), give the larger Llama-3.2-1B a shot!

## 📚 Step 3: Prepare the Training Data

For this tutorial, we'll use 9B tokens of text from the [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset. This dataset is a free approximation of the WebText data OpenAI used for GPT-2, and it's perfect for our test run!

Create a configuration file for the dataset preparation. Copy the following content:

=== "SmolLM2-135M"

    ```yaml
    output_path: /mnt/inputs/openwebtext-SmolLM2

    loading_workers: 4
    tokenize_workers: 4
    saving_workers: 4

    dataset:
      path: openwebtext
      trust_remote_code: true

    tokenizer:
      path: /mnt/inputs/SmolLM2-135M/tokenizer.json

    remove_downloads: false
    ```

=== "Llama-3.2-1B"

    ```yaml
    output_path: /mnt/inputs/openwebtext-Llama

    loading_workers: 4
    tokenize_workers: 4
    saving_workers: 4

    dataset:
      path: openwebtext
      trust_remote_code: true
    
    tokenizer:
      path: /mnt/inputs/Llama-3.2-1B/tokenizer.json
    
    remove_downloads: false
    ```

and save it as `prepare-config.yaml` in your inputs folder.

Fast-LLM ships with a `prepare` command that'll download and preprocess the dataset for you. Run it like this:

=== "Docker"

    ```bash
    docker run -it --rm ghcr.io/servicenow/fast-llm:latest \
        -v ~/inputs:/mnt/inputs \
        fast-llm prepare gpt_memmap --config /mnt/inputs/prepare-config.yaml
    ```

=== "Local Environment"

    ```bash
    fast-llm prepare gpt_memmap --config /mnt/inputs/prepare-config.yaml
    ```

=== "Slurm"

    ```bash
    sbatch <<EOF
    #!/bin/bash
    # SBATCH --nodes=1
    # SBATCH --ntasks-per-node=1
    # SBATCH --exclusive
    # SBATCH --output=/mnt/results/job_output.log
    # SBATCH --error=/mnt/results/job_error.log

    srun \
        --container-image="ghcr.io/servicenow/fast-llm:latest" \
        --container-mounts="${HOME}/inputs:/mnt/inputs,${HOME}/results:/mnt/results" \
        --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
        bash -c "fast-llm prepare gpt_memmap --config /mnt/inputs/prepare-config.yaml"
    EOF
    ```

    You can follow the job's progress by running `squeue -u $USER` and checking the logs in `job_output.log` and `job_error.log` in your results folder.

=== "Kubernetes"

    ```bash
    kubectl apply -f prepare-job.yaml
    ```

    where `prepare-job.yaml` is a file containing the following configuration:

    ```yaml
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: fast-llm-prepare
    spec:
      template:
        spec:
          containers:
            - name: fast-llm-prepare-container
              image: ghcr.io/servicenow/fast-llm:latest
              command: ["fast-llm", "prepare", "gpt_memmap"]
              args:
                - "--config"
                - "/mnt/inputs/prepare-config.yaml"
              resources:
                requests:
                  cpu: 4
              volumeMounts:
                - name: inputs
                  mountPath: /mnt/inputs
          volumes:
            - name: inputs
              persistentVolumeClaim:
                claimName: pvc-fast-llm-inputs
    ```

    You can follow the job's progress by running `kubectl get pods` and checking the logs with `kubectl logs fast-llm-prepare`.

!!! tip "Use a Smaller Dataset for Testing"

    The full OpenWebText dataset is quite large and will take a while to process, around 2 hours. If you're just testing things out, you can also use a smaller dataset. Replace `openwebtext` with `stas/openwebtext-10k` to use a small subset representing the first 10K records from the original dataset. This will speed up the process and let you see how things work without waiting for hours.

## ⚙️ Step 4: Configure Fast-LLM

Next, we'll create a configuration file for Fast-LLM. Save the following as `train-config.yaml` in your inputs folder:

=== "SmolLM2-135M"

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
        project_name: fast-llm-quickstart
        group_name: SmolLM2-135M
        entity_name: servicenow
    batch:
      micro_batch_size: 60  # (4)!
      sequence_length: 1024
      batch_size: 480  # (5)!
    data:
      format: file
      path: /mnt/inputs/openwebtext-SmolLM2/fast_llm_dataset.json  # (6)!
      split: [99, 1, 0]  # (7)!
    optimizer:  # (8)!
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
      path: /mnt/inputs/SmolLM2-135M
      model_weights: no  # (11)!
    model:
      base_model:
        transformer:
          use_flash_attention: yes  # (12)!
      multi_stage:
        zero_stage: null  # (13)!
      distributed:
        training_dtype: bf16  # (14)!
    run:
      experiment_dir: /mnt/results/SmolLM2-135M
    ```

    1.  Total number of training tokens will be approximately 300B.
    2.  A Llama model will be saved in Hugging Face format to `~/results` directory every 20,000 iterations.
    3.  Entirely optional, but it's a good idea to track your training progress with Weights & Biases. Replace `servicenow` with your own W&B entity name. If you don't want to use W&B, just remove this section.
    4.  Adjust the number of sequences per GPU based on GPU memory. For SmolLM2-135M and an A100-80GB, a `micro_batch_size` of 60 should work well.
    5.  Must be divisible by the number of GPUs and the `micro_batch_size`. At 1024 tokens per sequence, 480 corresponds to about 500,000 tokens per batch.
    6.  Location of the dataset metadata file generated in Step 4.
    7.  99% train, 1% validation, 0% test. These settings need to be adjusted based on the size of your dataset. If you're using a smaller dataset, you need to increase the validation split.
    8.  These are good default optimizer settings for training models.
    9.  We are using a cosine decay schedule with linear warmup. After reaching the peak learning rate `base` at `warmup_iterations`, the learning rate will decay to `minimum` at `decay_iterations`, following a cosine curve. The minimum learning rate should be 1/10th of the base learning rate per Chinchilla.
    10.  Format of the pretrained model. Since SmolLM is a Llama model, we set this to `llama`.
    11.  We'll train SmolLM2-135M from scratch. You can set to `yes` to continue training from a checkpoint (if you put one in `~/inputs`).
    12.  If you're using Ampere GPUs or higher, you can enable FlashAttention for faster training. Otherwise, set this to `no`. The default is `yes`.
    13.  We're not using ZeRO for this tutorial, so we set `zero_stage` to `null`. You can set this to `1`, `2`, or `3` for ZeRO-1, ZeRO-2, or ZeRO-3, respectively.
    14.  `bf16` (bfloat16, or Brain Floating Point 16) is supported on Ampere GPUs and higher. On Volta GPUs, you can use `fp16` (half-precision floating point) for training instead of `bf16`.

=== "Llama-3.2-1B"

    ```yaml
    training:
      train_iters: 100_000  # (1)!
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
        project_name: fast-llm-quickstart
        group_name: Llama-3.2-1B
        entity_name: servicenow
    batch:
      micro_batch_size: 20  # (4)!
      sequence_length: 1024
      batch_size: 480  # (5)!
    data:
      format: file
      path: /mnt/inputs/openwebtext-Llama/fast_llm_dataset.json  # (6)!
      split: [99, 1, 0]  # (7)!
    optimizer:  # (8)!
      weight_decay: 0.1
      beta_1: 0.9
      beta_2: 0.95
      learning_rate:  # (9)!
        base: 6.0e-04
        minimum: 6.0e-05
        decay_style: cosine
        decay_iterations: 100_000
        warmup_iterations: 2000
    pretrained:
      format: llama  # (10)!
      path: /mnt/inputs/Llama-3.2-1B
      model_weights: yes  # (11)!
    model:
      base_model:
        transformer:
          use_flash_attention: yes  # (12)!
        cross_entropy_impl: fused  # (13)!
      multi_stage:
        zero_stage: null  # (14)!
      distributed:
        training_dtype: bf16  # (15)!
    run:
      experiment_dir: /mnt/results/Llama-3.2-1B
    ```

    1.  Total number of training tokens will be approximately 300B.
    2.  A Llama model will be saved in Hugging Face format to `~/results` directory every 20,000 iterations.
    3.  Entirely optional, but it's a good idea to track your training progress with Weights & Biases. Replace `servicenow` with your own W&B entity name. If you don't want to use W&B, just remove this section.
    4.  Adjust the number of sequences per GPU based on GPU memory. For Llama-3.2-1B and an A100-80GB, a `micro_batch_size` of 20 should work well.
    5.  Must be divisible by the number of GPUs and the `micro_batch_size`. At 1024 tokens per sequence, 480 corresponds to about 500,000 tokens per batch.
    6.  Location of the dataset metadata file generated in Step 4.
    7.  99% train, 1% validation, 0% test. These settings need to be adjusted based on the size of your dataset. If you're using a smaller dataset, you need to increase the validation split.
    8.  These are good default optimizer settings for training models.
    9.  We are using a cosine decay schedule with linear warmup. After reaching the peak learning rate `base` at `warmup_iterations`, the learning rate will decay to `minimum` at `decay_iterations`, following a cosine curve. The minimum learning rate should be 1/10th of the base learning rate per Chinchilla.
    10.  Format of the pretrained model. Since it's a Llama model, we set this to `llama`.
    11.  We want to continue training Llama-3.2-1B from a checkpoint. If you're training from scratch, set this to `no`.
    12.  If you're using Ampere GPUs or higher, you can enable FlashAttention for faster training. Otherwise, set this to `no`. The default is `yes`.
    13.  Configure Fast-LLM to use the fused cross-entropy loss implementation rather than the default Triton implementation for Llama models. This avoids issues with block size limitations in our current Triton code, which can cause training failures.
    14.  We're not using ZeRO for this tutorial, so we set `zero_stage` to `null`. You can set this to `1`, `2`, or `3` for ZeRO-1, ZeRO-2, or ZeRO-3, respectively.
    15.  `bf16` (bfloat16, or Brain Floating Point 16) is supported on Ampere GPUs and higher. On Volta GPUs, you can use `fp16` (half-precision floating point) for training instead of `bf16`.

## 🔑 (Optional) Step 6: Add Your Weights & Biases API Key

If you included the W&B section in your configuration, you'll need to add your API key. Save your W&B API key to `.wandb_api_key` in your inputs folder so Fast-LLM can track your training progress there. You can create a free W&B account if you don't already have one.

## 🚀 Step 7: Launch Training

Alright, the big moment! Let's launch the training run.

=== "Docker"

    If you're on an 8-GPU machine, run the following to kick off training:

    ```bash
    docker run --gpus all -it --rm ghcr.io/servicenow/fast-llm:latest \
        -v ~/inputs:/mnt/inputs \
        -v ~/results:/mnt/results \
        -e PYTHONHASHSEED=0 \
        -e WANDB_API_KEY_PATH=/mnt/inputs/.wandb_api_key \
        torchrun --standalone --nnodes 1 --nproc_per_node=8 --no_python \
        fast-llm train gpt --config /mnt/inputs/fast-llm-config.yaml
    ```

    !!! tip "Customize Your Docker Command"

        * Adjust `--nproc_per_node` based on the number of GPUs you have available.
        * Replace `--gpus all` with `--gpus '"device=0,1,2,3,4,5,6,7"'` if you want to use specific GPUs.
        * Remove `-e WANDB_API_KEY_PATH=/mnt/inputs/.wandb_api_key` if you're not using W&B.

=== "Local Environment"

    If you have 8 GPUs available, run the following to start training:

    ```bash
    export PYTHONHASHSEED=0
    export WANDB_API_KEY_PATH=/mnt/inputs/.wandb_api_key
    torchrun --standalone --nnodes 1 --nproc_per_node=8 --no_python \
        fast-llm train gpt --config /mnt/inputs/fast-llm-config.yaml
    ```

    !!! tip "Customize Your Command"
    
        * Adjust `--nproc_per_node` based on the number of GPUs you have available.
        * Remove `export WANDB_API_KEY_PATH=/mnt/inputs/.wandb_api_key` if you're not using W&B.

=== "Slurm"

    We create a Slurm batch script to run the training job. Save the following as `fast-llm.sbat`:

    ```bash
    #!/bin/bash
    # SBATCH --job-name=fast-llm
    # SBATCH --nodes=1
    # SBATCH --gpus-per-node=8
    # SBATCH --ntasks-per-node=1
    # SBATCH --exclusive
    # SBATCH --output=/mnt/outputs/job_output.log
    # SBATCH --error=/mnt/outputs/job_error.log

    export PYTHONHASHSEED=0
    export WANDB_API_KEY_PATH=/mnt/inputs/.wandb_api_key

    srun \
        --container-image="ghcr.io/servicenow/fast-llm:latest" \
        --container-mounts="${HOME}/inputs:/mnt/inputs,${HOME}/results:/mnt/results" \
        --container-env="PYTHONHASHSEED,WANDB_API_KEY_PATH" \
        --gpus-per-node=$SLURM_GPUS_PER_NODE \
        --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
        bash -c "
            torchrun \
            --standalone \
            --nnodes=\$SLURM_NNODES \
            --nproc_per_node=\$SLURM_GPUS_PER_NODE \
            --no_python \
            fast-llm train gpt \
            --config /mnt/inputs/fast-llm-config.yaml"
    ```

    !!! tip "Customize Your Slurm Script"

        * Change the `--gpus-per-node` value to match the number of GPUs on your node.
        * If you're not using W&B, remove the references to `WANDB_API_KEY_PATH`.

    Submit the job to the Slurm cluster:

    ```bash
    sbatch fast-llm.sbat
    ```

=== "Kubernetes"

    We create a [PyTorchJob](https://www.kubeflow.org/docs/components/training/user-guides/pytorch/) resource with the following configuration and save it as `fast-llm.pytorchjob.yaml`:

    ```yaml
    apiVersion: "kubeflow.org/v1"
    kind: "PyTorchJob"
    metadata:
      name: "fast-llm"
    spec:
      nprocPerNode: "8"
      pytorchReplicaSpecs:
        Master:
          replicas: 1
          restartPolicy: Never
          template:
            spec:
              tolerations:
                - key: nvidia.com/gpu
                  value: "true"
                  operator: Equal
                  effect: NoSchedule
              containers:
                - name: pytorch
                  image: ghcr.io/servicenow/fast-llm:latest
                  resources:
                    limits:
                      nvidia.com/gpu: 8
                      rdma/rdma_shared_device_a: 1
                      memory: "1024Gi"
                      cpu:
                    requests:
                      nvidia.com/gpu: 8
                      rdma/rdma_shared_device_a: 1
                      memory: "1024Gi"
                      cpu: 128
                  command:
                    - /bin/bash
                    - -c
                    - |
                      torchrun --standalone
                               --nnodes=${PET_NNODES} \
                               --nproc_per_node=${PET_NPROC_PER_NODE} \
                               --no_python \
                               fast-llm train gpt \
                               --config /mnt/inputs/fast-llm-config.yaml
                  env:
                    - name: PYTHONHASHSEED
                      value: "0"
                    - name: WANDB_API_KEY_PATH
                      value: "/mnt/inputs/.wandb_api_key"
                  securityContext:
                    capabilities:
                      add:
                        - IPC_LOCK
                  volumeMounts:
                    - mountPath: /mnt/inputs
                      name: fast-llm-inputs
                    - mountPath: /mnt/results
                      name: fast-llm-results
                    - mountPath: /dev/shm
                      name: dshm
              volumes:
                - name: fast-llm-inputs
                  persistentVolumeClaim:
                    claimName: pvc-fast-llm-inputs
                - name: fast-llm-results
                  persistentVolumeClaim:
                    claimName: pvc-fast-llm-results
                - name: dshm
                  emptyDir:
                    medium: Memory
                    sizeLimit: "1024Gi"
    ```

    !!! tip "Customize Your PyTorchJob"

        * Change the `nprocPerNode` value to match the number of GPUs on your node.
        * If you're not using W&B, remove the references to `WARDB_API_KEY_PATH`.

    Submit the job to the Kubernetes cluster:

    ```bash
    kubectl apply -f fast-llm.pytorchjob.yaml
    ```

!!! warning "Python Hash Seed"

    Setting the Python hash seed to 0 ensures consistent, reproducible ordering in hash-dependent operations across processes. Training will fail if this isn't set.

## 📊 Step 8. Track Training Progress

=== "Docker"

    Fast-LLM will log training progress to the console every 10 iterations.

=== "Local Environment"

    Fast-LLM will log training progress to the console every 10 iterations.

=== "Slurm"

    Use `squeue -u $USER` to see the job status.
    Follow `job_output.log` and `job_error.log` in your working directory for logs.
    Fast-LLM will log training progress to those files every 10 iterations.

=== "Kubernetes"

    Use `kubectl get pods` to see the job status.
    Use `kubectl logs fast-llm-master-0` to check the logs.
    Fast-LLM will log training progress to the console every 10 iterations.

You can expect to see the following performance metrics in Fast-LLM's output:

=== "SmolLM2-135M"

    | Performance Metric  | 8x V100-SXM2-32GB[^SmolLM2-V100] | 8x A100-SXM4-80GB[^SmolLM2-A100] | 8x H100-SXM5-80GB[^SmolLM2-H100] |
    |---------------------|---------------------------------:|---------------------------------:|---------------------------------:|
    | tokens/s/GPU        | 18,300                           |                                  | 294,000                          |
    | tflop/s (model)     | 16.7                             |                                  | 268                              |
    | tflop/s (hardware)  | 17.0                             |                                  | 274                              |
    | total training time | 23.3 days                        |                                  | 1.45 days                        |

    [^SmolLM2-V100]:
        `bf16` is not supported on V100 GPUs. Precision was set to `fp16`.
        FlashAttention is not supported on V100 GPUs, so it was disabled.
        Micro-batch size was set to 12.
    [^SmolLM2-A100]:
        Precision was set to `bf16`.
        FlashAttention was enabled.
        Micro-batch size was set to 60.
    [^SmolLM2-H100]:
        Precision was set to `bf16`.
        FlashAttention was enabled.
        Micro-batch size was set to 60.

=== "Llama-3.2-1B"

    | Performance Metric  | 8x V100-SXM2-32GB[^Llama-V100] | 8x A100-SXM4-80GB[^Llama-A100] | 8x H100-SXM5-80GB[^Llama-H100] |
    |---------------------|-------------------------------:|-------------------------------:|-------------------------------:|
    | tokens/s/GPU        | 5,680                          |                                | 66,600                         |
    | tflop/s (model)     | 43.3                           |                                | 508                            |
    | tflop/s (hardware)  | 43.4                           |                                | 510                            |
    | total training time | 12.5 days                      |                                | 1.07 days                      |

    [^Llama-V100]:
        `bf16` is not supported on V100 GPUs. Precision was set to `fp16`.
        FlashAttention is not supported on V100 GPUs, so it was disabled.
        Micro-batch size was set to 4.
    [^Llama-A100]:
        Precision was set to `bf16`.
        FlashAttention was enabled.
        Micro-batch size was set to 20.
    [^Llama-H100]:
        Precision was set to `bf16`.
        FlashAttention was enabled.
        Micro-batch size was set to 20.

If you included the W&B section in your configuration, you can also track your training progress on the Weights & Biases dashboard as well. Follow the link in the console output to view your training run.

## 🛠️ Troubleshooting Basics

Here are some common issues you might encounter and how to address them:

-   **CUDA Out of Memory**: Try lowering the `micro_batch_size` or `sequence_length` in your configuration to fit within available memory.

-   **Underutilized GPU or Low Memory Usage**: If memory usage is low or GPU utilization isn't maxed out, try increasing `micro_batch_size` (to 4, 8, or 16 if memory allows) or extending `sequence_length` (up to 2048, 3072, or 4096, as memory permits). Larger batches and longer sequences help keep GPUs engaged and reduce idle time.

## 🎉 Final Thoughts

And that's it! You've set up, prepped data, chosen a model, configured training, and launched a full training run with Fast-LLM. From here, feel free to tweak the model, try out larger datasets, or scale things up to a multi-node setup if you're on a cluster. Happy training!
