---
title: "Quick Start"
---

This guide will get you up and running with Fast-LLM. Let's train a model and see some results!

## Prerequisites

To follow this guide, you'll need:

-   **Hardware**: At least one NVIDIA GPU, preferably with Ampere architecture or newer. Note that this tutorial is designed for 80 GB A100s or H100 GPUs, and some adjustments are needed to run it with less memory or an earlier architecture.
-   **Software**: Depending on your setup, you'll need one of the following:
    -   **Docker**: If you're using the prebuilt Docker image on your local machine.
    -   **Python 3.10**: If you're setting up a custom environment (virtual environment, bare-metal, etc.) on your local machine.
    -   **Cluster Setup**: Access to a Docker-enabled Slurm cluster or to a Kubernetes cluster with Kubeflow if you're using those environments.

## 🏗 Step 1: Initial Setup

First, create a working directory for this tutorial:

```bash
mkdir ./fast-llm-tutorial
```

We'll use this directory to store all the files and data needed for training.

Now, select the compute environment that matches your setup or preferred workflow. Once you select an environment, all sections of this guide will adapt to provide instructions specific to your choice:

=== "Prebuilt Docker"

    Use a preconfigured Docker container with the Fast-LLM image, which includes all the required software and dependencies. Run the following command to pull the image and start a container:

    ```bash
    docker run --gpus all -it --rm \
        -v $(pwd)/fast-llm-tutorial:/app/fast-llm-tutorial \
        ghcr.io/servicenow/fast-llm:latest \
        bash
    ```

    Replace `--gpus all` with `--gpus '"device=0,1,2,3,4,5,6,7"'` etc. if you want to use specific GPUs.

    Once inside the container, all commands from this guide can be executed as-is. The `fast-llm-tutorial` directory is mounted inside the container at `/app/fast-llm-tutorial`, so any files saved there will persist and be accessible on your host machine as well.

=== "Custom Installation"

    If you prefer not to use the prebuilt Docker image or already have an environment you'd like to use (e.g., a custom Docker image, virtual environment, or bare-metal setup), follow these steps to install the necessary software and dependencies:

    1.  **Ensure Python 3.10**:
        Install Python 3.10 (or later) if it's not already available on your system. For a Python virtual environment, run:

        ```bash
        python3.10 -m venv ./fast-llm-tutorial/venv
        source ./fast-llm-tutorial/venv/bin/activate
        pip install --upgrade pip
        ```

        You can deactivate the virtual environment later with `deactivate`.

    2.  **Verify CUDA Installation**:
        Make sure [CUDA](https://developer.nvidia.com/about-cuda) 12.1 or later is installed in your environment. Verify with:

        ```bash
        nvcc --version
        ```

        If CUDA is not installed or the version is incorrect, follow the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) to set it up.

    3.  **Pre-install PyTorch and pybind11**:
        Install PyTorch and pybind11 to meet Fast-LLM's requirements:

        ```bash
        pip install pybind11 "torch>=2.2.2"
        ```

    4.  **Install NVIDIA APEX**:
        Fast-LLM uses certain kernels from [APEX](https://github.com/NVIDIA/apex). Follow the installation instructions on their GitHub page, ensuring you use the `--cuda_ext` and `--fast_layer_norm` options to install all kernels supported by Fast-LLM:

        ```bash
        git clone https://github.com/NVIDIA/apex ./fast-llm-tutorial/apex
        pushd ./fast-llm-tutorial/apex
        pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --config-settings "--build-option=--fast_layer_norm" ./
        popd
        ```

    5.  **Install Fast-LLM and Dependencies**:
        Finally, install Fast-LLM along with its remaining dependencies, including [FlashAttention-2](https://github.com/Dao-AILab/flash-attention):

        ```bash
        pip install --no-build-isolation "git+https://github.com/ServiceNow/Fast-LLM.git#egg=fast_llm[CORE,OPTIONAL,DEV]"
        ```

    6.  **Verify the Installation**:
        Confirm the setup with the following commands:

        ```bash
        python -c "import torch; print(torch.cuda.is_available())"
        python -c "from amp_C import *"
        python -c "import flash_attn; print(flash_attn.__version__)"
        python -c "import fast_llm; print(fast_llm.__version__)"
        ```

    If you made it this far without any errors, your local environment is ready to run Fast-LLM.

=== "Slurm"

    Use Docker-enabled [Slurm](https://slurm.schedmd.com/) for this tutorial. The `ghcr.io/servicenow/fast-llm:latest` Docker image will be pulled and run on the compute nodes. Ensure the `fast-llm-tutorial` directory is accessible across all nodes (e.g., via a shared filesystem like NFS).

=== "Kubeflow"

    Use [Kubernetes](https://kubernetes.io/) with [Kubeflow](https://www.kubeflow.org/) and a `PyTorchJob` resource to train our model using the `ghcr.io/servicenow/fast-llm:latest` Docker image. We'll copy the configuration files and dataset to shared [persistent volume claims](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) (PVCs) to ensure all nodes have access to the same data. Follow these steps:

    1.  **Create a Persistent Volume Claim (PVC)**

        Create a PVC named `pvc-fast-llm-tutorial` to store input data and output results:

        ```yaml
        kubectl apply -f - <<EOF
        apiVersion: "v1"
        kind: "PersistentVolumeClaim"
        metadata:
          name: "pvc-fast-llm-tutorial"
        spec:
          storageClassName: local-path  # (1)!
          accessModes:
            - ReadWriteMany
          resources:
            requests:
              storage: 100Gi  # (2)!
        EOF
        ```

        1. Replace with your cluster's StorageClassName.
        2. Adjust the storage size as needed.

        !!! note "StorageClassName"

            Replace `local-path` with the appropriate `StorageClassName` for your Kubernetes cluster. Consult your cluster admin or documentation if unsure.

    2.  **Set Up a Temporary Pod for Data Management**

        Create a temporary pod to manage input data and results:

        ```yaml
        kubectl apply -f - <<EOF
        apiVersion: v1
        kind: Pod
        metadata:
          name: pod-fast-llm-tutorial
        spec:
          containers:
            - name: fast-llm-tutorial-container
              image: ghcr.io/servicenow/fast-llm:latest
              command: ["sleep", "infinity"]
              volumeMounts:
                - mountPath: /app/fast-llm-tutorial
                  name: fast-llm-tutorial
          volumes:
            - name: fast-llm-tutorial
              persistentVolumeClaim:
                claimName: pvc-fast-llm-tutorial
        EOF
        ```

        !!! note "Purpose of the Temporary Pod"

            This pod ensures you have an interactive container for managing input data and retrieving results. Use `kubectl exec` to interact with it:

            ```bash
            kubectl exec -it pod-fast-llm-tutorial -- bash
            ```

            Use `kubectl cp` to copy files between the pod and your local machine:

            ```bash
            kubectl cp ./fast-llm-tutorial pod-fast-llm-tutorial:/app
            ```

## 🤖 Step 2: Choose Your Training Configuration

This guide offers two training configurations:

=== "Small"

    For a quick, single-node setup and immediate results to test Fast-LLM with a smaller model. Ideal for getting started and understanding the basics. It's the "hello world" of Fast-LLM.

=== "Big"

    For a more advanced setup with more data and larger models to explore Fast-LLM's full capabilities. This configuration requires more resources and time to complete, but it prepares you for production-like workloads.

Choose based on your goals for this tutorial.

## 📥 Step 3: Download the Pretrained Model

=== "Small"

    For the small configuration, we'll use a SmolLM2 model configuration with 135M parameters, which is fast to train. Run the following commands to download the model configuration and tokenizer:

    ```bash
    git lfs install
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/HuggingFaceTB/SmolLM2-135M ./fast-llm-tutorial/pretrained-model
    ```

=== "Big"

    For the big configuration, we'll use a Llama model with 8B parameters. We'll grab the model from the Huggingface Hub and save it to our inputs folder.

    !!! note "Access Required"

        Meta gates access to their Llama models. You need to request access to the model from Meta before you can download it at https://huggingface.co/meta-llama/Llama-3.1-8B. You'll need to authenticate with your Hugging Face account to download the model:

        ```bash
        pip install huggingface_hub
        huggingface-cli login
        ```

        When asked for whether to use this as git credentials, answer in the affirmative.
    
    ```
    git lfs install
    git clone https://huggingface.co/meta-llama/Llama-3.1-8B ./fast-llm-tutorial/pretrained-model
    ```

## 📚 Step 3: Prepare the Training Data

For this tutorial, we'll use text from the [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset. This dataset is a free approximation of the WebText data OpenAI used for GPT-2, and it's perfect for our test run!

Create a configuration file for the dataset preparation. Copy the following content:

=== "Small"

    ```yaml
    output_path: fast-llm-tutorial/dataset

    loading_workers: 16  # (1)!
    tokenize_workers: 16
    saving_workers: 16

    dataset:
      path: openwebtext
      split: "train[:10000]"  # (2)!
      trust_remote_code: true

    tokenizer:
      path: fast-llm-tutorial/pretrained-model
    ```

    1. Processing speed scales linearly with the number of CPUs.
    2. We're [slicing](https://huggingface.co/docs/datasets/v1.11.0/splits.html) the dataset to the first 10K records of the OpenWebText dataset to speed up the process. If you want to use the full dataset, set the `split` to `train`.

=== "Big"

    ```yaml
    output_path: fast-llm-tutorial/dataset

    loading_workers: 128  # (1)!
    tokenize_workers: 128
    saving_workers: 128

    dataset:
      path: openwebtext
      split: train
      trust_remote_code: true

    tokenizer:
      path: fast-llm-tutorial/pretrained-model
    ```

    1. Processing speed scales linearly with the number of CPUs.

Save it as `./fast-llm-tutorial/prepare-config.yaml`.

Fast-LLM ships with a `prepare` command that will download and preprocess the dataset for you.

=== "Prebuilt Docker"

    Run data preparation with the following command:

    ```bash
    fast-llm prepare gpt_memmap --config fast-llm-tutorial/prepare-config.yaml
    ```

=== "Custom Installation"

    Run data preparation with the following command:

    ```bash
    fast-llm prepare gpt_memmap --config fast-llm-tutorial/prepare-config.yaml
    ```

=== "Slurm"

    Run data preparation with the following command:

    ```bash
    sbatch <<EOF
    #!/bin/bash
    # SBATCH --job-name=fast-llm-prepare
    # SBATCH --nodes=4
    # SBATCH --ntasks-per-node=1
    # SBATCH --exclusive
    # SBATCH --output=/app/fast-llm-tutorial/prepare-output.log
    # SBATCH --error=/app/fast-llm-tutorial/prepare-error.log

    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    MASTER_PORT=8001

    export PYTHONHASHSEED=0

    srun \
        --container-image="ghcr.io/servicenow/fast-llm:latest" \
        --container-mounts="$(pwd)/fast-llm-tutorial:/app/fast-llm-tutorial" \
        --container-env="PYTHONHASHSEED" \
        --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
        bash -c "
            torchrun --rdzv_backend=static \
                     --rdzv_id=0 \
                     --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
                     --node_rank=\$SLURM_NODEID \
                     --nproc_per_node=\$SLURM_NTASKS_PER_NODE \
                     --nnodes=\$SLURM_NNODES:\$SLURM_NNODES \
                     --max_restarts=0 \
                     --rdzv_conf=timeout=3600 \
                     --no_python \
                     fast-llm prepare gpt_memmap \
                     --config fast-llm-tutorial/prepare-config.yaml"
    EOF
    ```

    You can follow the job's progress by running `squeue -u $USER` and checking the logs in `fast-llm-tutorial/prepare-output.log` and `fast-llm-tutorial/prepare-error.log`, respectively.

=== "Kubeflow"

    Copy the files to the shared PVC if they're not already there:

    ```bash
    kubectl cp ./fast-llm-tutorial pod-fast-llm-tutorial:/app
    ```

    Then, run data preparation with the following command:

    ```yaml
    kubectl apply -f - <<EOF
    apiVersion: "kubeflow.org/v1"
    kind: "PyTorchJob"
    metadata:
      name: "fast-llm-prepare"
    spec:
      nprocPerNode: "1"
      pytorchReplicaSpecs:
        Master:
          replicas: 1
          restartPolicy: Never
          template:
            spec:
              containers:
                - name: pytorch
                  image: ghcr.io/servicenow/fast-llm:latest
                  resources:
                    limits:
                      memory: "1024Gi"
                      cpu:
                    requests:
                      memory: "1024Gi"
                      cpu: 128
                  command:
                    - /bin/bash
                    - -c
                    - |
                      torchrun --rdzv_backend=static \
                               --rdzv_id=0 \
                               --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
                               --node_rank=${RANK} \
                               --nproc_per_node=${PET_NPROC_PER_NODE} \
                               --nnodes=${PET_NNODES}:${PET_NNODES} \
                               --max_restarts=0 \
                               --rdzv_conf=timeout=3600 \
                               --no_python \
                               fast-llm prepare gpt_memmap \
                               --config fast-llm-tutorial/prepare-config.yaml
                  env:
                    - name: PYTHONHASHSEED
                      value: "0"
                  securityContext:
                    capabilities:
                      add:
                        - IPC_LOCK
                  volumeMounts:
                    - mountPath: /app/fast-llm-tutorial
                      name: fast-llm-tutorial
                    - mountPath: /dev/shm
                      name: dshm
              volumes:
                - name: fast-llm-tutorial
                  persistentVolumeClaim:
                    claimName: pvc-fast-llm-tutorial
                - name: dshm
                  emptyDir:
                    medium: Memory
                    sizeLimit: "1024Gi"
        Worker:
          replicas: 3
          restartPolicy: Never
          template:
            spec:
              containers:
                - name: pytorch
                  image: ghcr.io/servicenow/fast-llm:latest
                  resources:
                    limits:
                      memory: "1024Gi"
                      cpu:
                    requests:
                      memory: "1024Gi"
                      cpu: 128
                  command:
                    - /bin/bash
                    - -c
                    - |
                      torchrun --rdzv_backend=static \
                               --rdzv_id=0 \
                               --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
                               --node_rank=${RANK} \
                               --nproc_per_node=${PET_NPROC_PER_NODE} \
                               --nnodes=${PET_NNODES}:${PET_NNODES} \
                               --max_restarts=0 \
                               --rdzv_conf=timeout=3600 \
                               --no_python \
                               fast-llm prepare gpt_memmap \
                               --config fast-llm-tutorial/prepare-config.yaml
                  env:
                    - name: PYTHONHASHSEED
                      value: "0"
                  securityContext:
                    capabilities:
                      add:
                        - IPC_LOCK
                  volumeMounts:
                    - mountPath: /app/fast-llm-tutorial
                      name: fast-llm-tutorial
                    - mountPath: /dev/shm
                      name: dshm
              volumes:
                - name: fast-llm-tutorial
                  persistentVolumeClaim:
                    claimName: pvc-fast-llm-tutorial
                - name: dshm
                  emptyDir:
                    medium: Memory
                    sizeLimit: "1024Gi"
    EOF
    ```

    You can follow the job's progress by running `kubectl get pods` and checking the logs with `kubectl logs fast-llm-prepare-master-0`.

## ⚙️ Step 4: Configure Fast-LLM

Next, we'll create a configuration file for Fast-LLM.

!!! warning "FlashAttention"

    Fast-LLM uses FlashAttention by default. If you're using Volta GPUs, you must disable FlashAttention by setting `use_flash_attention: no` in the configuration file, as shown below.

!!! warning "Micro-Batch Size"

    The `micro_batch_size` in the configuration below is optimized for 80GB GPUs. If you're using GPUs with less memory, you will need to lower this value.

Save the following as `fast-llm-tutorial/train-config.yaml`:

=== "Small"

    ```yaml
    training:
      train_iters: 1000  # (1)!
      logs:
        interval: 10
      validation:
        iterations: 25
        interval: 1000
      export:  # (2)!
        format: llama
        interval: 1000
      wandb:  # (3)!
        # project_name: fast-llm-tutorial
        # group_name: Small
        # entity_name: ???
    batch:
      micro_batch_size: 60  # (4)!
      sequence_length: 1024
      batch_size: 480  # (5)!
    data:
      format: file
      path: fast-llm-tutorial/dataset/fast_llm_dataset.json  # (6)!
      split: [9, 1, 0]  # (7)!
    optimizer:
      learning_rate:
        base: 6.0e-04
    pretrained:
      format: llama  # (8)!
      path: fast-llm-tutorial/pretrained_model
      model_weights: no  # (9)!
    model:
      base_model:
        transformer:
          use_flash_attention: yes  # (10)!
      distributed:
        training_dtype: bf16  # (11)!
    run:
      experiment_dir:  fast-llm-tutorial/experiment
    ```

    1.  For the small run, we'll stop after 1000 iterations.
    2.  A Llama model will be saved in Hugging Face format to experiment directory at the end of the small run.
    3.  Entirely optional, but it's a good idea to track your training progress with Weights & Biases. Replace `???` with your own W&B entity name. If you don't want to use W&B, just ignore this section.
    3.  Adjust the number of sequences per GPU based on GPU memory. For SmolLM2-135M at 1024 sequenced length and a 80GB GPU, a `micro_batch_size` of 60 should work well.
    4.  Must be divisible by the number of GPUs and the `micro_batch_size`. At 1024 tokens per sequence, 480 corresponds to about 500,000 tokens per batch.
    5.  Location of the dataset metadata file generated in Step 4.
    6.  90% train, 10% validation, 0% test. These settings need to be adjusted based on the size of your dataset.
    7.  Format of the pretrained model. Since SmolLM is a Llama model, we set this to `llama`.
    8.  We'll train SmolLM2-135M from scratch. You can set to `yes` to continue training from a checkpoint (if you put one in the model directory).
    9.  By default, Fast-LLM uses FlashAttention for faster training. If you're using Volta GPUs, set this to `no`.
    10. `bf16` (bfloat16, or Brain Floating Point 16) is supported on Ampere GPUs and higher. On Volta GPUs, use `fp16` (half-precision floating point) for training instead of `bf16`.

=== "Big"

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
        # project_name: fast-llm-tutorial
        # group_name: Big
        # entity_name: ???
    batch:
      micro_batch_size: 4  # (4)!
      sequence_length: 4096
      batch_size: 480  # (5)!
    data:
      format: file
      path: fast-llm-tutorial/dataset/fast_llm_dataset.json  # (6)!
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
      path: fast-llm-tutorial/pretrained_model
      model_weights: yes  # (11)!
    model:
      base_model:
        transformer:
          use_flash_attention: yes  # (12)!
        cross_entropy_impl: fused  # (13)!
      multi_stage:
        zero_stage: 2  # (14)!
      distributed:
        training_dtype: bf16  # (15)!
    run:
      experiment_dir: fast-llm-tutorial/experiment
    ```

    1.  Total number of training tokens will be approximately 200B: 100,000 iterations * 480 * 4096 tokens per batch.
    2.  A Llama model will be saved in Hugging Face format to `~/results` directory every 20,000 iterations.
    3.  Entirely optional, but it's a good idea to track your training progress with Weights & Biases. Replace `???` with your own W&B entity name. If you don't want to use W&B, just ignore this section.
    4.  Adjust the number of sequences per GPU based on GPU memory. Considering a 4k token sequence length and 80GB GPUs, a `micro_batch_size` of 4 should work well.
    5.  Must be divisible by the number of GPUs and the `micro_batch_size`. At 1024 tokens per sequence, 480 corresponds to about 500,000 tokens per batch.
    6.  Location of the dataset metadata file generated in Step 4.
    7.  99% train, 1% validation, 0% test. These settings need to be adjusted based on the size of your dataset. If you're using a smaller dataset, you need to increase the validation split.
    8.  These are good default optimizer settings for training models.
    9.  We are using a cosine decay schedule with linear warmup. After reaching the peak learning rate `base` at `warmup_iterations`, the learning rate will decay to `minimum` at `decay_iterations`, following a cosine curve. The minimum learning rate should be 1/10th of the base learning rate per Chinchilla.
    10.  Format of the pretrained model. Since it's a Llama model, we set this to `llama`.
    11.  We want to continue training Llama-3.1-8B from a checkpoint. If you're training from scratch, set this to `no`.
    12.  By default, Fast-LLM uses FlashAttention for faster training. If you're using Volta GPUs, set this to `no`.
    13.  Configure Fast-LLM to use the fused cross-entropy loss implementation rather than the default Triton implementation for models with a large vocabulary size such as Llama-3.1-8B. This avoids issues with block size limitations in our current Triton code.
    14.  We are using ZeRO stage 2 for this tutorial. You can set this to `1`, `2`, or `3` for ZeRO-1, ZeRO-2, or ZeRO-3, respectively.
    15.  `bf16` (bfloat16, or Brain Floating Point 16) is supported on Ampere GPUs and higher. On Volta GPUs, use `fp16` (half-precision floating point) for training instead of `bf16`.

## 🔑 (Optional) Step 6: Add Your Weights & Biases API Key

If you included the W&B section in your configuration, you'll need to add your API key. Save it to `./fast-llm-tutorial/.wandb_api_key` and use the `WANDB_API_KEY_PATH` environment variable as shown in the training command.

## 🚀 Step 7: Launch Training

Alright, the big moment! Let's launch the training run.

!!! warning "Python Hash Seed"

    The Python hash seed must be set to 0 to ensure consistent, reproducible ordering in hash-dependent operations across processes. Training will fail if this isn't set.

=== "Prebuilt Docker"

    If you have 8 GPUs available, run the following to start training:

    ```bash
    export PYTHONHASHSEED=0
    # export WANDB_API_KEY_PATH=/app/fast-llm-tutorial/.wandb_api_key
    torchrun --standalone --nnodes 1 --nproc_per_node=8 --no_python \
        fast-llm train gpt --config fast-llm-tutorial/train-config.yaml
    ```

=== "Custom Installation"

    If you have 8 GPUs available, run the following to start training:

    ```bash
    export PYTHONHASHSEED=0
    # export WANDB_API_KEY_PATH=/app/fast-llm-tutorial/.wandb_api_key
    torchrun --standalone --nnodes 1 --nproc_per_node=8 --no_python \
        fast-llm train gpt --config fast-llm-tutorial/train-config.yaml
    ```

=== "Slurm"

    If you have 4 nodes with 8 GPUs each, run the following to start training:

    ```bash
    sbatch <<EOF
    #!/bin/bash
    # SBATCH --job-name=fast-llm-train
    # SBATCH --nodes=4
    # SBATCH --gpus-per-node=8
    # SBATCH --ntasks-per-node=1
    # SBATCH --exclusive
    # SBATCH --output=/app/fast-llm-tutorial/train-output.log
    # SBATCH --error=/app/fast-llm-tutorial/train-error.log

    export PYTHONHASHSEED=0
    export WANDB_API_KEY_PATH=/app/fast-llm-tutorial/.wandb_api_key
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_DEBUG=INFO

    srun \
        --container-image="ghcr.io/servicenow/fast-llm:latest" \
        --container-mounts="$(pwd)/fast-llm-tutorial:/app/fast-llm-tutorial" \
        --container-env="PYTHONHASHSEED,WANDB_API_KEY_PATH,TORCH_NCCL_ASYNC_ERROR_HANDLING,NCCL_DEBUG" \
        --gpus-per-node=$SLURM_GPUS_PER_NODE \
        --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
        bash -c "
            torchrun --rdzv_backend=static \
                     --rdzv_id=0 \
                     --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
                     --node_rank=\$SLURM_NODEID \
                     --nproc_per_node=\$SLURM_GPUS_PER_NODE \
                     --nnodes=\$SLURM_NNODES \
                     --max_restarts=0 \
                     --rdzv_conf=timeout=3600 \
                     --no_python \
                     fast-llm train gpt \
                     --config fast-llm-tutorial/train-config.yaml"
    EOF 
    ```

=== "Kubeflow"

    Copy the configuration file to the shared PVC:

    ```bash
    kubectl cp ./fast-llm-tutorial/train-config.yaml pod-fast-llm-tutorial:/app/fast-llm-tutorial
    ```

    If you have 4 nodes with 8 GPUs each, run the following to start training:

    ```yaml
    kubectl apply -f - <<EOF
    apiVersion: "kubeflow.org/v1"
    kind: "PyTorchJob"
    metadata:
      name: "fast-llm-train"
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
                      torchrun --rdzv_backend=static \
                               --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
                               --node_rank=${RANK} \
                               --nproc_per_node=${PET_NPROC_PER_NODE} \
                               --nnodes=${PET_NNODES} \
                               --max_restarts=0 \
                               --rdzv_conf=timeout=3600 \
                               --no_python \
                               fast-llm train gpt \
                               --config fast-llm-tutorial/train-config.yaml
                  env:
                    - name: PYTHONHASHSEED
                      value: "0"
                    - name: WANDB_API_KEY_PATH
                      value: "/app/fast-llm-tutorial/.wandb_api_key"
                    - name: TORCH_NCCL_ASYNC_ERROR_HANDLING
                      value: "1"
                    - name: NCCL_DEBUG
                      value: "INFO"
                  securityContext:
                    capabilities:
                      add:
                        - IPC_LOCK
                  volumeMounts:
                    - mountPath: /app/fast-llm-tutorial
                      name: fast-llm-inputs
                    - mountPath: /dev/shm
                      name: dshm
              volumes:
                - name: fast-llm-inputs
                  persistentVolumeClaim:
                    claimName: pvc-fast-llm-tutorial
                - name: dshm
                  emptyDir:
                    medium: Memory
                    sizeLimit: "1024Gi"
    EOF
    ```

## 📊 Step 8. Track Training Progress

=== "Prebuilt Docker"

    Fast-LLM will log training progress to the console every 10 iterations.

    You can cancel training at any time by pressing `Ctrl+C` in the terminal.

=== "Custom Installation"

    Fast-LLM will log training progress to the console every 10 iterations.

    You can cancel training at any time by pressing `Ctrl+C` in the terminal.

=== "Slurm"

    Use `squeue -u $USER` to see the job status.
    Follow `train-output.log` and `train-error.log` in your working directory for logs.
    Fast-LLM will log training progress to those files every 10 iterations.

    You can cancel training by running `scancel <job_id>`.

=== "Kubeflow"

    Use `kubectl get pods` to see the job status.
    Use `kubectl logs fast-llm-train-master-0` to check the logs.
    Fast-LLM will log training progress to the console every 10 iterations.

    You can cancel training by deleting the PyTorchJob:

    ```bash
    kubectl delete pytorchjob fast-llm-train
    ```

    !!! note "Cleaning Up Resources"

        Delete the data management pod and PVC if you're finished with the tutorial:

        ```bash
        kubectl delete pod pod-fast-llm-tutorial
        kubectl delete pvc pvc-fast-llm-tutorial
        ```

        This will shut down the temporary pod and remove the PVC with all its contents.

You can expect to see the following performance metrics in Fast-LLM's output:

=== "Small"

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

=== "Big"

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

## 🎉 Final Thoughts

And that's it! You've set up, prepped data, chosen a model, configured training, and launched a full training run with Fast-LLM. From here, feel free to tweak the model, try out larger datasets, or scale things up to larger clusters. The sky's the limit!

Happy training!
