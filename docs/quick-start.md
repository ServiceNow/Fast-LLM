---
title: "Quick Start 🚀"
---

This guide will get you up and running with Fast-LLM on a single machine. Let's train a model and see some results!

## Prerequisites

To follow this guide, you'll need:

-   **Hardware**: At least one NVIDIA GPU with Ampere architecture or newer. For optimal results in this tutorial, we recommend 8 A100 GPUs or better. 🤑
-   **Software**:
    -   **Docker** (if using the Docker setup), or
    -   **Local Environment**: PyTorch 2.2 or later, CUDA 12.1 or later, and APEX AMP (if building from source), or
    -   **Cluster Setup**: Access to a Slurm or Kubernetes cluster.
-   **Time**: The initial setup and training process requires some patience. 😊

## Step 1: Initial Setup 🏗 ️

First, choose your environment. You can use Docker, your local environment, Slurm, or Kubernetes.

=== "Docker"

    You selected Docker for this tutorial. We'll use the Fast-LLM Docker image to train our model, which includes all the necessary dependencies. Grab the pre-built Fast-LLM Docker image:

    ```bash
    docker pull ghcr.io/servicenow/fast-llm:latest
    ```

    Let's also create folders to store our input data and output results:

    ```bash
    mkdir ~/inputs ~/results
    ```

=== "Local Environment"

    You selected to use your local environment to run Fast-LLM. You should have a machine with at least one NVIDIA GPU with Ampere architecture or newer. We need to install Fast-LLM and its dependencies in your environment. Our Fast-LLM docker image already includes all this, and we recommend using it for simplicity and reproducibility. If you still want to install Fast-LLM in your local environment, follow the steps below.

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

    You selected Docker-enabled [Slurm](https://slurm.schedmd.com/) for this tutorial. The Slurm setup requires a Slurm cluster with at least one node and one GPU of Ampere architecture or newer. Slurm will use the `ghcr.io/servicenow/fast-llm:latest` Docker image to train our model. It will need a shared file system for input data and output results. We will assume that your home directory is shared across all nodes.

    Let's create a folder to store our input data and output results in the shared home directory:

    ```bash
    mkdir ~/inputs ~/results
    ```

=== "Kubernetes"

    You selected to use [Kubernetes](https://kubernetes.io/) with [KubeFlow](https://www.kubeflow.org/) for this tutorial. We will use a `PyTorchJob` resource to train our model with the `ghcr.io/servicenow/fast-llm:latest` Docker image and store our input data and output results in shared [persistent volume claims](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) (PVCs). The Kubernetes cluster should have at least one node and one GPU of Ampere architecture or newer.

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

## Step 2: Choose Your Model 🤖

Fast-LLM supports many GPT variants, including (but not limited to) Llama, Mistral, and Mixtral. For this tutorial, let's train a Llama model with data parallelism. You can choose from two models:

=== "SmolLM-135M"

    SmolLM is a smaller, more manageable model with 135M parameters. It's perfect for testing and getting familiar with Fast-LLM. We'll grab its configuration file from Huggingface Hub and save it to our inputs folder:

    === "Docker"

        ```bash
        curl -O https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/config.json
        mv config.json ~/inputs
        ```

    === "Local Environment"

        ```bash
        curl -O https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/config.json
        mv config.json /mnt/inputs
        ```

    === "Slurm"

        ```bash
        curl -O https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/config.json
        mv config.json ~/inputs
        ```

    === "Kubernetes"

        First, download the configuration file to your local machine:

        ```bash
        curl -O https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/config.json
        ```

        Then, create a temporary pod that mounts the inputs PVC, allowing you to copy files to it. Here's a basic YAML configuration for such a pod:

        ```yaml
        apiVersion: v1
        kind: Pod
        metadata:
          name: file-transfer
        spec:
          containers:
            - name: file-transfer-container
              image: ubuntu
              command: ["sleep", "infinity"]
              volumeMounts:
                - mountPath: /mnt/inputs
                  name: inputs
          volumes:
            - name: inputs
              persistentVolumeClaim:
                claimName: pvc-fast-llm-inputs
        ```

        Save this configuration to a file named `file-transfer-pod.yaml` and apply it to your Kubernetes cluster:

        ```bash
        kubectl apply -f file-transfer-pod.yaml
        ```

        Copy the configuration file to the pod:

        ```bash
        kubectl cp config.json file-transfer:/mnt/inputs
        ```

        Finally, clean up the temporary pod and configuration file:

        ```bash
        kubectl delete pod file-transfer
        rm config.json
        ```

=== "Llama-3.2-1B"

    Llama is a larger model with 1B parameters. It's more powerful but requires more resources to train. We'll grab the model from the Huggingface Hub and save it to our inputs folder:

    === "Docker"

        First, sign in to your Hugging Face account:

        ```bash
        pip install huggingface_hub
        huggingface-cli login
        ```

        Then, clone the model:

        ```bash
        git lfs install
        git clone https://huggingface.co/meta-llama/Llama-3.2-1B ~/inputs
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
        git clone https://huggingface.co/meta-llama/Llama-3.2-1B /mnt/inputs
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
        git clone https://huggingface.co/meta-llama/Llama-3.2-1B ~/inputs
        ```
    
    === "Kubernetes"
    
        We need to create a temporary pod that mounts the inputs PVC and allows us to download the model. Here's a basic YAML configuration for such a pod:
    
        ```yaml
        apiVersion: v1
        kind: Pod
        metadata:
          name: clone-model
        spec:
          containers:
            - name: clone-model-container
              image: ubuntu
              command: ["sleep", "infinity"]
              volumeMounts:
                - mountPath: /mnt/inputs
                  name: inputs
          volumes:
            - name: inputs
              persistentVolumeClaim:
                claimName: pvc-fast-llm-inputs
        ```

        Save this configuration to a file named `clone-model-pod.yaml`. Next, apply this configuration to your Kubernetes cluster:

        ```bash
        kubectl apply -f clone-model-pod.yaml
        ```

        Now, enter the pod, log in to your Hugging Face account, and clone the model:

        ```bash
        kubectl exec -it clone-model -- /bin/bash
        pip install huggingface_hub
        huggingface-cli login
        git lfs install
        git clone https://huggingface.co/meta-llama/Llama-3.2-1B /mnt/inputs
        ```

        Finally, clean up the temporary pod, it's no longer needed:

        ```bash
        kubectl delete pod clone-model
        ```

!!! tip "Model Size Matters"

    Smaller models like SmolLM-135M will train relatively quickly, especially if you've only got a few GPUs. But if you're feeling adventurous (and patient), give the larger Llama-3.2-1B a shot!

## Step 3: Prepare the Training Data 📚

For this tutorial, we'll use 9B tokens of text from the [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset. This dataset is a free approximation of the WebText data OpenAI used for GPT-2, and it's perfect for our test run!

=== "SmolLM-135M"

    === "Docker"

        We've got a script that'll download and preprocess the dataset for you. Run it like this:

        ```bash
        docker run -it --rm ghcr.io/servicenow/fast-llm:latest \
            -v ~/inputs:/mnt/inputs \
            python tools/prepare_dataset.py \
            tokenizer_path_or_name="HuggingFaceTB/SmolLM-135M" \
            dataset_name_or_path="openwebtext" \
            dataset_split="train" \
            output_dir="/mnt/inputs" \
            num_processes_load=4 \
            num_processes_map=4 \
            num_processes_save=4 \
            num_tokens_per_shard=100000000
        ```
    
    === "Local Environment"

        Fast-LLM ships with a [script](https://github.com/ServiceNow/Fast-LLM/blob/main/tools/prepare_dataset.py) that downloads and preprocesses the dataset for you. Download and run it like this:

        ```bash
        curl -O https://raw.githubusercontent.com/ServiceNow/Fast-LLM/main/tools/prepare_dataset.py
        python prepare_dataset.py \
            tokenizer_path_or_name="HuggingFaceTB/SmolLM-135M" \
            dataset_name_or_path="openwebtext" \
            dataset_split="train" \
            output_dir="/mnt/inputs" \
            num_processes_load=4 \
            num_processes_map=4 \
            num_processes_save=4 \
            num_tokens_per_shard=100000000
        ```
    
    === "Slurm"

        Fast-LLM has got you covered with a script that'll download and preprocess the dataset for you. Run it like this:

        ```bash
        sbatch <<EOF
        #!/bin/bash
        # SBATCH --nodes=1
        # SBATCH --ntasks-per-node=1
        # SBATCH --exclusive
        # SBATCH --output=/mnt/outputs/job_output.log
        # SBATCH --error=/mnt/outputs/job_error.log

        srun \
            --container-image="ghcr.io/servicenow/fast-llm:latest" \
            --container-mounts="${HOME}/inputs:/mnt/inputs,${HOME}/results:/mnt/results" \
            --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
            bash -c "
                python tools/prepare_dataset.py \
                    tokenizer_path_or_name='HuggingFaceTB/SmolLM-135M' \
                    dataset_name_or_path='openwebtext' \
                    dataset_split='train' \
                    output_dir='/mnt/inputs' \
                    num_processes_load=4 \
                    num_processes_map=4 \
                    num_processes_save=4 \
                    num_tokens_per_shard=100000000"
        EOF
        ```

        You can follow the job's progress by running `squeue -u $USER` and checking the logs in `~/results/job_output.log` and `~/results/job_error.log`.
    
    === "Kubernetes"

        Fast-LLM comes with a script that'll download and preprocess the dataset for you. We will run this script in a Kubernetes job. Here's a basic configuration for the job:

        ```yaml
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: prepare-dataset
        spec:
          template:
            spec:
              containers:
                - name: prepare-dataset
                  image: ghcr.io/servicenow/fast-llm:latest
                  command: ["python", "tools/prepare_dataset.py"]
                  args:
                    - tokenizer_path_or_name=HuggingFaceTB/SmolLM-135M
                    - dataset_name_or_path=openwebtext
                    - dataset_split=train
                    - output_dir=/mnt/inputs
                    - num_processes_load=4
                    - num_processes_map=4
                    - num_processes_save=4
                    - num_tokens_per_shard=100000000
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

        Save this configuration to a file named `prepare-dataset-job.yaml` and apply it to your Kubernetes cluster:

        ```bash
        kubectl apply -f prepare-dataset-job.yaml
        ```

        You can follow the job's progress by running `kubectl get pods` and checking the logs with `kubectl logs prepare-dataset`.

=== "Llama-3.2-1B"

    === "Docker"

        We've got a script that'll download and preprocess the dataset for you. Run it like this:

        ```bash
        docker run -it --rm ghcr.io/servicenow/fast-llm:latest \
            -v ~/inputs:/mnt/inputs \
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
    
    === "Local Environment"

        Fast-LLM ships with a [script](https://github.com/ServiceNow/Fast-LLM/blob/main/tools/prepare_dataset.py) that downloads and preprocesses the dataset for you. Download and run it like this:

        ```bash
        curl -O https://raw.githubusercontent.com/ServiceNow/Fast-LLM/main/tools/prepare_dataset.py
        python prepare_dataset.py \
            tokenizer_path_or_name="meta-llama/Llama-3.2-1B" \
            dataset_name_or_path="openwebtext" \
            dataset_split="train" \
            output_dir="/mnt/inputs" \
            num_processes_load=4 \
            num_processes_map=4 \
            num_processes_save=4 \
            num_tokens_per_shard=100000000
        ```

    === "Slurm"

        Fast-LLM has got you covered with a script that'll download and preprocess the dataset for you. Run it like this:

        ```bash
        sbatch <<EOF
        #!/bin/bash
        # SBATCH --nodes=1
        # SBATCH --ntasks-per-node=1
        # SBATCH --exclusive
        # SBATCH --output=/mnt/outputs/job_output.log
        # SBATCH --error=/mnt/outputs/job_error.log

        srun \
            --container-image="ghcr.io/servicenow/fast-llm:latest" \
            --container-mounts="${HOME}/inputs:/mnt/inputs,${HOME}/results:/mnt/results" \
            --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
            bash -c "
                python tools/prepare_dataset.py \
                    tokenizer_path_or_name='meta-llama/Llama-3.2-1B' \
                    dataset_name_or_path='openwebtext' \
                    dataset_split='train' \
                    output_dir='/mnt/inputs' \
                    num_processes_load=4 \
                    num_processes_map=4 \
                    num_processes_save=4 \
                    num_tokens_per_shard=100000000"
        EOF
        ```

        You can follow the job's progress by running `squeue -u $USER` and checking the logs in `~/results/job_output.log` and `~/results/job_error.log`.

    === "Kubernetes"

        Fast-LLM comes with a script that'll download and preprocess the dataset for you. We will run this script in a Kubernetes job. Here's a basic configuration for the job:

        ```yaml
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: prepare-dataset
        spec:
          template:
            spec:
              containers:
                - name: prepare-dataset
                  image: ghcr.io/servicenow/fast-llm:latest
                  command: ["python", "tools/prepare_dataset.py"]
                  args:
                    - tokenizer_path_or_name=meta-llama/Llama-3.2-1B
                    - dataset_name_or_path=openwebtext
                    - dataset_split=train
                    - output_dir=/mnt/inputs
                    - num_processes_load=4
                    - num_processes_map=4
                    - num_processes_save=4
                    - num_tokens_per_shard=100000000
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

        Save this configuration to a file named `prepare-dataset-job.yaml` and apply it to your Kubernetes cluster:

        ```bash
        kubectl apply -f prepare-dataset-job.yaml
        ```

        You can follow the job's progress by running `kubectl get pods` and checking the logs with `kubectl logs prepare-dataset`.

!!! info "What's Happening Here?"

    The `prepare_dataset.py` script will grab the OpenWebText data from the Huggingface Hub, tokenize it, and save it in 91 shards of 100M tokens each to the input folder. Expect around 2 hours for the whole thing to finish, mainly due to tokenization. If you've got more CPU cores, try upping `num_processes_*` to speed things up.

!!! tip "Use a Smaller Dataset for Testing"

    If you're just testing things out, you can also use a smaller dataset. Replace `openwebtext` with `stas/openwebtext-10k` to use a small subset representing the first 10K records from the original dataset. This will speed up the process and let you see how things work without waiting for hours.

## Step 4: Configure Fast-LLM ⚙️

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
      path: /mnt/inputs/fast_llm_dataset.json  # (6)!
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
      path: /mnt/inputs
      load_weights: no  # (11)!
    model:
      multi_stage:
        zero_stage: null  # (12)!
      distributed:
        training_dtype: bf16  # (13)!
    run:
      experiment_dir: /mnt/results
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
      path: /mnt/inputs/fast_llm_dataset.json  # (6)!
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
      path: /mnt/inputs
      load_weights: yes  # (11)!
    model:
      multi_stage:
        zero_stage: null  # (12)!
      distributed:
        training_dtype: bf16  # (13)!
    run:
      experiment_dir: /mnt/results
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

    Adjust `--nproc_per_node` based on the number of GPUs you have available.
    Replace `--gpus all` with `--gpus '"device=0,1,2,3,4,5,6,7"'` if you want to use specific GPUs.
    Remove `-e WANDB_API_KEY_PATH=/mnt/inputs/.wandb_api_key` if you're not using W&B.

=== "Local Environment"

    ```bash
    export PYTHONHASHSEED=0
    export WANDB_API_KEY_PATH=/mnt/inputs/.wandb_api_key
    torchrun --standalone --nnodes 1 --nproc_per_node=8 --no_python \
        fast-llm train gpt --config /mnt/inputs/fast-llm-config.yaml
    ```

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

    Change the `--gpus-per-node` value to match the number of GPUs on your node.
    If you're not using W&B, remove the references to `WARDB_API_KEY_PATH`.

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

    Change the `nprocPerNode` value to match the number of GPUs on your node. If you're not using W&B, remove the references to `WARDB_API_KEY_PATH`.

    Submit the job to the Kubernetes cluster:

    ```bash
    kubectl apply -f fast-llm.pytorchjob.yaml
    ```

!!! warning "Python Hash Seed"

    Setting the Python hash seed to 0 ensures consistent, reproducible ordering in hash-dependent operations across processes. Training will fail if this isn't set.

## Step 8. Track Training Progress 📊

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

You can expect to see the following throughput:

=== "SmolLM-135M"

    | Metric              | A100-80GB    | H100         |
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

## Final Thoughts

And that's it! You've set up, prepped data, chosen a model, configured training, and launched a full training run with Fast-LLM. From here, feel free to tweak the model, try out larger datasets, or scale things up to a multi-node setup if you're on a cluster. Happy training! 🚀
