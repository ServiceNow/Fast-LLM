---
title: Preparing Data for Training
---

If you're wondering if you can use your favorite dataset from Huggingface Datasets with Fast-LLM, the answer is a resounding yes! Let's see how to do that.

## Prerequisites

For this guide, you would need:

-   **Hardware**: Just a machine with CPUs will do. But having a large numbers of CPUs and nodes helps distribute the data preparation job and significantly speed things up.

-   **Software**: Depending on your setup, you'll need one of the following:
    -   **Docker**: If you're using the prebuilt Docker image on your local machine.
    -   **Python 3.10**: If you're setting up a custom environment (virtual environment, bare-metal, etc.) on your local machine.
    -   **Cluster Setup**: Access to a Docker-enabled Slurm cluster or to a Kubernetes cluster with Kubeflow if you're using those environments.

## 📚 Step 1: Download the dataset from Huggingface

We'll use [the-stack](https://huggingface.co/datasets/bigcode/the-stack) dataset for this tutorial, which is one of the largest collections of permissively-licensed source code files.

First, set `HF_HOME` to your Huggingface cache folder:

```bash
export HF_HOME=/path/to/hf_cache
```

Next, let's create a working folder for this tutorial:

```bash
mkdir ./prep-stack-tutorial
```

Let's create a folder called `hf_dataset` and download the-stack dataset from huggingface here:

```bash
mkdir ./prep-stack-tutorial/hf_dataset
while ! huggingface-cli download bigcode/the-stack --revision v1.2 --repo-type dataset --max_workers 64 --local-dir ./prep-stack-tutorial/hf_dataset; 
do sleep 1; done
```

!!! warning "Choice of num_workers"

    Setting a large num_workers sometimes leads to connection errors.

## ⚙️ Step 2: Prepare the tokenizer and configs for conversion of data to Fast-LLM's memory-mapped indexed dataset format

In this step, we download the tokenizer and create configs required to run the data preparation scripts.

We'll use [Mistral-Nemo-Base-2407](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407)'s tokenizer for this tutorial.
Let's create a folder first:

```bash
mkdir -p ./prep-stack-tutorial/checkpoints/Mistral-Nemo-Base-2407
```

And then download the tokenizer with this Python script:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mistral-Nemo-Base-2407"
tokenizer = AutoTokenizer.from_pretrained(model_id) 
tokenizer.save_pretrained("./prep-stack-tutorial/checkpoints/Mistral-Nemo-Base-2407")
```

Let's create a folder to store the converted dataset:

```bash
mkdir -p ./prep-stack-tutorial/tokenized/Mistral-Nemo-Base-2407
```

Create a config like this -

```yaml
output_path: ./prep-stack-tutorial/tokenized/Mistral-Nemo-Base-2407

loading_workers: 32
tokenize_workers: 32
saving_workers: 32

dataset:
  path: ./prep-stack-tutorial/hf_dataset
  split: "train"
  trust_remote_code: true

tokenizer:
  path: ./prep-stack-tutorial/checkpoints/Mistral-Nemo-Base-2407/tokenizer.json
```

Save it as `./prep-stack-tutorial/the-stack-prepare.yaml`

## 🚀 Step 3: Launch data preparation job

Fast-LLM's prepare command processes the dataset by tokenizing and saving it in Fast-LLM's memory-mapped indexed dataset format.

=== "Prebuilt Docker"

    ```bash
    docker run -it --rm ghcr.io/servicenow/fast-llm:latest \
        -v ./prep-stack-tutorial:/app/prep-stack-tutorial \
        fast-llm prepare gpt_memmap --config /app/prep-stack-tutorial/the-stack-prepare.yaml
    ```

=== "Custom Installation"

    Please follow the instructions in the [Quick-Start guide](quick-start/#step-1-initial-setup-custom-installation) to set up Fast-LLM in your environment.

    Then, run the following command:

    ```bash
    fast-llm prepare gpt_memmap --config ./prep-stack-tutorial/the-stack-prepare.yaml
    ```

=== "Slurm"

    ```bash
    sbatch <<EOF
    #!/bin/bash
    # SBATCH --job-name=fast-llm-stack-prepare
    # SBATCH --nodes=4
    # SBATCH --ntasks-per-node=1
    # SBATCH --exclusive
    # SBATCH --output=/app/prep-stack-tutorial/prepare-output.log
    # SBATCH --error=/app/prep-stack-tutorial/prepare-error.log

    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    MASTER_PORT=8001

    export PYTHONHASHSEED=0

    srun \
        --container-image="ghcr.io/servicenow/fast-llm:latest" \
        --container-mounts="$(pwd)/prep-stack-tutorial:/app/prep-stack-tutorial" \
        --container-env="PYTHONHASHSEED" \
        --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
        bash -c "
            torchrun --rdzv_backend=static \
                     --rdzv_id=0 \
                     --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \
                     --node_rank=\\$SLURM_NODEID \
                     --nproc_per_node=\\$SLURM_NTASKS_PER_NODE \
                     --nnodes=\\$SLURM_NNODES:\\$SLURM_NNODES \
                     --max_restarts=0 \
                     --rdzv_conf=timeout=3600 \
                     --no_python \
                     fast-llm prepare gpt_memmap \
                     --config /app/prep-stack-tutorial/the-stack-prepare.yaml"
    EOF
    ```

    You can follow the job's progress by running `squeue -u $USER` and checking the logs in `prep-stack-tutorial/prepare-output.log` and `prep-stack-tutorial/prepare-error.log`, respectively.

=== "Kubeflow"

    First, you need a shared PVC to store the dataset.
    If you haven't already, create a shared PVC to store the dataset:

    ```yaml
    kubectl apply -f - <<EOF
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
    name: pvc-prep-stack-tutorial
    spec:
    accessModes:
      - ReadWriteMany
    resources:
      requests:
        storage: 100Gi
    EOF
    ```

    Next, create a pod to copy the files to the shared PVC:

    ```yaml
    kubectl apply -f - <<EOF
    apiVersion: v1
    kind: Pod
    metadata:
    name: pod-prep-stack-tutorial
    spec:
    containers:
      - name: busybox
        image: busybox
        command: ["sleep", "3600"]
        volumeMounts:
          - mountPath: /app
            name: prep-stack-tutorial
    volumes:
      - name: prep-stack-tutorial
        persistentVolumeClaim:
          claimName: pvc-prep-stack-tutorial
    EOF
    ```

    Now, copy the files to the shared PVC:

    ```bash
    kubectl cp ./prep-stack-tutorial pod-prep-stack-tutorial:/app
    ```

    You can shut down the pod after copying the files now:

    ```bash
    kubectl delete pod pod-prep-stack-tutorial
    ```

    Then, run data preparation with the following command:

    ```yaml
    kubectl apply -f - <<EOF
    apiVersion: "kubeflow.org/v1"
    kind: "PyTorchJob"
    metadata:
      name: "fast-llm-stack-prepare"
    spec:
      nprocPerNode: "1"
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
                               --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \
                               --node_rank=\${RANK} \
                               --nproc_per_node=\${PET_NPROC_PER_NODE} \
                               --nnodes=\${PET_NNODES}:\${PET_NNODES} \
                               --max_restarts=0 \
                               --rdzv_conf=timeout=3600 \
                               --no_python \
                               fast-llm prepare gpt_memmap \
                               --config prep-stack-tutorial/the-stack-prepare.yaml
                  env:
                    - name: PYTHONHASHSEED
                      value: "0"
                  securityContext:
                    capabilities:
                      add:
                        - IPC_LOCK
                  volumeMounts:
                    - mountPath: /app/prep-stack-tutorial
                      name: prep-stack-tutorial
                    - mountPath: /dev/shm
                      name: dshm
              volumes:
                - name: prep-stack-tutorial
                  persistentVolumeClaim:
                    claimName: pvc-prep-stack-tutorial
                - name: dshm
                  emptyDir:
                    medium: Memory
                    sizeLimit: "1024Gi"
        Worker:
          replicas: 3
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
                               --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \
                               --node_rank=\${RANK} \
                               --nproc_per_node=\${PET_NPROC_PER_NODE} \
                               --nnodes=\${PET_NNODES}:\${PET_NNODES} \
                               --max_restarts=0 \
                               --rdzv_conf=timeout=3600 \
                               --no_python \
                               fast-llm prepare gpt_memmap \
                               --config prep-stack-tutorial/the-stack-prepare.yaml
                  env:
                    - name: PYTHONHASHSEED
                      value: "0"
                  securityContext:
                    capabilities:
                      add:
                        - IPC_LOCK
                  volumeMounts:
                    - mountPath: /app/prep-stack-tutorial
                      name: prep-stack-tutorial
                    - mountPath: /dev/shm
                      name: dshm
              volumes:
                - name: prep-stack-tutorial
                  persistentVolumeClaim:
                    claimName: pvc-prep-stack-tutorial
                - name: dshm
                  emptyDir:
                    medium: Memory
                    sizeLimit: "1024Gi"
    EOF
    ```

    You can follow the job's progress by running `kubectl get pods` and checking the logs with `kubectl logs fast-llm-stack-prepare-master-0`.

That is all! Once the jobs complete, you'll see the data in Fast-LLM's memory-mapped indexed dataset format in `./prep-stack-tutorial/tokenized/Mistral-Nemo-Base-2407` which can be used with Fast-LLM to set off a training run.
