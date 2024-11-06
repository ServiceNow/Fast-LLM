---
title: "Slurm"
---

-   **Purpose:** These guides cover specific environments and configurations for deploying Fast-LLM in different setups.
-   **Content Organization:**
    -   **in-action/slurm**: Provide detailed instructions on deploying Fast-LLM on a Slurm cluster, covering multi-node setups, configuring Slurm scripts, and managing jobs.
    -   **in-action/kubernetes**: Guide for deploying Fast-LLM using Kubernetes, including creating the appropriate workloads (e.g., Job, Pod, StatefulSet), handling private Docker images, and configuring multi-node training.
    -   **File Single Node Guide Here Too?** If you include a "single-node" guide in this section as well, make it more advanced, focusing on optimizing performance, using different configurations, or tuning settings for different GPU models.
-   **Why It Makes Sense:** Organizing by deployment environment ensures users can quickly find the relevant guide based on their setup. Including both multi-node cluster guides and single-node advanced setups allows users to scale their knowledge.

---

We'll walk you through how to use Fast-LLM to train a large language model on a cluster with multiple nodes and GPUs. We'll show an example setup using a Slurm cluster and a Kubernetes cluster.

For this demo, we will train a Mistral-7B model from scratch for 100 steps on random data. The config file `examples/mistral-4-node-benchmark.yaml` is pre-configured for a multi-node setup with 4 DGX nodes, each with 8 A100-80GB or H100-80GB GPUs.

> [!NOTE]
> Fast-LLM scales from a single GPU to large clusters. You can start small and expand based on your resources.

Expect to see a significant speedup in training time compared to other libraries! For training Mistral-7B, Fast-LLM is expected to achieve a throughput of **9,800 tokens/s/H100** (batch size 32, sequence length 8k) on a 4-node cluster with 32 H100s.

### Running Fast-LLM on a Slurm Cluster without Docker

#### Prerequisites

-   A [Slurm](https://slurm.schedmd.com/) cluster with at least 4 DGX nodes with 8 A100-80GB or H100-80GB GPUs each.
-   CUDA 12.1 or higher.
-   Dependencies: [PyTorch][pytorch], [Triton][triton], and [Apex](https://github.com/NVIDIA/apex) installed on all nodes.

#### Steps

1.  Deploy the [nvcr.io/nvidia/pytorch:24.07-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) Docker image to all nodes (recommended), because it contains all the necessary dependencies.
2.  Install Fast-LLM on all nodes:

    ```bash
    sbatch <<EOF
    #!/bin/bash
    #SBATCH --nodes=$(scontrol show node | grep -c NodeName)
    #SBATCH --ntasks-per-node=1
    #SBATCH --ntasks=$(scontrol show node | grep -c NodeName)
    #SBATCH --exclusive

    srun bash -c 'pip install --no-cache-dir -e "git+https://github.com/ServiceNow/Fast-LLM.git#egg=llm[CORE,OPTIONAL,DEV]"'
    EOF
    ```

3.  Use the example Slurm job script [examples/fast-llm.sbat](examples/fast-llm.sbat) to submit the job to the cluster:

    ```bash
    sbatch examples/fast-llm.sbat
    ```

4.  Monitor the job's progress:

    -   Logs: Follow `job_output.log` and `job_error.log` in your working directory for logs.
    -   Status: Use `squeue -u $USER` to see the job status.

Now, you can sit back and relax while Fast-LLM trains your model at full speed! ☕

### Running Fast-LLM on a Slurm Cluster with Docker
