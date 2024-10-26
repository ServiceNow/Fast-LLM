---
title: "Kubernetes"
---

- **Purpose:** These guides cover specific environments and configurations for deploying Fast-LLM in different setups.
- **Content Organization:**
  - **in-action/slurm**: Provide detailed instructions on deploying Fast-LLM on a Slurm cluster, covering multi-node setups, configuring Slurm scripts, and managing jobs.
  - **in-action/kubernetes**: Guide for deploying Fast-LLM using Kubernetes, including creating the appropriate workloads (e.g., Job, Pod, StatefulSet), handling private Docker images, and configuring multi-node training.
  - **File Single Node Guide Here Too?** If you include a "single-node" guide in this section as well, make it more advanced, focusing on optimizing performance, using different configurations, or tuning settings for different GPU models.
- **Why It Makes Sense:** Organizing by deployment environment ensures users can quickly find the relevant guide based on their setup. Including both multi-node cluster guides and single-node advanced setups allows users to scale their knowledge.

---

We'll walk you through how to use Fast-LLM to train a large language model on a cluster with multiple nodes and GPUs. We'll show an example setup using a Slurm cluster and a Kubernetes cluster.

For this demo, we will train a Mistral-7B model from scratch for 100 steps on random data. The config file `examples/mistral-4-node-benchmark.yaml` is pre-configured for a multi-node setup with 4 DGX nodes, each with 8 A100-80GB or H100-80GB GPUs.

> [!NOTE]
> Fast-LLM scales from a single GPU to large clusters. You can start small and expand based on your resources.

Expect to see a significant speedup in training time compared to other libraries! For training Mistral-7B, Fast-LLM is expected to achieve a throughput of **9,800 tokens/s/H100** (batch size 32, sequence length 8k) on a 4-node cluster with 32 H100s.


### Running Fast-LLM on a Kubernetes Cluster

#### Prerequisites

- A [Kubernetes](https://kubernetes.io/) cluster with at least 4 DGX nodes with 8 A100-80GB or H100-80GB GPUs each.
- [KubeFlow](https://www.kubeflow.org/) installed.
- Locked memory limit set to unlimited at the host level on all nodes. Ask your cluster admin to do this if needed.

#### Steps

1. Create a Kubernetes [PersistentVolumeClaim](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) (PVC) named `fast-llm-home` that will be mounted to `/home/fast-llm` in the container using [examples/fast-llm-pvc.yaml](examples/fast-llm-pvc.yaml):

    ```bash
    kubectl apply -f examples/fast-llm-pvc.yaml
    ```

2. Create a [PyTorchJob](https://www.kubeflow.org/docs/components/training/user-guides/pytorch/) resource using the example configuration file [examples/fast-llm.pytorchjob.yaml](examples/fast-llm.pytorchjob.yaml):

    ```bash
    kubectl apply -f examples/fast-llm.pytorchjob.yaml
    ```

3. Monitor the job status:

    - Use `kubectl get pytorchjobs` to see the job status.
    - Use `kubectl logs -f fast-llm-master-0 -c pytorch` to follow the logs.

That's it! You're now up and running with Fast-LLM on Kubernetes. 🚀
