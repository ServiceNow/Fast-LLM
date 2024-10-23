<div align="center" style="margin-bottom: 1em;">

<img width=50% src="docs/assets/images/logo.png" alt="Fast-LLM Logo"></img>

[![Docker][ci-badge]][ci-workflow]
[![Documentation][docs-badge]][docs-workflow]
[![License][license-badge]][license]

*Accelerating your LLM training to full speed*

Made with ❤️ by [ServiceNow Research][servicenow-research]

</div>

## Overview

Fast-LLM is a cutting-edge open-source library for training large language models with exceptional speed, scalability, and flexibility. Built on [PyTorch][pytorch] and [Triton][triton], Fast-LLM empowers AI teams to push the limits of generative AI, from research to production.

Optimized for training models of all sizes—from small 1B-parameter models to massive clusters with 70B+ parameters—Fast-LLM delivers faster training, lower costs, and seamless scalability. Its fine-tuned kernels, advanced parallelism techniques, and efficient memory management make it the go-to choice for diverse training needs.

As a truly open-source project, Fast-LLM allows full customization and extension without proprietary restrictions. Developed transparently by a community of professionals on GitHub, the library benefits from collaborative innovation, with every change discussed and reviewed in the open to ensure trust and quality. Fast-LLM combines professional-grade tools with unified support for GPT-like architectures, offering the cost efficiency and flexibility that serious AI practitioners demand.

> [!NOTE]
> Fast-LLM is not affiliated with Fast.AI, FastHTML, FastAPI, FastText, or other similarly named projects. Our library's name refers to its speed and efficiency in language model training.

## Why Fast-LLM?

1. 🚀 **Fast-LLM is Blazingly Fast**:
    - ⚡️ Optimized kernel efficiency and reduced overheads.
    - 🔋 Optimized memory usage for best performance.
    - ⏳ Minimizes training time and cost.

2. 📈 **Fast-LLM is Highly Scalable**:
    - 📡 Distributed training across multiple GPUs and nodes using 3D parallelism (Data, Tensor, and Pipeline).
    - 🔗 Supports sequence length parallelism to handle longer sequences effectively.
    - 🧠 ZeRO-1, ZeRO-2, and ZeRO-3 implementations for improved memory efficiency.
    - 🎛️ Mixed precision training support for better performance.
    - 🏋️‍♂️ Large batch training and gradient accumulation support.
    - 🔄 Reproducible training with deterministic behavior.

3. 🎨 **Fast-LLM is Incredibly Flexible**:
    - 🤖 Compatible with all common language model architectures in a unified class.
    - ⚡ Efficient dropless Mixture-of-Experts (MoE) implementation with SoTA performance.
    - 🧩 Customizable language model architectures, data loaders, loss functions, and optimizers (in progress).
    - 🤗 Seamless integration with [Hugging Face Transformers][transformers].

4. 🎯 **Fast-LLM is Super Easy to Use**:
    - 📦 [Pre-built Docker images](https://github.com/ServiceNow/Fast-LLM/pkgs/container/fast-llm) for quick deployment.
    - 📝 Simple YAML configuration for hassle-free setup.
    - 💻 Command-line interface for easy launches.
    - 📊 Detailed logging and real-time monitoring features.
    - 📚 Extensive [documentation][docs] and practical tutorials (in progress).

5. 🌐 **Fast-LLM is Truly Open Source**:
    - ⚖️ Licensed under [Apache 2.0][license] for maximum freedom to use Fast-LLM at work, in your projects, or for research.
    - 💻 Transparently developed on GitHub with public [roadmap][roadmap] and [issue tracking][issues].
    - 🤝 Contributions and collaboration are always welcome!

## Usage

We'll walk you through how to use Fast-LLM to train a large language model on a cluster with multiple nodes and GPUs. We'll show an example setup using a Slurm cluster and a Kubernetes cluster.

For this demo, we will train a Mistral-7B model from scratch for 100 steps on random data. The config file `examples/mistral-4-node-benchmark.yaml` is pre-configured for a multi-node setup with 4 DGX nodes, each with 8 A100-80GB or H100-80GB GPUs.

> [!NOTE]
> Fast-LLM scales from a single GPU to large clusters. You can start small and expand based on your resources.

Expect to see a significant speedup in training time compared to other libraries! For training Mistral-7B, Fast-LLM is expected to achieve a throughput of **9,800 tokens/s/H100** (batch size 32, sequence length 8k) on a 4-node cluster with 32 H100s.

### Running Fast-LLM on a Slurm Cluster

#### Prerequisites

- A [Slurm](https://slurm.schedmd.com/) cluster with at least 4 DGX nodes with 8 A100-80GB or H100-80GB GPUs each.
- CUDA 12.1 or higher.
- Dependencies: [PyTorch][pytorch], [Triton][triton], and [Apex](https://github.com/NVIDIA/apex) installed on all nodes.

#### Steps

1. Deploy the [nvcr.io/nvidia/pytorch:24.07-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) Docker image to all nodes (recommended), because it contains all the necessary dependencies.
2. Install Fast-LLM on all nodes:

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

3. Use the example Slurm job script [examples/fast-llm.sbat](examples/fast-llm.sbat) to submit the job to the cluster:

    ```bash
    sbatch examples/fast-llm.sbat
    ```

4. Monitor the job's progress:

    - Logs: Follow `job_output.log` and `job_error.log` in your working directory for logs.
    - Status: Use `squeue -u $USER` to see the job status.

Now, you can sit back and relax while Fast-LLM trains your model at full speed! ☕

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

## Next Steps

📖 **Want to learn more?** Check out our [documentation][docs] for more information on how to use Fast-LLM.

🔨 **We welcome contributions to Fast-LLM!** Have a look at our [contribution guidelines](CONTRIBUTING.md).

🐞 **Something doesn't work?** Open an [issue](https://github.com/ServiceNow/Fast-LLM/issues)!

## License

Fast-LLM is licensed by ServiceNow, Inc. under the Apache 2.0 License. See [LICENSE][license] for more information.

## Vulnerability Reporting

For security issues, email [disclosure@servicenow.com](mailto:disclosure@servicenow.com). See our [security policy](SECURITY.md).

[roadmap]: https://github.com/ServiceNow/Fast-LLM/milestones
[issues]: https://github.com/ServiceNow/Fast-LLM/issues
[ci-badge]: https://github.com/ServiceNow/Fast-LLM/actions/workflows/ci.yaml/badge.svg
[ci-workflow]: https://github.com/ServiceNow/Fast-LLM/actions/workflows/ci.yaml
[docs-badge]: https://github.com/ServiceNow/Fast-LLM/actions/workflows/docs.yaml/badge.svg
[docs-workflow]: https://github.com/ServiceNow/Fast-LLM/actions/workflows/docs.yaml
[docs]: https://servicenow.github.io/Fast-LLM
[license-badge]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
[license]: LICENSE
[servicenow-research]: https://www.servicenow.com/research/
[pytorch]: https://pytorch.org/
[triton]: https://triton-lang.org
[transformers]: https://huggingface.co/transformers
