---
title: Fast-LLM
hide:
  - navigation
  - toc
  - feedback
---

Welcome to **Fast-LLM**, the cutting-edge open-source library built for training large language models (LLMs) with exceptional speed, scalability, and customization. Developed by ServiceNow Research's Foundation Models Lab, Fast-LLM is engineered to meet the rigorous demands of professional AI teams, research institutions, and enterprises pushing the limits of generative AI. Whether you're training models for groundbreaking research or high-stakes production, Fast-LLM empowers you to achieve unparalleled results.

## Why Fast-LLM?

Fast-LLM is purpose-built for serious AI practitioners who need more than off-the-shelf solutions. It is designed to handle the most demanding language model training tasks, offering a robust, flexible, and high-performance alternative to commercial frameworks like Megatron-LM or NeMo.

### Key Features

- **🚀 Speed Like No Other:** Achieve record-breaking training throughput with Fast-LLM. For instance, train Mistral-7B at nearly **9,800 tokens/s/GPU** on a 4-node cluster with 32 H100 GPUs. Our optimized kernels, advanced parallelism, and memory-efficient techniques drastically reduce training time and cost.
- **📡 Unmatched Scalability:** Fast-LLM scales seamlessly from a single GPU to large compute clusters, supporting 3D parallelism (data, tensor, and pipeline), sequence length parallelism, and ZeRO-1, ZeRO-2, and ZeRO-3 techniques for maximum memory efficiency. Scale to the size you need without sacrificing performance.
- **🎛️ Total Flexibility:** Fast-LLM is compatible with all major language model architectures, including GPT, Llama, Mistral, StarCoder, and Mixtral. Its modular design enables extensive customization of model architectures, optimizers, data loaders, and training loops, giving you full control over your training workflows.
- **📦 Seamless Integration:** Fast-LLM integrates smoothly with popular libraries such as [Hugging Face Transformers](https://huggingface.co/transformers), making it easy to leverage existing models and datasets while benefiting from our optimizations.
- **🛠️ Professional-Grade Tools:** Fast-LLM supports mixed precision training, large batch training, and gradient accumulation, all while maintaining reproducibility through deterministic behavior. Our pre-built Docker images, YAML-based configurations, and command-line interface make setup straightforward, so you can focus on what matters most—innovating with AI.

## The Fast-LLM Advantage

Designed for professionals who demand speed, scale, and customization, Fast-LLM is not just another library, it's a platform for powering the next generation of AI breakthroughs. Here's what sets Fast-LLM apart:

- **Purpose-Built for Large-Scale AI:** Unlike generic frameworks, Fast-LLM is optimized specifically for training large language models, with features tuned for massive compute clusters and high-throughput workflows.
- **Openness Without Compromise:** Our commitment to open-source ensures that you can customize and extend Fast-LLM to suit your specific needs, without the limitations of proprietary software.
- **Community-Driven Development:** While our focus is on professionals and enterprise users, we believe in open innovation. Fast-LLM's development is transparent, and we actively welcome contributions that help make our platform even more powerful.

## Project Scope and Objectives

Fast-LLM is designed to be the go-to solution for those training the most sophisticated language models. Our objectives include:

- **Accelerating Training Workflows:** By leveraging optimized kernel efficiency, advanced parallelism, and custom memory management techniques, we aim to deliver the fastest LLM training experience available.
- **Supporting a Broad Range of Architectures:** Fast-LLM offers built-in support for GPT, Llama, StarCoder, Mistral, Mixtral, and more, with an architecture-agnostic approach that allows users to easily adapt the framework to emerging models.
- **Enabling Seamless Integration and Deployment:** From training to deployment, Fast-LLM integrates effortlessly with existing ML pipelines, including Hugging Face Transformers and Kubernetes-based clusters.
- **Advancing LLM Research and Production-Readiness:** With support for mixed precision training, ZeRO optimizations, and reproducibility features, Fast-LLM is equipped for both cutting-edge research and mission-critical production environments.

## Collaboration and Contribution

As we continue to expand Fast-LLM, we're looking for contributions from the community to help shape its future. We welcome:

- **Testing and Bug Fixes:** Help us identify issues and improve stability.
- **Feature Development:** Contribute new capabilities, such as custom kernels or support for alternative hardware like AMD and Intel.
- **Documentation and Tutorials:** Make Fast-LLM more accessible by improving our [documentation](https://servicenow.github.io/Fast-LLM) and writing practical guides.

Fast-LLM is more than just software—it's a community. Get involved by exploring our [contribution guidelines](https://github.com/ServiceNow/Fast-LLM/CONTRIBUTING.md) and engaging with us on [GitHub Discussions](). 

## Getting Started

Ready to dive in? Check out our [quickstart guide](quickstart.md) for an overview of how to set up and run Fast-LLM on different platforms, including Slurm and Kubernetes. Explore the [examples](examples/) section for pre-configured setups to help you get started quickly with your own training experiments.

For any questions or issues, don't hesitate to open an [issue](https://github.com/ServiceNow/Fast-LLM/issues) or reach out to the community. We're here to help you accelerate your LLM training to full speed.
