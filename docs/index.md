---
title: Fast-LLM
hide:
  - navigation
  - toc
  - feedback
---

Welcome to **Fast-LLM**, the cutting-edge open-source library built for training large language models (LLMs) with exceptional speed, scalability, and customization. Developed by [ServiceNow Research](https://www.servicenow.com/research/)'s Foundation Models Lab, Fast-LLM is engineered to meet the rigorous demands of professional AI teams, research institutions, and enterprises pushing the limits of generative AI. Whether you're training models for groundbreaking research or high-stakes production, Fast-LLM empowers you to achieve unparalleled results.

## Why Fast-LLM?

Fast-LLM is designed for professionals who demand speed, scalability, and customization in training large language models. It goes beyond off-the-shelf solutions to meet the rigorous requirements of large-scale AI projects, offering a robust, flexible, and high-performance alternative to frameworks like NVIDIA NeMo Megatron. With Fast-LLM, you can train your most sophisticated models while optimizing for both performance and cost.

### The Fast-LLM Advantage

Fast-LLM isn't just another library, it's a platform for powering the next generation of AI breakthroughs. Here's what sets it apart:

- **🚀 Purpose-Built for Large-Scale AI:** Optimized specifically for training large language models at scale, Fast-LLM comes with features fine-tuned for massive compute clusters and high-throughput workflows. It supports advanced parallelism techniques, ZeRO optimizations, and high-throughput kernels, making it ideal for handling the most demanding training tasks.
- **💰 Cost Efficiency That Sets Fast-LLM Apart:** Fast-LLM's optimizations translate directly into significant cost savings:
  - **Lower Training Costs:** Fast-LLM achieves higher throughput per GPU, reducing the number of hours needed to complete training tasks. For example, training a Mistral-7B model can be up to xx% cheaper compared to other frameworks due to faster processing (insert exact point of reference here).
  - **More Tokens for Your Budget:** Train on significantly more data within the same budget, up to xx% more tokens per dollar—leading to better-trained models and higher-quality results (insert exact point of reference here).
  [Learn more about Fast-LLM's cost efficiency and see detailed comparisons](cost-efficiency.md).
- **🔓 Openness Without Compromise:** Our commitment to open-source ensures that you can customize and extend Fast-LLM to suit your specific needs without the limitations of proprietary software. Fast-LLM gives you full control over your training workflows, from experimentation to production.
- **🌍 Community-Driven Development:** While our focus is on professionals and enterprise users, we believe in open innovation. Fast-LLM's development is transparent, and we actively welcome contributions that make our platform even more powerful and versatile.

### Built for the Most Demanding Training Tasks

Fast-LLM is engineered to handle complex AI projects with ease, offering a scalable solution that supports various model architectures, including Llama, Mistral, StarCoder, and Mixtral. Whether you're training on a single GPU or a multi-node cluster, Fast-LLM adapts to your setup and scales effortlessly to meet your requirements.

### Key Features

Fast-LLM offers all the features you need to accelerate your LLM training to full speed:

- **🚀 Speed Like No Other:** Achieve record-breaking training throughput with Fast-LLM. For instance, train Mistral-7B at nearly **9,800 tokens/s/GPU** on a 4-node cluster with 32 H100 GPUs. Our optimized kernels, advanced parallelism, and memory-efficient techniques drastically reduce training time and cost.
- **📡 Unmatched Scalability:** Fast-LLM scales seamlessly from a single GPU to large compute clusters, supporting 3D parallelism (data, tensor, and pipeline), sequence length parallelism, and ZeRO-1, ZeRO-2, and ZeRO-3 techniques for maximum memory efficiency. Scale to the size you need without sacrificing performance.
- **🎛️ Total Flexibility:** Fast-LLM is compatible with all major language model architectures, including GPT, Llama, Mistral, StarCoder, and Mixtral. Its modular design enables extensive customization of model architectures, optimizers, data loaders, and training loops, giving you full control over your training workflows.
- **📦 Seamless Integration:** Fast-LLM integrates smoothly with popular libraries such as [Hugging Face Transformers](https://huggingface.co/transformers), making it easy to leverage existing models and datasets while benefiting from our optimizations.
- **🛠️ Professional-Grade Tools:** Fast-LLM supports mixed precision training, large batch training, and gradient accumulation, all while maintaining reproducibility through deterministic behavior. Our pre-built Docker images, YAML-based configurations, and command-line interface make setup straightforward, so you can focus on what matters most: innovating with AI.

## Project Scope and Objectives

Fast-LLM is designed to be the go-to solution for those training the most sophisticated language models. Our objectives include:

- **Accelerating Training Workflows:** By leveraging optimized kernel efficiency, advanced parallelism, and custom memory management techniques, we aim to deliver the fastest LLM training experience available.
- **Supporting a Broad Range of Architectures:** Fast-LLM offers built-in support for GPT, Llama, StarCoder, Mistral, Mixtral, and more, with an architecture-agnostic approach that allows users to easily adapt the framework to emerging models.
- **Enabling Seamless Integration and Deployment:** From training to deployment, Fast-LLM integrates effortlessly with existing ML pipelines, including [Hugging Face Transformers](https://huggingface.co/transformers) and [Kubernetes](https://kubernetes.io)-based clusters.
- **Advancing LLM Research and Production-Readiness:** With support for mixed precision training, Zero Redundancy Optimizer (ZeRO) techniques, and reproducibility features, Fast-LLM is equipped for both cutting-edge research and mission-critical production workloads.

## Collaboration and Contribution

As we continue to expand Fast-LLM, we're looking for contributions from the community to help shape its future. We welcome:

- **Testing and Bug Fixes:** Help us identify issues and improve stability.
- **Feature Development:** Contribute new capabilities, such as custom kernels or support for alternative hardware like AMD and Intel.
- **Documentation and Tutorials:** Make Fast-LLM more accessible by improving our [documentation](https://servicenow.github.io/Fast-LLM) and writing practical guides.

Fast-LLM is more than just software, it's a community. Get involved by exploring our [contribution guidelines](https://github.com/ServiceNow/Fast-LLM/CONTRIBUTING.md) and engaging with us on [GitHub Discussions](https://github.com/ServiceNow/Fast-LLM/discussions).

## Getting Started

Ready to dive in? Check out our [quickstart guide](quickstart.md) for an overview of how to set up and run Fast-LLM on different platforms, including [Slurm](https://slurm.schedmd.com) and [Kubernetes](https://kubernetes.io). Explore the [examples](https://github.com/ServiceNow/Fast-LLM/tree/main/examples) for pre-configured setups to help you get started quickly with your own training experiments.

For any questions or issues, don't hesitate to open an [issue](https://github.com/ServiceNow/Fast-LLM/issues) or reach out to the community. We're here to help you accelerate your LLM training to full speed.
