---
title: "Fast-LLM: Train Large Language Models Faster Than Ever Before"
hide:
  - navigation
  - toc
  - feedback
---

Welcome to **Fast-LLM**, the cutting-edge open-source library built for training large language models (LLMs) with **unmatched speed**, scalability, and cost-efficiency**. Developed by [ServiceNow Research](https://www.servicenow.com/research/)'s Foundation Models Lab, Fast-LLM is engineered to meet the rigorous demands of professional AI teams, research institutions, and enterprises pushing the limits of generative AI. **Achieve groundbreaking research and high-stakes production goals faster with Fast-LLM.**

[Get started with Fast-LLM](quickstart.md) and experience the next generation of LLM training. [See Fast-LLM in action](in-action.md) and discover how it can transform your training workflows.

## Why Fast-LLM?

Fast-LLM is designed for professionals who demand exceptional performance in large-scale language model training. It goes beyond off-the-shelf solutions to deliver a **robust, flexible, and high-performance open-source alternative** to commercial frameworks like NVIDIA NeMo Megatron. Whether you're optimizing for speed, cost, or scalability, Fast-LLM helps you get the most out of your training resources.

### The Fast-LLM Advantage

Fast-LLM isn't just another library, **it's a platform for powering the next generation of AI breakthroughs**. Here’s what sets it apart:

- **🚀 Purpose-Built for Large-Scale AI:** Optimized specifically for training large language models at scale, Fast-LLM features advanced parallelism techniques, ZeRO optimizations, and high-throughput kernels, making it ideal for handling demanding training tasks across small and massive compute clusters.

- **💰 Cost Efficiency That Sets Fast-LLM Apart:**

  - **Lower Training Costs:** With higher throughput per GPU, Fast-LLM reduces the training time required. For instance, training a Mistral-7B model can be up to **xx% cheaper** compared to other frameworks due to faster processing and memory efficiency.
  - **More Tokens for Your Budget:** Train up to xx% more tokens for the same budget, leading to better-trained models without breaking your financial constraints.

  [Learn more about Fast-LLM's cost efficiency and see detailed comparisons](cost-efficiency.md).

- **🔓 Openness Without Compromise:** Fast-LLM's commitment to open-source ensures full customization and extension capabilities, allowing users to tailor the framework to specific needs without the limitations of proprietary software.

- **🌍 Community-Driven Development:** Built by professionals for professionals, Fast-LLM's development is transparent, with an open invitation to the community to contribute. [**Join the Fast-LLM community**](community/join-us) to help shape the future of large-scale AI training.

### Key Features

Fast-LLM offers all the capabilities you need to accelerate your LLM training and **push the boundaries of what's possible**:

- **🚀 Speed Like No Other:** Achieve record-breaking training throughput with Fast-LLM. For instance, train Mistral-7B at **9,800 tokens/s/GPU** on a 4-node cluster with 32 H100 GPUs (batch size 32, sequence length 8k). Our optimized kernels, advanced parallelism, and memory-efficient techniques drastically reduce training time and cost.

- **📡 Unmatched Scalability:** Seamlessly scale from a single GPU to large compute clusters. Fast-LLM supports 3D parallelism (data, tensor, and pipeline), sequence length parallelism, and ZeRO-1,2,3 techniques for maximum memory efficiency. Scale to the size you need without sacrificing performance.

- **🎛️ Total Flexibility:** Compatible with all major language model architectures, including but not limited to Llama, Mistral, StarCoder, and Mixtral. Fast-LLM's modular design gives you full control over your training workflows.

- **📦 Seamless Integration:** Integrate smoothly with popular libraries such as [Hugging Face Transformers](https://huggingface.co/transformers). Benefit from Fast-LLM's optimizations without disrupting your existing pipelines.

- **🛠️ Professional-Grade Tools:** Enjoy mixed precision training, large batch training, and gradient accumulation. Fast-LLM ensures reproducibility through deterministic behavior and provides pre-built Docker images, YAML configurations, and a simple, intuitive command-line interface.

[Download Fast-LLM](https://github.com/ServiceNow/Fast-LLM/releases) and start training your large language models at full speed. [Join the Fast-LLM community](community/join-us) and collaborate with like-minded professionals to advance AI research and development.

## Use Cases and Success Stories

Fast-LLM powers the world's most advanced AI projects:

- **NLP Research and Development:** Train state-of-the-art language models for natural language understanding, summarization, and conversational AI.
- **Enterprise AI Solutions:** Accelerate time-to-market for AI products by reducing training costs and enabling faster iteration.
- **Academic Collaborations:** Drive AI innovation with high-performance training capabilities that support cutting-edge research in machine learning.

See how Fast-LLM has helped early adopters achieve up to xx% faster results. [Explore use cases and success stories](success-stories).

## Project Scope and Objectives

Fast-LLM is designed to be the **go-to solution** for those training the most sophisticated language models. Our objectives include:

- **Accelerating Training Workflows:** Deliver the fastest LLM training experience with optimized kernel efficiency, parallelism, and memory management.
- **Supporting a Broad Range of Architectures:** Offer built-in support for all major language model architectures, with an architecture-agnostic approach that allows users to easily adapt the framework to emerging models.
- **Enabling Seamless Integration and Deployment:** Integrate effortlessly into existing ML pipelines, including [Hugging Face Transformers](https://huggingface.co/transformers) and [Kubernetes](https://kubernetes.io)-based clusters.
- **Advancing LLM Research and Production-Readiness:** Be suitable for both cutting-edge research and mission-critical production workloads.

## Collaboration and Contribution

As we continue to expand Fast-LLM, we're looking for contributions from the community to help shape its future. We welcome:

- **Testing and Bug Fixes:** Help us identify issues and improve stability.
- **Feature Development:** Contribute new capabilities, such as custom kernels or support for alternative hardware like AMD and Intel.
- **Documentation and Tutorials:** Make Fast-LLM more accessible by improving our [documentation](https://servicenow.github.io/Fast-LLM) and writing practical guides.

Fast-LLM is more than just software, it's a community. Get involved by exploring our [contribution guidelines](https://github.com/ServiceNow/Fast-LLM/CONTRIBUTING.md) and engaging with us on [GitHub Discussions](https://github.com/ServiceNow/Fast-LLM/discussions).

## Getting Started

Ready to dive in? Check out our [quickstart guide](quickstart.md) for an overview of how to set up and run Fast-LLM on different platforms, including [Slurm](https://slurm.schedmd.com) and [Kubernetes](https://kubernetes.io). Explore the [examples](https://github.com/ServiceNow/Fast-LLM/tree/main/examples) for pre-configured setups to help you get started quickly with your own training experiments.

For any questions or issues, open an [issue](https://github.com/ServiceNow/Fast-LLM/issues) or join the [community discussion](https://github.com/ServiceNow/Fast-LLM/discussions).
