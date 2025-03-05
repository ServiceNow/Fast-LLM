---
title: "Fast-LLM: Train Large Language Models Faster Than Ever Before"
hide:
  - navigation
---

Introducing **Fast-LLM**, the cutting-edge open-source library built for training large language models (LLMs) with **unmatched speed, scalability, and cost-efficiency**. Developed by [ServiceNow Research](https://www.servicenow.com/research/)'s Foundation Models Lab, Fast-LLM is engineered to meet the rigorous demands of professional AI researchers, AI/ML engineers, academic and industrial research institutions, and enterprise product development teams pushing the limits of generative AI. **Achieve groundbreaking research and high-stakes production goals faster with Fast-LLM.**

[Start your journey with Fast-LLM](quick-start.md) and explore the future of LLM training. Dive into [real-world use cases](recipes/train.md) to see how Fast-LLM can elevate your training workflows.

## Why Fast-LLM?

Fast-LLM is designed for professionals who demand exceptional performance for efficient, large-scale language model training on GPUs, where maximizing FLOPS is key. Fast-LLM integrates effortlessly into existing ML pipelines and goes beyond off-the-shelf commercial frameworks to deliver a **robust, flexible, and high-performance open-source alternative**. Whether you're optimizing for speed, cost, or scalability, Fast-LLM helps you get the most out of your training infrastructure.

### The Fast-LLM Advantage

Fast-LLM isn't just another library, **it's a platform for powering the next generation of AI breakthroughs**. Here's what sets it apart:

-   **🚀 Purpose-Built for Small- and Large-Scale AI:** Optimized specifically for training language models of all sizes, Fast-LLM excels from **small models around 1B parameters to massive clusters running 70B+ parameter models**, with kernels that are fine-tuned for maximum throughput across this entire range. At 10B-parameter scale, Fast-LLM avoids costly 3D-parallelism through memory optimization techniques such as ZeRO and activation recomputation, whereas at 100B-parameter scale, Fast-LLM optimally supports 3D-parallelism; making Fast-LLM the go-to choice for diverse training needs.

-   **🧠 Unified Support for GPT-Like Architectures:** Fast-LLM streamlines the implementation of GPT-like models into a [single, unified module](https://github.com/ServiceNow/Fast-LLM/blob/main/fast_llm/models/gpt/model.py), significantly reducing redundancy and simplifying adaptation to custom architectures. This approach ensures consistency and flexibility while minimizing development overhead.

-   **💰 Cost Efficiency That Sets Fast-LLM Apart:**

    -   **Lower Training Costs:** With higher throughput per GPU, Fast-LLM reduces the training time required. Training models can be cheaper compared to other frameworks due to faster processing and better memory efficiency.

    -   **More Tokens for Your Budget:** Train on more tokens for the same budget, leading to better-trained models without breaking your financial constraints.

-   **🔓 Openness Without Compromise:** Fast-LLM's open-source approach ensures that you can **fully customize and extend the library** to fit your exact needs, without the restrictions of proprietary software. Developed transparently by a community of experts on GitHub, every change is **publicly discussed and vetted**, fostering **trust and collaboration** so you can innovate with confidence, knowing the entire development process and decision making is out in the open.

-   **🌍 Community-Driven Development:** Built by professionals for professionals, Fast-LLM's development is transparent, with an open invitation to the community to contribute. [**Join the Fast-LLM community**](join-us.md) to help shape the future of large-scale AI training.

### Key Features

Fast-LLM offers all the capabilities you need to accelerate your LLM training and **push the boundaries of what's possible**:

-   **🚀 Speed Like No Other:** Achieve record-breaking training throughput with Fast-LLM. For instance, train Mistral-7B at **10,350 tokens/s/GPU** on a 4-node cluster with 32 H100 GPUs (batch size 64, sequence length 8k). Our optimized kernels, advanced parallelism, and memory-efficient techniques drastically reduce training time and cost.

-   **📡 Unmatched Scalability:** Seamlessly scale from a single GPU to large compute clusters. Fast-LLM supports 3D parallelism (data, tensor, and pipeline), sequence length parallelism, and ZeRO-1,2,3 techniques for maximum memory efficiency. Scale to the size you need without sacrificing performance.

-   **🎛️ Total Flexibility:** Compatible with all major language model architectures, including but not limited to Llama, Mistral, StarCoder, and Mixtral. Fast-LLM's modular design gives you full control over your training workflows.

-   **📦 Seamless Integration:** Integrate smoothly with popular libraries such as [HuggingFace Transformers](https://huggingface.co/transformers). Benefit from Fast-LLM's optimizations without disrupting your existing pipelines.

-   **🛠️ Professional-Grade Tools:** Enjoy mixed precision training, large batch training, and gradient accumulation. Fast-LLM ensures reproducibility through deterministic behavior and provides [pre-built Docker images](https://github.com/ServiceNow/Fast-LLM/pkgs/container/fast-llm), YAML configurations, and a simple, intuitive command-line interface.

[Get Fast-LLM](https://github.com/ServiceNow/Fast-LLM/releases) and start training your large language models in record time. [Join the Fast-LLM community](join-us.md) and collaborate with like-minded professionals to advance the state-of-the-art in AI research and development.

## Use Cases and Success Stories

Fast-LLM powers the world's most advanced AI projects:

-   **NLP Research and Development:** Train state-of-the-art language models for natural language understanding, summarization, and conversational AI.
-   **Enterprise AI Solutions:** Accelerate time-to-market for AI products by reducing training costs and enabling faster iteration.
-   **Academic Collaborations:** Drive AI innovation with high-performance training capabilities that support cutting-edge research in machine learning.

See how Fast-LLM has helped early adopters achieve faster results. [Explore use cases and success stories](success-stories/starcoder-2.md).

## Project Scope and Objectives

Fast-LLM is designed to be the **go-to solution** for those training the most sophisticated language models. Our objectives include:

-   **Accelerating Training Workflows:** Deliver the fastest LLM training experience with optimized kernel efficiency, parallelism, and memory management.
-   **Supporting a Broad Range of Architectures:** Offer built-in support for all major language model architectures, with an architecture-agnostic approach that allows users to easily adapt the framework to emerging models.
-   **Enabling Seamless Integration and Deployment:** Integrate effortlessly into existing ML pipelines, including [HuggingFace Transformers](https://huggingface.co/transformers) and [Kubernetes](https://kubernetes.io)-based clusters.
-   **Advancing LLM Research and Production-Readiness:** Be suitable for both cutting-edge research and mission-critical production workloads.

## Collaboration and Contribution

As Fast-LLM evolves, we invite the community to contribute and help shape its future. We welcome:

-   **Testing and Bug Fixes:** Help us identify issues and improve stability.
-   **Feature Development:** Contribute new models, new training features, and new optimizations.
-   **Documentation and Tutorials:** Make Fast-LLM more accessible by improving our documentation and writing practical guides.

Fast-LLM is more than just software, it's a community. Get involved by exploring our [contribution guidelines](contributing/contributing.md) and engaging with us on [GitHub Discussions](https://github.com/ServiceNow/Fast-LLM/discussions).

## Getting Started

Ready to dive in? Check out our [quick-start guide](quick-start.md) for an overview of how to set up and run Fast-LLM on different platforms, including [Slurm](https://slurm.schedmd.com) and [Kubernetes](https://kubernetes.io). Explore the [examples](https://github.com/ServiceNow/Fast-LLM/tree/main/examples) for pre-configured setups to help you get started quickly with your own training experiments.

For any questions or issues, open an [issue](https://github.com/ServiceNow/Fast-LLM/issues) or join the [community discussion](https://github.com/ServiceNow/Fast-LLM/discussions).
