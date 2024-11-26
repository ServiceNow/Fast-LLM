---
title: "StarCoder2"
---

2023 was a transformative year for ServiceNow Research's Foundation Model Lab. Partnering with [BigCode](https://www.bigcode-project.org), we set out to build **StarCoder2** [@lozhkov2024starcoder], an open-source language model designed specifically for coding tasks. This iteration of StarCoder [@li2023starcoder] has been built to handle a wide range of programming languages with performance on par with some larger models.

Our goal was ambitious: to train the [3-billion-parameter StarCoder2 model](https://huggingface.co/bigcode/starcoder2-3b) on over **3 trillion tokens** from **The Stack V2**—a rich, diverse dataset compiled by BigCode from the Software Heritage archive. This data provided StarCoder2 with the breadth of real-world code examples and programming paradigms it needed to tackle complex coding tasks with high accuracy and deep contextual understanding.

To bring StarCoder2 to life, we ran Fast-LLM on **NVIDIA's DGX SuperCloud**, utilizing **DGX A100-80GB nodes**. Fast-LLM allowed us to maximize GPU throughput and streamline our entire training pipeline. The complexity of scaling StarCoder2's training across nodes became a seamless experience.

## How Fast-LLM Made StarCoder2 Possible

Fast-LLM was designed to maximize efficiency in large-scale language model training—especially for tasks like StarCoder2. Here's how Fast-LLM's capabilities helped us achieve our goals:

-   **Optimized Throughput and GPU Utilization**: Fast-LLM's data parallelism allowed each A100-80GB GPU to operate at its peak, sustaining **10,000 tokens per second** throughput. This boosted GPU utilization and brought down training time by **20%** compared to other frameworks. Fast-LLM made sure every GPU cycle was used efficiently, cutting down on idle time across the board.

-   **Support for Long Contexts**: With Fast-LLM's built-in Grouped Query Attention (GQA), StarCoder2-3B was able to leverage a **16,384 token context window**. This is essential for code comprehension, where context often spans hundreds of lines or more. GQA enabled the model to hold extensive context across sequences, which translates into better understanding of long code snippets, in-depth documentation, and detailed coding conversations.

-   **Fill-in-the-Middle (FIM) Training**: Fast-LLM supported FIM training objectives natively, allowing StarCoder2-3B to complete and understand code by predicting missing snippets in various contexts. This structure-focused training enhanced the model's performance, making it adept at understanding code structure, flow, and syntax.

## The Takeaway

StarCoder2-3B is the first large-scale, real-world demonstration of Fast-LLM's capabilities in specialized language model training. This project exemplifies how Fast-LLM not only powers large models but does so with adaptability and efficiency. It's not just about achieving results—it's about doing so in a way that's replicable and accessible to labs of all sizes.

With Fast-LLM, we've made a leap in efficiency and performance, setting the stage for future innovation in LLM training. This is just the beginning, and we're excited to see how Fast-LLM will continue to push the boundaries of language model development for coding and beyond.
