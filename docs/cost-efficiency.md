---
title: Cost Efficiency Comparison
---

Fast-LLM is built for speed and scalability to minimize training costs. Its advanced parallelism techniques, memory-efficient implementations, and kernel optimizations enable significant cost savings compared to other training frameworks. Below, we present a detailed comparison of training costs for different model configurations and cluster sizes, demonstrating how Fast-LLM delivers more value for your budget.

## Comparing Training Costs Across Frameworks

To showcase the cost-saving potential of Fast-LLM, we've compared the cost of training a language model across various frameworks for different scenarios. For these calculations, we assume a cost of **USD 2.50 per H100 GPU per hour**.

!!! note "Disclaimer"

    All comparisons were conducted with identical model configurations and training setups across frameworks to maintain fairness. We optimized training parameters within each framework to achieve the best possible performance. Detailed configuration files are available in the footnotes for reference. If you have questions about our methods, assumptions, or suggestions for enhancing performance on any framework, please contact us at [fast-llm-team@servicenow.com](mailto:fast-llm-team@servicenow.com).

### Scenario Comparison: Training Costs and Token Efficiency

The tables below provide a comparison of training costs for three different model setups, including costs for training on **1 trillion tokens** and the total tokens trained within a **$100,000 budget**.

#### 1B Model on 1 DGX Node (8 H100s)

| Framework                                  | Training Throughput (tokens/s/GPU) | Cost to Train 1T Tokens (USD)  | Tokens Trained for $100k (Billion)  |
|:-------------------------------------------|-----------------------------------:|-------------------------------:|------------------------------------:|
| **Fast-LLM**[^fast-llm-1b]                 | [PLACEHOLDER]                      | **[PLACEHOLDER]**              | **[PLACEHOLDER]**                   |
| NVIDIA Megatron[^megatron-1b]              | [PLACEHOLDER]                      | [PLACEHOLDER]                  | [PLACEHOLDER]                       |
| MosaicML Composer[^mosaic-1b]              | [PLACEHOLDER]                      | [PLACEHOLDER]                  | [PLACEHOLDER]                       |
| Hugging Face Transformers[^huggingface-1b] | [PLACEHOLDER]                      | [PLACEHOLDER]                  | [PLACEHOLDER]                       |
| Meta Lingua[^metaligua-1b]                 | [PLACEHOLDER]                      | [PLACEHOLDER]                  | [PLACEHOLDER]                       |

#### 8B Model on 4 DGX Nodes (32 H100s)

| Framework                                  | Training Throughput (tokens/s/GPU) | Cost to Train 1T Tokens (USD)  | Tokens Trained for $100k (Billion)  |
|:-------------------------------------------|-----------------------------------:|-------------------------------:|------------------------------------:|
| **Fast-LLM**[^fast-llm-8b]                 | [PLACEHOLDER]                      | **[PLACEHOLDER]**              | **[PLACEHOLDER]**                   |
| NVIDIA Megatron[^megatron-8b]              | [PLACEHOLDER]                      | [PLACEHOLDER]                  | [PLACEHOLDER]                       |
| MosaicML Composer[^mosaic-8b]              | [PLACEHOLDER]                      | [PLACEHOLDER]                  | [PLACEHOLDER]                       |
| Hugging Face Transformers[^huggingface-8b] | [PLACEHOLDER]                      | [PLACEHOLDER]                  | [PLACEHOLDER]                       |
| Meta Lingua[^metaligua-8b]                 | [PLACEHOLDER]                      | [PLACEHOLDER]                  | [PLACEHOLDER]                       |

#### Mixtral-8x7B Model on 16 DGX Nodes (128 H100s)

| Framework                                       | Training Throughput (tokens/s/GPU) | Cost to Train 1T Tokens (USD)  | Tokens Trained for $100k (Billion)  |
|:------------------------------------------------|-----------------------------------:|-------------------------------:|------------------------------------:|
| **Fast-LLM**[^fast-llm-mixtral]                 | [PLACEHOLDER]                      | **[PLACEHOLDER]**              | **[PLACEHOLDER]**                   |
| NVIDIA Megatron[^megatron-mixtral]              | [PLACEHOLDER]                      | [PLACEHOLDER]                  | [PLACEHOLDER]                       |
| MosaicML Composer[^mosaic-mixtral]              | [PLACEHOLDER]                      | [PLACEHOLDER]                  | [PLACEHOLDER]                       |
| Hugging Face Transformers[^huggingface-mixtral] | [PLACEHOLDER]                      | [PLACEHOLDER]                  | [PLACEHOLDER]                       |
| Meta Lingua[^metaligua-mixtral]                 | not supported                      | not supported                  | not supported                       |

### Key Takeaways

-   **Cost efficiency at all scales:** Fast-LLM consistently achieves lower training costs due to its advanced parallelism and memory efficiency, delivering value across various model sizes and hardware configurations.
-   **Superior token throughput:** By processing more tokens per second per GPU than other frameworks, Fast-LLM maximizes token efficiency, leading to substantial savings, particularly for longer training durations or larger GPU clusters.
-   **Optimized for large-scale training:** Fast-LLM's design allows it to scale effectively as model size and training setups expand, ensuring that the benefits of its optimizations grow with the size of the deployment.

[^fast-llm-1b]:
    Testing conducted in [Month, Year] using 8 NVIDIA H100 SXM5 80 GB GPUs in 1 DGX node connected with 3200 Gbps Infiniband. Fast-LLM version [VERSION/COMMIT HASH], CUDA version [VERSION]. Training was performed on randomly generated data. Configuration file: [Link to config file].

[^megatron-1b]:
    Testing conducted in [Month, Year] using 8 NVIDIA H100 SXM5 80 GB GPUs in 1 DGX node connected with 3200 Gbps Infiniband. NVIDIA Megatron version [VERSION], CUDA version [VERSION]. Training was performed on randomly generated data. Configuration file: [Link to config file].

[^mosaic-1b]:
    Testing conducted in [Month, Year] using 8 NVIDIA H100 SXM5 80 GB GPUs in 1 DGX node connected with 3200 Gbps Infiniband. MosaicML Composer version [VERSION], CUDA version [VERSION]. Training was performed on randomly generated data. Configuration file: [Link to config file].

[^huggingface-1b]:
    Testing conducted in [Month, Year] using 8 NVIDIA H100 SXM5 80 GB GPUs in 1 DGX node connected with 3200 Gbps Infiniband. Hugging Face Transformers version [VERSION], CUDA version [VERSION]. Training was performed on randomly generated data. Configuration file: [Link to config file].

[^metaligua-1b]:
    Testing conducted in [Month, Year] using 8 NVIDIA H100 SXM5 80 GB GPUs in 1 DGX node connected with 3200 Gbps Infiniband. Meta Lingua version [VERSION], CUDA version [VERSION]. Training was performed on randomly generated data. Configuration file: [Link to config file].

[^fast-llm-8b]:
    Testing conducted in [Month, Year] using 32 NVIDIA H100 SXM5 80 GB GPUs across 4 DGX nodes connected with 3200 Gbps Infiniband. Fast-LLM version [VERSION/COMMIT HASH], CUDA version [VERSION]. Training was performed on randomly generated data. Configuration file: [Link to config file].

[^megatron-8b]:
    Testing conducted in [Month, Year] using 32 NVIDIA H100 SXM5 80 GB GPUs across 4 DGX nodes connected with 3200 Gbps Infiniband. NVIDIA Megatron version [VERSION], CUDA version [VERSION]. Training was performed on randomly generated data. Configuration file: [Link to config file].

[^mosaic-8b]:
    Testing conducted in [Month, Year] using 32 NVIDIA H100 SXM5 80 GB GPUs across 4 DGX nodes connected with 3200 Gbps Infiniband. MosaicML Composer version [VERSION], CUDA version [VERSION]. Training was performed on randomly generated data. Configuration file: [Link to config file].

[^huggingface-8b]:
    Testing conducted in [Month, Year] using 32 NVIDIA H100 SXM5 80 GB GPUs across 4 DGX nodes connected with 3200 Gbps Infiniband. Hugging Face Transformers version [VERSION], CUDA version [VERSION]. Training was performed on randomly generated data. Configuration file: [Link to config file].

[^metaligua-8b]:
    Testing conducted in [Month, Year] using 32 NVIDIA H100 SXM5 80 GB GPUs across 4 DGX nodes connected with 3200 Gbps Infiniband. Meta Lingua version [VERSION], CUDA version [VERSION]. Training was performed on randomly generated data. Configuration file: [Link to config file].

[^fast-llm-mixtral]:
    Testing conducted in [Month, Year] using 128 NVIDIA H100 SXM5 80 GB GPUs across 16 DGX nodes connected with 3200 Gbps Infiniband. Fast-LLM version [VERSION/COMMIT HASH], CUDA version [VERSION]. Training was performed on randomly generated data. Configuration file: [Link to config file].

[^megatron-mixtral]:
    Testing conducted in [Month, Year] using 128 NVIDIA H100 SXM5 80 GB GPUs across 16 DGX nodes connected with 3200 Gbps Infiniband. NVIDIA Megatron version [VERSION], CUDA version [VERSION]. Training was performed on randomly generated data. Configuration file: [Link to config file].

[^mosaic-mixtral]:
    Testing conducted in [Month, Year] using 128 NVIDIA H100 SXM5 80 GB GPUs across 16 DGX nodes connected with 3200 Gbps Infiniband. MosaicML Composer version [VERSION], CUDA version [VERSION]. Training was performed on randomly generated data. Configuration file: [Link to config file].

[^huggingface-mixtral]:
    Testing conducted in [Month, Year] using 128 NVIDIA H100 SXM5 80 GB GPUs across 16 DGX nodes connected with 3200 Gbps Infiniband. Hugging Face Transformers version [VERSION], CUDA version [VERSION]. Training was performed on randomly generated data. Configuration file: [Link to config file].

[^metaligua-mixtral]:
    In [Month, Year], Meta Lingua did not support training this configuration.
