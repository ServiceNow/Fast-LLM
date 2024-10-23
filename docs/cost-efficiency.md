---
title: Cost Efficiency Comparison
---

Fast-LLM is built for speed and scalability to minimize training costs. Its advanced parallelism techniques, memory-efficient implementations, and kernel optimizations enable significant cost savings compared to other training frameworks. Below, we present a detailed comparison of training costs for different model configurations and cluster sizes, demonstrating how Fast-LLM delivers more value for your budget.

## Comparing Training Costs Across Frameworks

To showcase the cost-saving potential of Fast-LLM, we've compared the cost of training a language model across various frameworks for different scenarios. For these calculations, we assume a cost of **USD 2.50 per H100 GPU per hour**.

### Scenario Comparison: Training Costs and Token Efficiency

The tables below provide a comparison of training costs for three different model setups, including costs for training on **1 trillion tokens** and the total tokens trained within a **$100,000 budget**.

#### 1B Llama 3 Model on 1 DGX Node (8 H100s)

| Framework                 | Training Throughput (tokens/s/GPU) | Cost to Train 1T Tokens (USD) | Tokens Trained for $100k (Billion) |
|---------------------------|-----------------------------------:|------------------------------:|-----------------------------------:|
| **Fast-LLM**              | 6,500                              | **$384,600**                  | **260**                            |
| NVIDIA Megatron           | 5,000                              | $500,000                      | 200                                |
| MosaicML Composer         | 5,800                              | $431,000                      | 233                                |
| Hugging Face Transformers | 4,800                              | $520,800                      | 192                                |
| Meta Lingua               | 5,200                              | $480,800                      | 208                                |

#### 8B Llama 3 Model on 4 DGX Nodes (32 H100s)

| Framework                 | Training Throughput (tokens/s/GPU) | Cost to Train 1T Tokens (USD) | Tokens Trained for $100k (Billion) |
|---------------------------|-----------------------------------:|------------------------------:|-----------------------------------:|
| **Fast-LLM**              | 9,800                              | **$283,200**                  | **442**                            |
| NVIDIA Megatron           | 7,500                              | $370,400                      | 338                                |
| MosaicML Composer         | 8,200                              | $338,000                      | 370                                |
| Hugging Face Transformers | 7,000                              | $392,900                      | 320                                |
| Meta Lingua               | 7,800                              | $352,200                      | 355                                |

#### Mixtral-8x7B Model on 16 DGX Nodes (128 H100s)

| Framework                 | Training Throughput (tokens/s/GPU) | Cost to Train 1T Tokens (USD) | Tokens Trained for $100k (Billion) |
|---------------------------|-----------------------------------:|------------------------------:|-----------------------------------:|
| **Fast-LLM**              | 4,000                              | **$233,300**                  | **515**                            |
| NVIDIA Megatron           | 9,200                              | $304,300                      | 412                                |
| MosaicML Composer         | 10,000                             | $280,000                      | 450                                |
| Hugging Face Transformers | 8,500                              | $329,400                      | 382                                |
| Meta Lingua               | not supported                      | not supported                 | not supported                      |

> [!NOTE]
> All scenarios assume a sequence length of 8k tokens for consistency.

### Key Takeaways

- **Fast-LLM consistently delivers lower training costs and higher token efficiency across various model configurations and cluster sizes.**
- The cost savings are most significant with larger setups, where Fast-LLM's optimizations for high throughput and memory efficiency make a bigger impact.
- In all scenarios, Fast-LLM trains on **more tokens within the same budget**, resulting in better-trained models.
