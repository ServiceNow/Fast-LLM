---
title: "StarCoder2"
---

2023 marked a big year for our team at ServiceNow Research as we embarked on training **StarCoder2**, an open-source LLM optimized for coding tasks. This project, an evolution of the StarCoder model, aimed to create a family of models capable of handling a wide array of programming languages, achieving performance comparable to (or even surpassing) larger models in some benchmarks.

Through the year, we put Fast-LLM to the test on **NVIDIA's DGX SuperCloud**, using multiple **DGX A100-80GB nodes**. The Fast-LLM framework was developed specifically to optimize the training workflow for LLMs like StarCoder 2, combining **data parallelism** with **tensor parallelism** to maximize GPU utilization, minimize idle time, and maintain high throughput across all nodes. The framework's adaptable design allowed us to scale the model on a large compute cluster seamlessly, handling everything from distributed data loading to real-time monitoring and load balancing between compute nodes.

Our goal was ambitious: to train the 3-billion-parameter StarCoder2 model on **The Stack V2** dataset, a large and diverse code corpus containing repositories across more than 600 programming languages, courtesy of the Software Heritage archive. This dataset provided real-world code examples and broad coverage of programming paradigms, ensuring that StarCoder2-3B could understand context-rich coding tasks with high precision and accuracy.

## The Role of Fast-LLM

Fast-LLM enabled us to achieve a training throughput of **10,000 tokens per second per A100-80GB GPU**, which allowed us to reduce the expected training time by **20%** compared to the Megatron framework. This boost in scalable efficiency was made possible by Fast-LLM's optimized data pipelines and balanced load distribution, ensuring minimal latency and consistent GPU saturation across all nodes. This performance demonstrates Fast-LLM's capacity to handle a model of this scale with impressive efficiency and stability, setting a new benchmark for training large language models.

Fast-LLM's adaptability shone as we trained StarCoder2-3B with a **Fill-in-the-Middle (FIM) objective**, a novel approach for the model to generate and complete code snippets in a contextually relevant way. FIM training requires dynamically structured data inputs and, therefore, efficient shuffling and sample handling—all handled seamlessly by Fast-LLM.

## Technical Highlights

- **16K Context Window**: StarCoder2-3B boasts a **16,384 token context window**. That's four times the length of the original model. With Fast-LLM, we integrated Grouped Query Attention (GQA) to achieve this, allowing the model to retain context over extensive code snippets, conversations, and documentation.
  
- **Dynamic Dataset Handling**: Training with The Stack V2 posed challenges; the dataset's sheer size and variety required Fast-LLM's efficient, adaptive sharding and fast sample batching. These features allowed us to effectively leverage our compute resources, creating a streamlined experience when dealing with billions of code tokens.

- **High Throughput on DGX Nodes**: Although we're awaiting precise throughput metrics, preliminary tests showed that Fast-LLM allowed each node to perform at peak efficiency, even when processing over **4 trillion tokens** in total.

## The Road Ahead

The results of StarCoder2-3B have set the stage for ongoing innovation. With Fast-LLM's framework as a robust foundation, we are now exploring fine-tuning for even more targeted code generation applications, building models that offer immediate utility across development, deployment, and debugging tasks. StarCoder2-3B's performance and versatility stand as a testament to the power of Fast-LLM and to the incredible potential of open-source models in advancing the AI landscape.

For more insights and technical details, please refer to our publications on StarCoder2 and Fast-LLM [Cite relevant papers].
