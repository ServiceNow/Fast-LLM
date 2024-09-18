# Tutorial


This guide will teach how to pretrain and/or extend pretraining of language models such as Mistral-7B with Fast-LLM on multiple GPU nodes.
Such training requires a careful selection and optimization of:
- The training hardware: GPU node specs, count and interconnect.
- The model architecture: layer types, hidden sizes, activations, etc.
- The training dataset and its sampling.
- The training parameters: optimizer, learning rate schedule, training duration, etc.
- The training performance optimizations: distributed layout, activation recomputation, etc.

When training a model with Fast LLM (and other training libraries),
we generally assume the first four points to be predetermined as they are unrelated to the training framework,
and focus on the last one, i.e., we optimize a fixed training scheme for throughput.
(However, in practice the batch size may be adjusted together with the distributed layout,
which in turn affects the training schedule.)

In this tutorial, we follow the extended pretraining for Mistral-7B over a corpus of 500 billion tokens using 16 DGX nodes,
each equipped with 8 A100 or H100 GPUs (totalling 128 GPUs).
We also explore some alternative settings such as training from scratch and the Mixtral-8x7B model.


- [Getting started](getting_started.md): Get started with Fast-LLM, set up and run a first training configuration.
- [Load Mistral-7B](prepare_mistral.md): Define the model architecture, download a checkpoint from the Huggingface Hub and load it in Fast-LLM.
- [Prepare and load the dataset](prepare_data.md): Prepare and configure the dataset.
- [Prepare the training configuration](prepare_training.md): Configure the optimizer, schedule, distributed layout, etc.
- [Launch and monitor training](launch_training.md): Launch training, configure and view experiment outputs.
- [Convert to Hugging Face](convert_to_huggingface.md): Convert to Hugging Face format and upload it to the Hugging Face model hub.
