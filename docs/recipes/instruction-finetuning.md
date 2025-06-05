---
title: Instruction Finetuning on Llama 3.1 8B
---


In this guide, we provide step-by-step instructions to do a supervised finetuning (SFT) of the Llama 3.1 8B model on an instruction tuning dataset [Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered](https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered). Typically, SFT differs from pretraining in 2 key ways:

1. **Loss Computation**: Instruction tuning datasets typically contain system, user and assistant messages formatted using a chat template e.g., [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md). The loss is typically computed only on the tokens from the assistant messages, and hence the gradient updates do not include the system or user tokens.
2. **Packing**: Fast-LLM packs multiple documents into a sequence to maximize training throughput. However, paying attention to other documents in a sequence can detrimental to instruction-finetuned models. Moreover, packing to sequence length can mean some document are split across multiple sequences.

Fast-LLM provides options to modify both of these, as we will see in this guide. Please follow the [Quick Start](../quick-start.md) guide for the initial setup.

## 📚 Step 1: Download the Pretrained Model

```bash
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.1-8B ./sft-tutorial/Llama-3.1-8B
```

## 🔄 Step 2: Format the dataset into a chat template

We'll use [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)'s tokenizer for this tutorial.
Let's create a folder first:

```bash
mkdir -p ./sft-tutorial/checkpoints/Llama-3.1-8B-Instruct
```

And then download the tokenizer with this Python script:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("./sft-tutorial/Llama-3.1-8B-Instruct")
```

To disable loss computation on pieces of texts like the user/system messages or the chat template tags, we need to define the character spans for this masking. The example below formats the Magpie dataset using Llama-3.1-8B-Instruct's chat template and defines spans where the loss should be masked.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

def apply_chat_template(conversation):
    chatml_conv = []
    for conv in conversation:
        if conv["from"] == "human":
            chatml_conv.append({"role": "user", "content": conv["value"]})
        elif conv["from"] == "gpt":
            chatml_conv.append({"role": "assistant", "content": conv["value"]})
    return tokenizer.apply_chat_template(conversation=chatml_conv, tokenize=False)


def get_spans(text: str, start_delimiter: str, end_delimiter: str):
    spans = []
    start = 0
    while start < len(text):
        end = text.find(start_delimiter, start) + len(start_delimiter)
        if end == -1:
            break
        spans.append([start, end - 1])  # character span indices are inclusive
        start = text.find(end_delimiter, end) + len(end_delimiter)
    return spans


def get_sample_with_spans(sample, start_delimiter, end_delimiter):
    text = apply_chat_template(sample["conversations"])
    spans = get_spans(text, start_delimiter, end_delimiter)
    return {"text": text, "spans": spans}

dataset_name = "Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered"
tokenizer_path = "./sft-tutorial/Llama-3.1-8B-Instruct"
dataset = load_dataset(dataset_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

start_delimiter = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
end_delimiter = "<|eot_id|>"
dataset = dataset.map(lambda x: get_sample_with_spans(x, start_delimiter, end_delimiter), num_proc=16)
dataset.save_to_disk("./sft-tutorial/chatml_dataset")
```

## 🔍 Step 3: Prepare the dataset

This step is similar to [Data preparation](data-preparation.md), with one additional option for the field with loss masking spans.

```yaml
output_path: ./sft-tutorial/tokenized/Llama-3.1-8B

loading_workers: 32
tokenize_workers: 32
saving_workers: 32

dataset:
  path: ./sft-tutorial/hf_dataset
  split: "train"
  trust_remote_code: true
  field: text
  loss_masking_spans: spans

tokenizer:
  path: ./sft-tutorial/Llama-3.1-8B
splits:
  training: 0.998
  validation: 0.002
```

## ⚙️ Step 4: Configure Fast-LLM

It's time to configure the Fast-LLM training config. This is very similar to [Quick Start](../quick-start.md) with two additional options, namely, `truncate_documents` and `cross_document_attention` which are important for improving the task performance of instruction-tuned models.

```yaml
training:
  train_iters: 5_000
  logs:
    interval: 1
  evaluators:
    validation:
      interval: 100
      evaluator:
        type: loss
        iterations: 25
        dataset_name: validation
  checkpoint:
    interval: 1000
    keep: 5
  test_iters: 0
  export:
    format: llama
    interval: 1000
batch:
  micro_batch_size: 1
  sequence_length: 4096
  batch_size: 32
  cross_document_attention: no # (1)!
data:
  datasets:
    training:
      type: file
      path: ./sft-tutorial/tokenized/Llama-3.1-8B/fast_llm_config_training.yaml
    validation:
      type: file
      path: ./sft-tutorial/tokenized/Llama-3.1-8B/fast_llm_config_validation.yaml
  truncate_documents: no # (2)!
  sampling:
    use_loss_masking_spans: yes
optimizer:
  weight_decay: 0.1
  beta_1: 0.9
  beta_2: 0.95
  learning_rate:
    base: 2.0e-05
    minimum: 0.0
    decay_style: cosine
    decay_iterations: 4_900
    warmup_iterations: 100
pretrained:
  format: llama
  path: ./sft-tutorial/Llama-3.1-8B
  model_weights: yes
model:
  base_model:
    transformer:
      use_flash_attention: yes
    cross_entropy_impl: fused
  multi_stage:
    zero_stage: 3
  distributed:
    timeout: 3600
    training_dtype: bf16
run:
  experiment_dir: ./sft-tutorial/llama-3.1-8b-instruct-magpie
```

1. Prevents paying attention to other documents in a packed sequence
2. Avoids truncating documents to fit into a packed sequence and starts a new sequence instead. Documents longer than sequence length will be skipped altogether.

Launching the training run is similar to Step 7 in the [Quick Start](../quick-start.md) guide.
