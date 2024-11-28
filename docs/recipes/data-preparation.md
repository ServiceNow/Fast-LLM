---
title: Preparing Data for Training
---

This guide will demonstrate how to prepare data for Fast-LLM starting from a huggingface dataset.

## Prerequisites

## 📚 Step 1: Download the dataset from Huggingface

First, set `HF_HOME` to your Huggingface cache folder.

Let's create the folder to store the huggingface dataset
```bash
mkdir -p ~/datasets/upstream/the-stack
```

Next we download the Stack dataset from huggingface.
```bash
huggingface-cli download bigcode/the-stack --revision v1.2 --repo-type dataset --max_workers 64 --local-dir /mnt/datasets/upstream/the-stack
```

!!! warning "Choice of num_workers"

    Setting a large num_workers sometimes leads to connection errors.

## ⚙️ Step 2: Prepare the configs for conversion of data to gpt_mmap format

In this step, we tokenize the huggingface dataset downloaded in Step 1 and save it in the gpt_mmap format that Fast-LLM accepts.

We'll use Mistral-Nemo-Base-2407 tokenizer. Let's create folder first
```bash
mkdir -p ~/checkpoints/upstream/Mistral-Nemo-Base-2407
```

And then download the tokenizer with this script
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mistral-Nemo-Base-2407"
tokenizer = AutoTokenizer.from_pretrained(model_id) 
tokenizer.save_pretrained("./models/tokenizer/")


Let's create a folder to store the gpt_mmap dataset
```


```bash
mkdir -p ~/datasets/tokenized/Mistral-Nemo-Base-2407
```

Create a config like this - 

```yaml
output_path: /mnt/datasets/tokenized/Mistral-Nemo-Base-2407/the-stack/python

loading_workers: 32
tokenize_workers: 32
saving_workers: 32

dataset:
  path: /mnt/datasets/upstream/the-stack
  config_name: "python"
  split: "train"

tokenizer:
  path: /mnt/checkpoints/upstream/Mistral-Nemo-Base-2407/tokenizer.json
```




