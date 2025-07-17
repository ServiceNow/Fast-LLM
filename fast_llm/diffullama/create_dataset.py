import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from fast_llm.diffullama.packed_dataset import PackedDatasetBuilder  # Update import if needed

# === Config ===
tokenizer_name = "/mnt/checkpoints/diffusion_models/SmolLM2-135M-MASK_TOKEN"
chunk_size = 2**29  # 512MB
output_dir = "/mnt/datasets/tokenized/SmolLM2-135M/packed_fineweb_350B_largefiles"
prefix = "fineweb_sample"
dataset_name = "HuggingFaceFW/fineweb"
dataset_config = "sample-350BT"
split = "train"
input_col = "text"  # The column in the dataset that contains the text data

# === Setup ===
os.makedirs(output_dir, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

sep_token = tokenizer.eos_token_id
vocab_size = tokenizer.vocab_size

builder = PackedDatasetBuilder(
    outdir=output_dir,
    prefix=prefix,
    chunk_size=chunk_size,
    sep_token=sep_token,
    vocab_size=vocab_size,
    dtype="auto",
    parallel_write=False,
)

print(f"sep_token: {sep_token}, vocab_size: {vocab_size}")

dataset = load_dataset(
    dataset_name,
    dataset_config,
    split=split,
    trust_remote_code=True,
    num_proc=128,
    cache_dir="/mnt/hf_home",
)


def tokenize_and_pack(example):
    text = example[input_col]
    if not text.strip():
        return {"ids": []}
    ids = tokenizer(text, add_special_tokens=False).input_ids
    ids.append(sep_token)
    return {"ids": ids}


print("Tokenizing and packing dataset with multiprocessing...")
tokenized_dataset = dataset.map(
    tokenize_and_pack,
    num_proc=128,
    desc="Tokenizing",
)

total_tokens = 0
for ex in tqdm(tokenized_dataset, desc="Packing dataset"):
    if not ex["ids"]:
        continue
    builder.add_array(np.array(ex["ids"], dtype=builder.dtype))
    total_tokens += len(ex["ids"])

builder.write_reminder()
print(f"Total tokens packed: {total_tokens}")
print("Done writing packed dataset.")
print(f"Chunks written: {len(builder.filenames)}")
