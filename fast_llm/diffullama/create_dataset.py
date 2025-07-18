import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from fast_llm.diffullama.packed_dataset import PackedDatasetBuilder  # Update import if needed

# === Config ===
tokenizer_name = "/mnt/checkpoints/diffusion_models/SmolLM2-135M-MASK_TOKEN"
chunk_size = 2**29  # 512MB

# HF dataset configuration``
output_dir = "/mnt/datasets/tokenized/SmolLM2-135M/packed_fineweb_350B_largefiles_parellel"
prefix = "fineweb_sample"
dataset_name = "HuggingFaceFW/fineweb"
dataset_config = "sample-350BT"
split = "train"
input_col = "text"  # The column in the dataset that contains the text data

# output_dir = "/mnt/datasets/tokenized/SmolLM2-135M/packed_wikitext_largefiles_test"
# prefix = "wikitext_sample"
# dataset_name = "Salesforce/wikitext"
# dataset_config = "wikitext-2-v1"
# split = "train"
# input_col = "text"  # The column in the dataset that contains the text data

# === Setup ===
os.makedirs(output_dir, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

sep_token = tokenizer.eos_token_id
vocab_size = tokenizer.vocab_size

print(f"sep_token: {sep_token}, vocab_size: {vocab_size}")

print(f"Loading dataset: {dataset_name} ({dataset_config})")
dataset = load_dataset(
    dataset_name,
    dataset_config,
    split=split,
    trust_remote_code=True,
    num_proc=128,
    cache_dir="/mnt/hf_home", # "/mnt/transformers_cache/"
)
print(f"Dataset loaded with {len(dataset)} examples.")

# multi process tokenization function
def tokenize_and_pack(example):
    # Re-initialize tokenizer inside the function for multiprocessing safety
    tokenizer_local = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer_local.pad_token is None:
        tokenizer_local.pad_token = tokenizer_local.eos_token
    text = example[input_col]
    if not text.strip():
        return {"ids": []}
    ids = tokenizer_local(text, add_special_tokens=False).input_ids
    ids.append(sep_token)
    return {"ids": ids}


print("Tokenizing and packing dataset with multiprocessing...")
tokenized_dataset = dataset.map(
    tokenize_and_pack,
    num_proc=128,
    desc="Tokenizing",
    load_from_cache_file=True,
)

print("Tokenization complete. Packing dataset...")

# builder created after multi process ends so build can use threads to save files.
builder = PackedDatasetBuilder(
    outdir=output_dir,
    prefix=prefix,
    chunk_size=chunk_size,
    sep_token=sep_token,
    vocab_size=vocab_size,
    dtype="auto",
    parallel_write=True,
    max_workers=128,  # Adjust based on your system's capabilities
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
