import os

import numpy as np
from datasets import load_dataset
from packed_dataset import PackedDatasetBuilder  # Make sure this is accessible as a module
from tqdm import tqdm
from transformers import AutoTokenizer

# if "HF_HOME" not in os.environ:
#     os.environ["HF_HOME"] = "/mnt/transformers_cache/"
# print(f"HF HOME: {os.environ.get('HF_HOME', 'Not set')}")

# === Config ===
tokenizer_name = "/mnt/checkpoints/diffusion_models/SmolLM2-135M-MASK_TOKEN"
chunk_size = 2**30  # 1GB  # 2**20 # 2**18  # 256K tokens per file
output_dir = "/mnt/datasets/tokenized/SmolLM2-135M/packed_fineweb_sample_350B_largefiles_job"
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
    parallel_write=True,
    max_workers=64,
)

print(f"sep_token: {sep_token}, vocab_size: {vocab_size}")

# === Load and tokenize dataset ===
# stream = load_dataset("HuggingFaceFW/fineweb", name="sample-350BT", split="train", streaming=True,
#                       cache_dir="/mnt/transformers_cache/datasets")
# dataset = []
# num_samples = 0
# max_samles = 100000
# for sample in stream:
#     dataset.append(sample)
#     num_samples += 1
#     if num_samples >= max_samles:
#         break

dataset = load_dataset(
    dataset_name,
    dataset_config,
    split=split,
    trust_remote_code=True,
    num_proc=8,
    cache_dir="/mnt/transformers_cache/datasets",
)

print(f"Dataset loaded: {len(dataset)} samples")

# === Tokenize and pack dataset ===
total_tokens = 0
for ex in tqdm(dataset, desc="Packing dataset"):
    text = ex[input_col]
    if not text.strip():
        continue
    ids = tokenizer(text, add_special_tokens=False).input_ids
    ids.append(sep_token)  # always separate sequences
    builder.add_array(np.array(ids, dtype=builder.dtype))
    total_tokens += len(ids)


# === Finalize ===
builder.write_reminder()
print(f"Total tokens packed: {total_tokens}")
print("Done writing packed dataset.")
print(f"Chunks written: {len(builder.filenames)}")
