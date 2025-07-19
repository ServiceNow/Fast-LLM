import multiprocessing
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from fast_llm.diffullama.packing_utils import pack_worker

# === Config ===
num_proc = 128
chunk_size = 2**28
tokenizer_name = "/mnt/checkpoints/diffusion_models/SmolLM2-135M-MASK_TOKEN"

# output_dir = "/mnt/datasets/tokenized/SmolLM2-135M/packed_wikitext_largefiles_test_divded"
# prefix = "wikitext_sample"
# dataset_name = "Salesforce/wikitext"
# dataset_config = "wikitext-2-v1"
# split = "train"
# input_col = "text"

output_dir = "/mnt/datasets/tokenized/SmolLM2-135M/packed_fineweb_350B_smallfiles_parellel_blocks"
prefix = "fineweb_sample"
dataset_name = "HuggingFaceFW/fineweb"
dataset_config = "sample-350BT"
split = "train"
input_col = "text"


# ...existing code...


def main():
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
        num_proc=num_proc,
        cache_dir="/mnt/hf_home",
    )
    print(f"Dataset loaded with {len(dataset)} examples.")

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
        num_proc=num_proc,
        desc="Tokenizing",
        load_from_cache_file=True,
    )

    # Split tokenized_dataset into num_proc blocks
    blocks = np.array_split(tokenized_dataset, 64)
    print(f"Total blocks created: {len(blocks)}")
    builder_args = {
        "outdir": output_dir,
        "prefix": prefix,
        "chunk_size": chunk_size,
        "sep_token": sep_token,
        "vocab_size": vocab_size,
        "dtype": "auto",
        "parallel_write": True,
        "max_workers": 2,
    }

    # Assign a unique starting counter for each worker
    # chunk_size_per_worker = len(tokenized_dataset) // num_proc
    start_counters = [i * 100000 for i in range(num_proc)]  # Each worker starts at a different 5-digit offset

    with multiprocessing.Pool(num_proc) as pool:
        results = pool.starmap(pack_worker, [(blk, builder_args, start_counters[i]) for i, blk in enumerate(blocks)])

    # Merge filenames and token counts
    all_filenames = []
    total_tokens = 0
    for filenames, tokens in results:
        all_filenames.extend(filenames)
        total_tokens += tokens

    print(f"Total tokens packed: {total_tokens}")
    print("Done writing packed dataset.")

    print(f"Chunks written: {len(all_filenames)}")


if __name__ == "__main__":
    main()
