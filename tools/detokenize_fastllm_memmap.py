#!/usr/bin/env python
"""
Utility to detokenize Fast-LLM memmap datasets back into text.

Example:
PYTHONPATH=. python tools/detokenize_fastllm_memmap.py \\
    --dataset-dir /mnt/datasets/tokenized/Mistral-Nemo-Base-2407/text/9_12_25_14b_oss_sft/fastllm \\
    --tokenizer mistralai/Mistral-Nemo-Base-2407 \\
    --output __tmp/detokenized_sample.txt \\
    --max-docs 5
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Iterable

import yaml
from transformers import AutoTokenizer

from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset


def load_memmap_datasets(root: pathlib.Path) -> list[GPTMemmapDataset]:
    """
    Build GPTMemmapDataset objects for every shard listed in fast_llm_config.yaml.
    """
    cfg_path = root / "fast_llm_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    datasets = []
    for i, ds_cfg in enumerate(cfg.get("datasets", [])):
        prefix = root / ds_cfg["path"]
        datasets.append(
            GPTMemmapDataset(
                name=f"{ds_cfg.get('path', f'shard_{i}')}",
                prefix=prefix,
                num_documents=ds_cfg.get("num_documents"),
                num_tokens=ds_cfg.get("num_tokens"),
                num_pixels=ds_cfg.get("num_pixels"),
            )
        )
    return datasets


def detokenize(
    datasets: Iterable[GPTMemmapDataset],
    tokenizer,
    output_path: pathlib.Path,
    start: int,
    max_docs: int,
    skip_special_tokens: bool,
) -> None:
    """
    Stream decoded documents to an output file.
    """
    written = 0
    to_skip = start
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for shard_idx, dataset in enumerate(datasets):
            for doc_idx in range(len(dataset)):
                if to_skip:
                    to_skip -= 1
                    continue
                sample = dataset.get(doc_idx)
                text = tokenizer.decode(
                    sample.token_ids.tolist(),
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=False,
                )
                handle.write(f"### shard={shard_idx} name={dataset.name} doc={doc_idx}\n")
                handle.write(text)
                handle.write("\n\n")
                written += 1
                if max_docs != -1 and written >= max_docs:
                    return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detokenize Fast-LLM memmap datasets to text.")
    parser.add_argument(
        "--dataset-dir",
        type=pathlib.Path,
        required=False,
        default="/mnt/datasets/tokenized/Mistral-Nemo-Base-2407/text/9_12_25_14b_oss_sft/fastllm",
        help="Directory containing fast_llm_config.yaml and shard_*.bin/.idx files.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=False,
        # default="/mnt/checkpoints/upstream/Apriel-1.5-15b-Thinker",
        default="/mnt/checkpoints/upstream/Mistral-Nemo-Base-2407",
        help="Hugging Face tokenizer name or path to load locally (e.g., mistralai/Mistral-Nemo-Base-2407).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=False,
        default="__tmp/detokenized_sample_mistral_tokenizer.txt",
        help="Where to write decoded text.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Number of documents to skip before decoding.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=5,
        help="Total number of documents to decode across shards. Use -1 to decode everything (large!).",
    )
    parser.add_argument(
        "--keep-special-tokens",
        action="store_true",
        help="Keep BOS/EOS and other special tokens in the output.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading the tokenizer if it's not cached locally.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=not args.allow_download,
    )

    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer must define both BOS and EOS tokens to decode these datasets.")

    datasets = load_memmap_datasets(args.dataset_dir)
    detokenize(
        datasets=datasets,
        tokenizer=tokenizer,
        output_path=args.output,
        start=args.start,
        max_docs=args.max_docs,
        skip_special_tokens=not args.keep_special_tokens,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
