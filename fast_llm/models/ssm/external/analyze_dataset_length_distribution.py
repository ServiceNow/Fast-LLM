#!/usr/bin/env python3
"""
Working script to analyze document length distribution from a tokenized dataset.
"""

import argparse
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import Fast-LLM dataset classes
from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_shard_document_lengths(shard_prefix: pathlib.Path) -> np.ndarray:
    """
    Analyze document lengths in a single shard.

    Args:
        shard_prefix: Path prefix for the shard (without .bin/.idx extension)

    Returns:
        Array of document lengths in tokens
    """
    try:
        # Load the dataset shard
        dataset = GPTMemmapDataset(name=shard_prefix.name, prefix=shard_prefix)

        logger.info(
            f"Loaded shard {shard_prefix.name}: {dataset._num_documents:,} documents, {dataset.num_tokens:,} tokens"
        )

        # Get document lengths directly from the document sizes array
        doc_lengths = dataset._document_sizes.copy()

        logger.info(f"  Mean doc length: {doc_lengths.mean():.1f} tokens")
        logger.info(f"  Median doc length: {np.median(doc_lengths):.1f} tokens")
        logger.info(f"  Min doc length: {doc_lengths.min()} tokens")
        logger.info(f"  Max doc length: {doc_lengths.max()} tokens")

        return doc_lengths

    except Exception as e:
        logger.error(f"Error processing shard {shard_prefix}: {e}")
        return np.array([])


def find_dataset_files(dataset_dir: pathlib.Path):
    """Find all .bin/.idx file pairs in a directory."""
    dataset_dir = pathlib.Path(dataset_dir)
    bin_files = list(dataset_dir.glob("*.bin"))

    file_pairs = []
    for bin_file in bin_files:
        idx_file = bin_file.with_suffix(".idx")
        if idx_file.exists():
            file_pairs.append((bin_file.with_suffix(""), bin_file))

    logger.info(f"Found {len(file_pairs)} dataset file pairs in {dataset_dir}")
    return file_pairs


def analyze_dataset_from_directory(dataset_dir: pathlib.Path, max_shards: int = None):
    """
    Analyze document lengths from a dataset directory.
    """
    file_pairs = find_dataset_files(dataset_dir)

    all_doc_lengths = []
    shard_stats = []

    pairs_to_process = file_pairs[:max_shards] if max_shards else file_pairs
    logger.info(f"Processing {len(pairs_to_process)} shards...")

    for shard_prefix, _ in tqdm(pairs_to_process, desc="Processing shards"):
        doc_lengths = analyze_shard_document_lengths(shard_prefix)

        if len(doc_lengths) > 0:
            all_doc_lengths.extend(doc_lengths)

            # Calculate shard statistics
            shard_stat = {
                "shard_name": shard_prefix.name,
                "num_documents": len(doc_lengths),
                "total_tokens": doc_lengths.sum(),
                "mean_length": doc_lengths.mean(),
                "median_length": np.median(doc_lengths),
                "std_length": doc_lengths.std(),
                "min_length": doc_lengths.min(),
                "max_length": doc_lengths.max(),
                "q25_length": np.percentile(doc_lengths, 25),
                "q75_length": np.percentile(doc_lengths, 75),
            }
            shard_stats.append(shard_stat)

    all_doc_lengths = np.array(all_doc_lengths)

    if len(all_doc_lengths) == 0:
        logger.error("No documents found!")
        return None

    # Calculate overall statistics
    overall_stats = {
        "total_documents": len(all_doc_lengths),
        "total_tokens": all_doc_lengths.sum(),
        "mean_length": all_doc_lengths.mean(),
        "median_length": np.median(all_doc_lengths),
        "std_length": all_doc_lengths.std(),
        "min_length": all_doc_lengths.min(),
        "max_length": all_doc_lengths.max(),
        "q25_length": np.percentile(all_doc_lengths, 25),
        "q75_length": np.percentile(all_doc_lengths, 75),
        "q90_length": np.percentile(all_doc_lengths, 90),
        "q95_length": np.percentile(all_doc_lengths, 95),
        "q99_length": np.percentile(all_doc_lengths, 99),
    }

    return {"doc_lengths": all_doc_lengths, "overall_stats": overall_stats, "shard_stats": shard_stats}


def plot_length_distribution(doc_lengths: np.ndarray, output_dir: pathlib.Path):
    """Create visualization plots for document length distribution."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. Histogram of document lengths
    plt.figure(figsize=(12, 8))
    plt.hist(doc_lengths, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Document Length (tokens)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Document Lengths")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "doc_length_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Log-scale histogram
    plt.figure(figsize=(12, 8))
    plt.hist(doc_lengths, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Document Length (tokens)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Document Lengths (Log Scale)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "doc_length_histogram_log.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(doc_lengths, vert=True)
    plt.ylabel("Document Length (tokens)")
    plt.title("Box Plot of Document Lengths")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "doc_length_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Cumulative distribution
    sorted_lengths = np.sort(doc_lengths)
    cumulative_prob = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)

    plt.figure(figsize=(12, 8))
    plt.plot(sorted_lengths, cumulative_prob, linewidth=2)
    plt.xlabel("Document Length (tokens)")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution of Document Lengths")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "doc_length_cdf.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Create a detailed histogram with percentile lines
    plt.figure(figsize=(14, 8))
    plt.hist(doc_lengths, bins=100, alpha=0.7, edgecolor="black", density=True)

    # Add percentile lines
    percentiles = [25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(doc_lengths, percentiles)
    colors = ["orange", "red", "purple", "brown", "pink", "black"]

    for p, val, color in zip(percentiles, percentile_values, colors):
        plt.axvline(val, color=color, linestyle="--", alpha=0.8, label=f"{p}th percentile: {val:.0f} tokens")

    plt.xlabel("Document Length (tokens)")
    plt.ylabel("Density")
    plt.title("Document Length Distribution with Percentiles")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "doc_length_histogram_with_percentiles.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved visualization plots to {output_dir}")


def save_results(results, output_dir: pathlib.Path):
    """Save analysis results to files."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save overall statistics
    with open(output_dir / "overall_stats.txt", "w") as f:
        f.write("Document Length Distribution Analysis\n")
        f.write("=" * 50 + "\n\n")

        stats = results["overall_stats"]
        f.write(f"Total Documents: {stats['total_documents']:,}\n")
        f.write(f"Total Tokens: {stats['total_tokens']:,}\n")
        f.write(f"Mean Length: {stats['mean_length']:.2f} tokens\n")
        f.write(f"Median Length: {stats['median_length']:.2f} tokens\n")
        f.write(f"Standard Deviation: {stats['std_length']:.2f} tokens\n")
        f.write(f"Min Length: {stats['min_length']} tokens\n")
        f.write(f"Max Length: {stats['max_length']} tokens\n")
        f.write(f"25th Percentile: {stats['q25_length']:.2f} tokens\n")
        f.write(f"75th Percentile: {stats['q75_length']:.2f} tokens\n")
        f.write(f"90th Percentile: {stats['q90_length']:.2f} tokens\n")
        f.write(f"95th Percentile: {stats['q95_length']:.2f} tokens\n")
        f.write(f"99th Percentile: {stats['q99_length']:.2f} tokens\n")

    # Save shard statistics as CSV
    if results["shard_stats"]:
        shard_df = pd.DataFrame(results["shard_stats"])
        shard_df.to_csv(output_dir / "shard_stats.csv", index=False)

    # Save document lengths
    doc_lengths = results["doc_lengths"]
    np.savetxt(output_dir / "doc_lengths.txt", doc_lengths, fmt="%d")

    # Save percentile analysis
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    percentile_values = np.percentile(doc_lengths, percentiles)

    percentile_df = pd.DataFrame(
        {"Percentile": [f"{p}%" for p in percentiles], "Length (tokens)": percentile_values.astype(int)}
    )
    percentile_df.to_csv(output_dir / "percentiles.csv", index=False)

    # Save length buckets analysis
    bucket_ranges = [
        (0, 100),
        (100, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 5000),
        (5000, 10000),
        (10000, 20000),
        (20000, float("inf")),
    ]

    bucket_stats = []
    for min_len, max_len in bucket_ranges:
        if max_len == float("inf"):
            mask = doc_lengths >= min_len
            bucket_name = f"{min_len}+"
        else:
            mask = (doc_lengths >= min_len) & (doc_lengths < max_len)
            bucket_name = f"{min_len}-{max_len}"

        count = mask.sum()
        percentage = (count / len(doc_lengths)) * 100

        bucket_stats.append({"Length Range": bucket_name, "Document Count": count, "Percentage": f"{percentage:.2f}%"})

    bucket_df = pd.DataFrame(bucket_stats)
    bucket_df.to_csv(output_dir / "length_buckets.csv", index=False)

    logger.info(f"Saved results to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze document length distribution from tokenized dataset")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=False,
        default="/mnt/datasets/tokenized/Mistral-Nemo-Base-2407/ssm_distillation/deepseek-r1-0528-annotations/openthoughts_w_chat_template",
        help="Directory containing dataset shard files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./__length_analysis_results",
        help="Output directory for results (default: ./length_analysis_results)",
    )
    parser.add_argument(
        "--max_shards", type=int, default=None, help="Maximum number of shards to process (default: all)"
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")

    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return 1

    output_dir = pathlib.Path(args.output_dir)

    # Analyze dataset
    logger.info(f"Analyzing dataset from directory: {dataset_dir}")
    results = analyze_dataset_from_directory(dataset_dir, args.max_shards)

    if results is None:
        logger.error("Analysis failed!")
        return 1

    # Save results
    save_results(results, output_dir)

    # Generate plots
    if not args.no_plots:
        plot_length_distribution(results["doc_lengths"], output_dir)

    # Print summary
    stats = results["overall_stats"]
    logger.info(f"Analysis complete!")
    logger.info(f"Processed {stats['total_documents']:,} documents totaling {stats['total_tokens']:,} tokens")
    logger.info(f"Mean document length: {stats['mean_length']:.2f} tokens")
    logger.info(f"Median document length: {stats['median_length']:.2f} tokens")
    logger.info(f"Results saved to: {output_dir}")

    # Print detailed statistics
    print("\n" + "=" * 60)
    print("DOCUMENT LENGTH DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"Total Documents: {stats['total_documents']:,}")
    print(f"Total Tokens: {stats['total_tokens']:,}")
    print(f"Mean Length: {stats['mean_length']:.2f} tokens")
    print(f"Median Length: {stats['median_length']:.2f} tokens")
    print(f"Standard Deviation: {stats['std_length']:.2f} tokens")
    print(f"Min Length: {stats['min_length']} tokens")
    print(f"Max Length: {stats['max_length']} tokens")
    print(f"25th Percentile: {stats['q25_length']:.2f} tokens")
    print(f"75th Percentile: {stats['q75_length']:.2f} tokens")
    print(f"90th Percentile: {stats['q90_length']:.2f} tokens")
    print(f"95th Percentile: {stats['q95_length']:.2f} tokens")
    print(f"99th Percentile: {stats['q99_length']:.2f} tokens")

    return 0


if __name__ == "__main__":
    exit(main())
