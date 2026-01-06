# Dataset Discovery

A tool to automatically discover `.fast_llm_dataset` files in a directory tree and generate blended configuration files with weights proportional to token counts.

## Overview

The dataset discovery preparator walks through a directory tree, identifies datasets by their `.fast_llm_dataset` files, and generates a configuration file that blends all discovered datasets with weights proportional to token counts.

### How It Works

1. Recursively scans a directory for `.fast_llm_dataset` files
2. Reads token counts directly from the binary file headers (no YAML parsing needed)
3. Generates a hierarchical blended config with weights proportional to token counts
4. Groups datasets by directory structure

This is useful when you have a collection of prepared datasets and want to automatically create a blended training config without manually specifying each dataset and its weight.

## Features

- **Automatic Discovery**: Recursively finds all `.fast_llm_dataset` files in nested directories
- **Direct Token Reading**: Reads token counts from binary file headers (fast and reliable)
- **Hierarchical Blending**: Preserves directory structure in generated config
- **Token-Proportional Weights**: Automatically calculates weights based on token counts for proportional sampling
- **Ignore Paths**: Exclude specific directories from discovery
- **Robust Error Handling**: Gracefully handles missing/invalid files

## Usage

### Command Line (using tools wrapper)

```bash
python tools/discover_datasets.py <directory> -o <output.yaml> [options]
```

**Arguments:**

- `directory`: Directory to search for datasets recursively (required)
- `-o, --output`: Output path for the generated config YAML file (required)
- `--ignore`: Path to ignore during dataset discovery (can be specified multiple times, optional)

**Examples:**

```bash
# Basic usage - discover all datasets and create blended config
python tools/discover_datasets.py /path/to/datasets -o blended_dataset.yaml

# Ignore specific paths during discovery
python tools/discover_datasets.py /path/to/datasets -o blended_dataset.yaml --ignore experiments/old --ignore tmp
```

### Config File

Create a YAML config file:

```yaml
# discover_config.yaml
type: prepare_dataset_discovery
directory: /path/to/datasets
output: blended_dataset.yaml
ignore_paths:  # Optional
  - ./test_data
  - ./checkpoints
```

Run with Fast-LLM's config system:

```bash
python -m fast_llm.engine.config_utils.run --config discover_config.yaml
```

Or directly via command line:

```bash
python -m fast_llm.engine.config_utils.run prepare_dataset_discovery \
    --directory /path/to/datasets \
    --output discovered_datasets.yaml
```

Or using the tools wrapper with config:

```bash
python tools/discover_datasets.py --config discover_config.yaml
```

## Configuration Options

- **directory** (required): Root directory to scan for `.fast_llm_dataset` files
- **output** (required): Where to save the generated blended config YAML
- **ignore_paths** (optional): List of paths to exclude from discovery
  - Can be absolute paths or relative to the directory
  - Any datasets under these paths will be skipped

## Dataset Identification

The tool identifies datasets by looking for files with the `.fast_llm_dataset` extension. These files are binary memmap files created by Fast-LLM's dataset preparation commands.

Unlike the old implementation, this tool:
- ❌ Does NOT look for `fast_llm_config*.yaml` files
- ✅ Directly scans for `.fast_llm_dataset` binary files
- ✅ Reads metadata from binary file headers (faster, more reliable)

## Output Format

### Hierarchical Blended Config

The tool generates a hierarchical blended config that mirrors your directory structure:

```yaml
type: blended
name: root_directory
datasets:
  - type: blended
    name: domain_a_local  # Local datasets grouped
    datasets:
      - type: memmap
        path: /path/to/datasets/domain_a/shard_0.fast_llm_dataset
      - type: memmap
        path: /path/to/datasets/domain_a/shard_1.fast_llm_dataset
    weights: [1.0, 1.0]  # Token counts in billions
  - type: memmap
    path: /path/to/datasets/domain_b/shard_0.fast_llm_dataset  # Single dataset (not wrapped)
weights: [2.0, 3.0]  # Total tokens per group in billions
```

### Directory Structure Example

Given this file structure:
```
datasets/
├── domain_a/
│   ├── shard_0.fast_llm_dataset  (1B tokens)
│   └── shard_1.fast_llm_dataset  (1B tokens)
├── domain_b/
│   ├── shard_0.fast_llm_dataset  (2B tokens)
│   └── shard_1.fast_llm_dataset  (2B tokens)
└── domain_c/
    └── shard_0.fast_llm_dataset   (4B tokens)
```

The generated config will blend:
- `domain_a`: 2B tokens total (20% of samples)
- `domain_b`: 4B tokens total (40% of samples)
- `domain_c`: 4B tokens total (40% of samples)

### Blended Datasets Explained

With blended datasets, samples are drawn from each dataset proportionally to their weights during training:

- **Proportional sampling**: Larger datasets (more tokens) are sampled more frequently
- **Interleaved samples**: Unlike concatenation, samples from different datasets are mixed
- **Automatic weights**: Calculated from token counts - no manual specification needed
- **Hierarchical weighting**: Subdirectories are weighted by their total token count

### Using in Training Config

The generated config can be used directly in a training config:

```yaml
data:
  datasets:
    training:
      type: file
      path: blended_dataset.yaml
```

## Example Workflow

### 1. Prepare Multiple Datasets

```bash
# Prepare dataset 1
fast-llm prepare --config dataset1_prepare.yaml

# Prepare dataset 2
fast-llm prepare --config dataset2_prepare.yaml

# Prepare dataset 3
fast-llm prepare --config dataset3_prepare.yaml
```

This creates a directory structure like:

```
my_datasets/
├── dataset1/
│   ├── shard_0_0.fast_llm_dataset
│   └── fast_llm_config.yaml
├── dataset2/
│   ├── shard_0_0.fast_llm_dataset
│   └── fast_llm_config.yaml
└── dataset3/
    ├── shard_0_0.fast_llm_dataset
    └── fast_llm_config.yaml
```

### 2. Discover and Blend Datasets

```bash
python tools/discover_datasets.py my_datasets/ -o blended_datasets.yaml
```

This generates `blended_datasets.yaml`:

```yaml
# This file was generated with fast_llm.data.preparator.dataset_discovery; weights are token-counts in billions.
# Configuration:
#   directory: my_datasets/

type: blended
name: my_datasets
datasets:
  - type: memmap
    path: my_datasets/dataset1/shard_0_0.fast_llm_dataset
  - type: memmap
    path: my_datasets/dataset2/shard_0_0.fast_llm_dataset
  - type: memmap
    path: my_datasets/dataset3/shard_0_0.fast_llm_dataset
weights:
  - 1.5   # Dataset 1: 1.5B tokens
  - 2.0   # Dataset 2: 2B tokens
  - 3.0   # Dataset 3: 3B tokens
```

### 3. Use in Training Config

```yaml
# training_config.yaml
model:
  # ... model config ...

data:
  datasets:
    training:
      type: file
      path: blended_datasets.yaml
  sampling:
    shuffle: skip_first_epoch
    seed: 784569

# ... rest of training config ...
```

### 4. Train

```bash
fast-llm train --config training_config.yaml
```

## Use Cases

### 1. Combining Multiple Data Sources

You have data from different sources (web scrapes, books, code, etc.) prepared separately:

```bash
python tools/discover_datasets.py /data/pretraining -o all_pretraining_data.yaml
```

### 2. Incremental Data Addition

You keep adding new datasets over time and want to automatically include all of them:

```bash
# Just add new prepared datasets to the directory
# Re-run discovery to update the combined config
python tools/discover_datasets.py /data/pretraining -o all_pretraining_data.yaml
```

### 3. Experiment Organization

You have experiments with different preprocessing or filtering:

```
experiments/
├── baseline/
│   └── dataset.fast_llm_dataset
├── filtered_v1/
│   └── dataset.fast_llm_dataset
└── filtered_v2/
    └── dataset.fast_llm_dataset
```

```bash
python tools/discover_datasets.py experiments/ -o all_experiments.yaml
```

## Error Handling

The tool gracefully handles errors:

- **Missing files**: Warns and skips (returns 0 tokens)
- **Invalid format**: Warns and skips (returns 0 tokens)
- **Read errors**: Warns and skips (returns 0 tokens)
- **Files with 0 tokens**: Excluded from final config with warning

All warnings are logged so you can see which files were skipped.

## Implementation Details

### File Format

`.fast_llm_dataset` files are binary memmap files with this structure:
```
[Header: "fast_llm_prepared_dataset"]
[Pointer to config: 8 bytes]
[Data: variable length]
[Config length: 4 bytes]
[Config JSON: variable length]
```

The config JSON contains metadata including `num_tokens`, which is used for weighting.

### Key Methods

- `_find_dataset_files()` - Recursively find `.fast_llm_dataset` files
- `_read_memmap_num_tokens()` - Read token count from binary file header
- `_create_memmap_config_for_dataset()` - Generate memmap config dict
- `_create_hierarchical_config()` - Build nested blended config tree
- `_create_directory_config()` - Recursively process directory structure
- `_group_files_by_directory()` - Group files by parent directory
- `_build_directory_tree()` - Build parent-child directory relationships

### Required Imports

The tool imports several config registries to ensure proper deserialization:
- `fast_llm.data.preprocessing.image_patch`
- `fast_llm.data.preprocessing.language_model`
- `fast_llm.data.sample.language_model`
- `fast_llm.data.sample.patch`
- `fast_llm.data.sample.range`
- `fast_llm.data.sample.token`

These are loaded when reading each file to ensure the config types are registered.

## Testing

Run tests:

```bash
pytest tests/data/test_dataset_discovery.py
```

Tests cover:
- Unit tests for helper methods (file discovery, grouping, tree building)
- End-to-end tests with actual prepared datasets
- Hierarchical directory structures
- Ignore paths functionality
- Error handling (empty directories, invalid files)

## Notes

- **Absolute Paths**: The tool uses absolute paths for memmap files to ensure configs work regardless of where they're used from

- **Ordering**: Datasets are discovered and ordered alphabetically by path for consistency

- **Single Dataset**: If only one `.fast_llm_dataset` file is found, it's returned directly (not wrapped in a blended config)

- **Empty Directories**: If no `.fast_llm_dataset` files are found, the tool will raise an error

- **All Files Included**: The tool discovers ALL `.fast_llm_dataset` files in the directory tree. Use `--ignore` to exclude specific paths if needed

## Benefits

- **Automatic weighting**: No manual calculation of token counts needed
- **Hierarchical organization**: Preserves directory structure in config
- **Self-contained**: Reads all metadata from dataset files themselves
- **Simple**: Direct file scanning, no complex YAML parsing
- **Robust**: Binary format is well-defined, graceful error handling
- **Fast**: Direct header reads, no full dataset loading

## Limitations

- Only works with `.fast_llm_dataset` files (not arbitrary configs)
- Always generates memmap configs (doesn't support complex config composition)
- No slicing or sampling during discovery (use separate tools for that)

For most workflows (processing datasets then training on them), these limitations are not relevant.

## See Also

- [Fast-LLM Data Configuration Documentation](../../../docs/recipes/data-configuration.md)
- [Dataset Preparation Guide](../../../docs/recipes/data-preparation.md)
- [GPT Memmap Preparator](../gpt_memmap/)
