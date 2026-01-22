# Dataset Discovery

Automatically discover `.fast_llm_dataset` files and generate a blended config with token-proportional weights.

## Quick Start

Using the tools wrapper:
```bash
python tools/discover_datasets.py <directory> -o <output.yaml>
```

Using Fast-LLM CLI with config file:
```yaml
type: prepare_dataset_discovery
directory: /path/to/datasets
output: blended_dataset.yaml
ignore_paths: [test_data, checkpoints]  # Optional
```

```bash
python -m fast_llm.cli --config config.yaml
```

## What It Does

1. Scans directory tree for `.fast_llm_dataset` files
2. Reads token counts from binary file headers
3. Generates hierarchical blended config with automatic weights
4. Preserves directory structure

## Example

Input directory structure:
```
datasets/
├── domain_a/
│   ├── shard_0.fast_llm_dataset  (1B tokens)
│   └── shard_1.fast_llm_dataset  (1B tokens)
└── domain_b/
    └── shard_0.fast_llm_dataset   (4B tokens)
```

Generated config (`blended.yaml`):
```yaml
type: blended
name: datasets
datasets:
  - type: blended
    name: domain_a
    datasets:
      - type: memmap
        path: datasets/domain_a/shard_0.fast_llm_dataset
      - type: memmap
        path: datasets/domain_a/shard_1.fast_llm_dataset
    weights: [1.0, 1.0]
  - type: memmap
    path: datasets/domain_b/shard_0.fast_llm_dataset
weights: [2.0, 4.0]  # In billions
```

Use in training:
```yaml
data:
  datasets:
    training:
      type: file
      path: blended.yaml
```

## Options

- **directory**: Root directory to scan (required)
- **output**: Output YAML file path (required)
- **ignore_paths**: Paths to exclude, relative or absolute (optional)

## Key Features

- **Token-proportional sampling**: Datasets sampled by token count (larger datasets sampled more)
- **Hierarchical grouping**: Directory structure preserved in config
- **Automatic weights**: Calculated from binary file metadata
- **Error handling**: Skips unreadable files with warnings

## Notes

- Single datasets returned directly (not wrapped)
- Files with 0 tokens skipped with warning
- Empty directories raise error
- Datasets sorted alphabetically

## Testing

```bash
pytest tests/data/test_dataset_discovery.py
```
