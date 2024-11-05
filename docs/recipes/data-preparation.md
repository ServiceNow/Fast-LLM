# Training Data Preparation

<!--- TODO: Provide an actual example dataset --->

## Prepare datasets

<!--- TODO: Tokenizer? --->

The data processing of Fast-LLM is designed to closely match that of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
In particular, it requires datasets to be converted to the Megatron-LM binary format.
Please refer to [this guide](https://github.com/NVIDIA/Megatron-LM?tab=readme-ov-file#data-preprocessing)
for details on how to prepare the dataset(s).

At the end of this process, each dataset should have a consist of a binary file `$DATA_PREFIX_[i].bin` and an index file `$DATA_PREFIX_[i].idx`

## List configuration

Datasets may be configured via a simple string in the `--data_path` argument.
(Again, in the exact same format as with Megatron-LM).
For a single dataset, we only need to specify its prefix:
```bash
export DATA_ARGS_SINGLE="\
--split=9998,2,0 \
--dataset_source=list \
--data_path=$DATA_PREFIX_0 \
"
```
Note that we also specify a train/validation/test split for the dataset.
Fow multiple datasets, we specify the prefixes together with relative dataset sampling probabilities.
For examples
```bash
export DATA_ARGS_MULTIPLE="\
--split=9998,2,0 \
--dataset_source=list \
--data_path=\"0.3 $DATA_PREFIX_0 0.5 $DATA_PREFIX_1 0.2 $DATA_PREFIX_2\" \
"
```

!!! warning

    The same dataset split is used for every dataset.
    This may cause problems for extremely small datasets, which we recommend avoiding.
    (If needed, we suggest concatenating small datasets into larger ones.)

!!! warning

    Make sure to dedicate enough data for validation and/or testing, and adjust the split according to you dataset.
    Our setup assumes a dataset of 500 billion tokens, and requires 26 million tokens for each validation,
    so allocating 0.02% of the total data (100 million tokens)
    ensures sufficient data without excessively reducing the training set size.


## Json configuration

While the list configuration is sufficient for a small number of datasets,
it becomes impractical when there are many of them.
For that purpose, Fast-LLM allows configuring a dataset from an external json file.

A common use case concerns large datasets with hundreds of billions of tokens,
which need to be split into multiple ones to keep the file size reasonable.
We want to sample each dataset as if it was not split, i.e. with probability proportional to its document count.
In that case, the json configuration file can be generated automatically using the `concatenate_dataset.py` script:
```bash
python3 tools/concatenate_dataset.py --directory=$DATASET_DIR --output_name=$JSON_DATA_PATH
"
```
This script will recursively scan `$DATASET_DIR` for datasets (`.idx` files),
and create a json dataset configuration at `$JSON_DATA_PATH` with the appropriate dataset prefixes and probabilities.
The resulting json file can be used to configure the datasets:
```bash
export DATA_ARGS="\
--split=9998,2,0 \
--dataset_source=file \
--data_path=$JSON_DATA_PATH \
"
```

??? question "More on the json dataset file"

    The json dataset file is a simple structure for holding the data prefixes and probabilities,
    to avoid writing them explicitly in the Fast-LLM configuration.
    It may be created manually or through a script such as `concatenate_dataset.py`
    It may also contain metadata about the dataset contents, for example the total number of tokens and documents.
    The file should be structured as:
    ```json
    {
        "datasets": [
            {
                "prefix": $RELATIVE_DATA_PREFIX_0"
                "weight": 0.3
                "num_documents": 12345,
                "num_tokens": 987654321,
                ...
            },
            ...
        ]
    }
    ```
    Note that in the json format, paths are relative to the directory containing the json file
    instead of the current working directory.
