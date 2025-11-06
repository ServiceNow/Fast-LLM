import pathlib
import typing

import datasets
import numpy as np

from fast_llm.data.preparator.gpt_memmap.config import GPTMemmapDatasetPreparatorConfig
from fast_llm.utils import padded_cumsum
from tests.utils.global_variables import DATASET_CACHE, MODEL_TEST_VOCAB_SIZE, TOKENIZER_FILE, TOKENIZER_PATH


def download_santacoder_tokenizer():
    if not TOKENIZER_FILE.is_file():
        import transformers

        transformers.AutoTokenizer.from_pretrained("bigcode/santacoder").save_pretrained(TOKENIZER_PATH)


def get_random_spans(
    num_documents: int,
    max_spans: int,
    lengths: np.ndarray | int,
    random_state: np.random.RandomState = np.random,
    use_last_format: bool = False,
    variable_length: bool = True,
):
    if variable_length:
        spans = random_state.randint(
            0, lengths[:, None] if isinstance(lengths, np.ndarray) else lengths, [num_documents, max_spans * 2]
        )
    else:
        spans = [
            random_state.choice(range(length), max_spans * 2, replace=False)
            for length in (lengths if isinstance(lengths, np.ndarray) else (lengths for _ in range(num_documents)))
        ]
    spans = [np.unique(sample_spans).tolist() for sample_spans in np.sort(spans)]
    return [
        [(begin, end - use_last_format) for begin, end in zip(sample_spans[::2], sample_spans[1::2], strict=False)]
        for sample_spans in spans
    ]


def get_random_preference_spans(texts, random_state: np.random.RandomState = np.random) -> dict[str, str]:
    texts_ = []
    chosen_spans = []
    rejected_spans = []
    for text in texts:
        # Split in three non-empty_chunks
        splits = np.sort(random_state.choice(range(1, len(text) - 1), 2, replace=False)).tolist()
        texts_.append(text[: splits[0]])
        chosen_spans.append(text[splits[0] : splits[1]])
        rejected_spans.append(text[splits[1] :])
    return {"text": texts_, "chosen_span": chosen_spans, "rejected_span": rejected_spans}


def _get_hf_test_dataset(
    seed: int = 1234,
    num_documents: int = 1000,
    min_document_size: int = 5,
    max_document_size: int = 100,
    max_loss_masking_spans: int = 0,
    has_preference_spans: bool = False,
):
    random_state = np.random.RandomState(seed)
    # Generate random document sizes (character count).
    document_sizes = random_state.randint(min_document_size, max_document_size, num_documents)
    size_cumsums = padded_cumsum(document_sizes)
    # Generate random ascii characters.
    random_text = random_state.randint(32, 127, document_sizes.sum(), dtype=np.uint8).tobytes().decode()
    texts = [random_text[begin:end] for begin, end in zip(size_cumsums[:-1], size_cumsums[1:])]

    if has_preference_spans:
        dataset_dict = get_random_preference_spans(texts, random_state)
    else:
        dataset_dict: dict[str, typing.Any] = {"text": texts}

    if max_loss_masking_spans > 0:
        dataset_dict["loss_masking_spans"] = get_random_spans(
            num_documents, max_loss_masking_spans, document_sizes, random_state, use_last_format=True
        )

    return datasets.Dataset.from_dict(dataset_dict)


def _get_test_dataset(
    path: pathlib.Path,
    seed: int,
    tokenizer_path: str = TOKENIZER_PATH,
    vocab_size: int | None = None,
    documents_per_shard: int = 10**6,
    num_documents: int = 1000,
    min_document_size: int = 5,
    max_document_size: int = 100,
    max_loss_masking_spans: int = 0,
    has_preference_spans: bool = False,
    splits: dict[str, float] | None = None,
):
    config_paths = (
        [path / "fast_llm_config.yaml"]
        if splits is None
        else [path / f"fast_llm_config_{split}.yaml" for split in splits]
    )
    hf_path = path / "hf"

    if not all(config_path.is_file() for config_path in config_paths):
        dataset = _get_hf_test_dataset(
            seed, num_documents, min_document_size, max_document_size, max_loss_masking_spans, has_preference_spans
        )
        datasets.DatasetDict({"train": dataset}).save_to_disk(hf_path)
        source_schema = {"text": "text"}
        if max_loss_masking_spans > 0:
            source_schema["loss_masking_spans"] = "loss_masking_spans"
        if has_preference_spans:
            source_schema["chosen_span"] = "chosen_span"
            source_schema["rejected_span"] = "rejected_span"

        download_santacoder_tokenizer()
        preparator_config = GPTMemmapDatasetPreparatorConfig.from_dict(
            {
                "dataset": {
                    "path": hf_path,
                    "load_from_disk": True,
                    "source_schema": source_schema,
                },
                "tokenizer": {"path": tokenizer_path, "max_vocab_size": vocab_size},
                "output_path": path,
                "documents_per_shard": documents_per_shard,
                "splits": splits,
            }
        )
        preparator_config.run()

    config = (
        {"type": "file", "path": config_paths[0]}
        if splits is None
        else {
            split: {"type": "file", "path": config_path}
            for split, config_path in zip(splits, config_paths, strict=True)
        }
    )
    return path, config, hf_path


def get_common_test_dataset():
    return _get_test_dataset(DATASET_CACHE / "common_dataset", seed=1234)


def get_alt_test_dataset():
    return _get_test_dataset(DATASET_CACHE / "other_dataset", seed=2345)


def get_sharded_test_dataset():
    return _get_test_dataset(DATASET_CACHE / "common_dataset_sharded", seed=1234, documents_per_shard=350)


def get_split_test_dataset():
    return _get_test_dataset(
        DATASET_CACHE / "common_dataset_split", seed=1234, splits={"training": 1, "validation": 1}
    )


def get_split_sharded_test_dataset():
    return _get_test_dataset(
        DATASET_CACHE / "common_dataset_split_sharded",
        seed=1234,
        documents_per_shard=350,
        splits={"training": 1, "validation": 1},
    )


def get_test_dataset_with_loss_masking_spans():
    return _get_test_dataset(DATASET_CACHE / "dataset_with_loss_masking_spans", seed=1234, max_loss_masking_spans=5)


def get_test_dataset_with_preference_spans():
    return _get_test_dataset(DATASET_CACHE / "dataset_with_preference_spans", seed=1234, has_preference_spans=True)


def get_model_test_dataset():
    return _get_test_dataset(DATASET_CACHE / "model_dataset", seed=1234, vocab_size=MODEL_TEST_VOCAB_SIZE)
