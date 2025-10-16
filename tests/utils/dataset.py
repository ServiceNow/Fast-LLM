import pathlib
import random

import numpy as np
import torch
import yaml

from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset
from tests.utils.global_variables import (
    DATASET_PREFIX,
    MODEL_DATASET_PREFIX,
    MODEL_TEST_VOCAB_SIZE,
    TEST_CHARACTERS,
    TEST_DATASET_TOKENS,
    TEST_VOCAB_SIZE,
    TOKENIZER_FILE,
    TOKENIZER_PATH,
)


def download_santacoder_tokenizer():
    if not TOKENIZER_FILE.is_file():
        import transformers

        transformers.AutoTokenizer.from_pretrained("bigcode/santacoder").save_pretrained(TOKENIZER_PATH)


def get_test_dataset(
    prefix: pathlib.Path = DATASET_PREFIX,
    seed: int = 1234,
    num_tokens: int = TEST_DATASET_TOKENS,
    characters: str = TEST_CHARACTERS,
    vocab_size: int = TEST_VOCAB_SIZE,
    max_spans: int = 0,
):
    download_santacoder_tokenizer()

    if not (
        prefix.with_suffix(".idx").is_file()
        and prefix.with_suffix(".bin").is_file()
        and prefix.parent.joinpath("fast_llm_config.yaml").is_file()
    ):
        import transformers

        texts = "".join(random.Random(seed).choices(characters, k=num_tokens)).splitlines()
        tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)

        samples = [
            (
                torch.from_numpy(np.array(tokenizer(document)["input_ids"], dtype=np.uint16) % vocab_size),
                None,
                None,
                None,
            )
            for document in texts
        ]
        if max_spans > 0:
            spans = np.sort(
                np.random.RandomState(seed + 3847).randint(
                    0, np.array([[max(len(tokens), 1)] for tokens, _, _, _ in samples]), [len(samples), max_spans]
                )
            )
            samples = (
                (tokens, np.unique(spans_).tolist()) for (tokens, _, _, _), spans_ in zip(samples, spans, strict=True)
            )
            samples = [
                (
                    tokens,
                    torch.tensor(
                        [[begin, end] for begin, end in zip(spans_[::2], spans_[1::2], strict=False)],
                        dtype=torch.int32,
                    ).reshape(-1, 2),
                    None,
                    None,
                )
                for tokens, spans_ in samples
            ]

        GPTMemmapDataset.write_dataset(prefix, samples)
        yaml.safe_dump(
            {"type": "memmap", "path": prefix.name}, prefix.parent.joinpath("fast_llm_config.yaml").open("w")
        )


def get_model_test_dataset(
    prefix: pathlib.Path = MODEL_DATASET_PREFIX,
    vocab_size: int = MODEL_TEST_VOCAB_SIZE,
):
    return get_test_dataset(prefix=prefix, vocab_size=vocab_size)
