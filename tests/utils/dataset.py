import pathlib
import random

import numpy as np
import torch
import yaml

from fast_llm.data.dataset.memmap import MemmapDataset
from fast_llm.data.sample.language_model import LanguageModelSample, LanguageModelWriter
from fast_llm.data.sample.range import RangeSample
from fast_llm.data.sample.token import TokenSample
from tests.utils.global_variables import (
    DATASET_PATH,
    MODEL_DATASET_PATH,
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


def get_random_spans(num_samples: int, max_spans: int, lengths: np.ndarray | int, seed: int = 0):
    spans = np.sort(np.random.RandomState(seed + 3847).randint(0, lengths, [num_samples, max_spans * 2]))
    spans = [np.unique(sample_spans).tolist() for sample_spans in spans]
    return [
        [(begin, end) for begin, end in zip(sample_spans[::2], sample_spans[1::2], strict=False)]
        for sample_spans in spans
    ]


def get_test_dataset(
    path: pathlib.Path = DATASET_PATH,
    seed: int = 1234,
    num_tokens: int = TEST_DATASET_TOKENS,
    characters: str = TEST_CHARACTERS,
    vocab_size: int = TEST_VOCAB_SIZE,
    max_spans: int = 0,
):
    download_santacoder_tokenizer()
    config_path = path.parent.joinpath("fast_llm_config.yaml")

    if not (path.is_file() and config_path.is_file()):
        import transformers

        texts = "".join(random.Random(seed).choices(characters, k=num_tokens)).splitlines()
        tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)

        samples = [
            LanguageModelSample(
                TokenSample(
                    torch.from_numpy(np.array(tokenizer(document)["input_ids"], dtype=np.uint16) % vocab_size)
                ),
            )
            for document in texts
        ]
        if max_spans > 0:
            spans = get_random_spans(
                len(samples), max_spans, np.array([[max(len(sample), 1)] for sample in samples]), seed
            )
            for sample, sample_spans in zip(samples, spans, strict=True):
                sample.loss_masking_spans = RangeSample(sample_spans, len(sample))

        MemmapDataset.write_dataset(path, samples, LanguageModelWriter)
        yaml.safe_dump({"type": "memmap", "path": path.name}, config_path.open("w"))


def get_model_test_dataset(
    path: pathlib.Path = MODEL_DATASET_PATH,
    vocab_size: int = MODEL_TEST_VOCAB_SIZE,
):
    return get_test_dataset(path, vocab_size=vocab_size)
