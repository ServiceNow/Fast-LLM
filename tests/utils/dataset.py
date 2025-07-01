import pathlib
import random
import string

import numpy as np
import yaml

from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset
from fast_llm.data.dataset.gpt.sampled import GPTSample
from tests.utils.utils import TEST_RESULTS_PATH

# TODO: Fixtures
TOKENIZER_PATH = TEST_RESULTS_PATH / "tokenizer" / "common"
TOKENIZER_FILE = TOKENIZER_PATH / "tokenizer.json"
DATASET_CACHE = TEST_RESULTS_PATH / "dataset"
DATASET_PREFIX = DATASET_CACHE / "common" / "dataset"
DATASET_SAMPLING_CACHE = TEST_RESULTS_PATH / "dataset" / "cache"
TEST_VOCAB_SIZE = 8192
# Random lowercase: 80.7% (3.1% each); space: 18.6%; doc end: 0.6%
TEST_CHARACTERS = (string.ascii_lowercase) * 5 + " " * 30 + "\n"
TEST_DATASET_TOKENS = 1000000


def get_test_dataset(
    prefix: pathlib.Path = DATASET_PREFIX,
    seed: int = 1234,
    num_tokens: int = TEST_DATASET_TOKENS,
    characters: str = TEST_CHARACTERS,
    vocab_size: int = TEST_VOCAB_SIZE,
    max_spans: int = 0,
):
    if not TOKENIZER_FILE.is_file():
        import transformers

        transformers.AutoTokenizer.from_pretrained("bigcode/santacoder").save_pretrained(TOKENIZER_PATH)

    if not (
        prefix.with_suffix(".idx").is_file()
        and prefix.with_suffix(".bin").is_file()
        and prefix.parent.joinpath("fast_llm_config.yaml").is_file()
    ):
        import transformers

        texts = "".join(random.Random(seed).choices(characters, k=num_tokens)).splitlines()
        tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)

        samples = [
            GPTSample(np.array(tokenizer(document)["input_ids"], dtype=np.uint16) % vocab_size) for document in texts
        ]
        if max_spans > 0:
            lengths = np.array([max(len(sample.token_ids), 1) for sample in samples])
            spans = np.sort(np.random.RandomState(seed + 3847).randint(0, lengths[:, None], [len(samples), max_spans]))
            for sample, span in zip(samples, spans):
                span = np.unique(span)
                sample.loss_masking_spans = span[: len(span) // 2 * 2].reshape(-1, 2)

        GPTMemmapDataset.write_dataset(prefix, samples)
        yaml.safe_dump(
            {"type": "memmap", "path": prefix.name}, prefix.parent.joinpath("fast_llm_config.yaml").open("w")
        )


def get_test_concatenated_memmap_dataset(
    path: pathlib.Path,
    num_files: int,
    seed: int = 1234,
    num_tokens: int = TEST_DATASET_TOKENS,
    characters: str = TEST_CHARACTERS,
    vocab_size: int = TEST_VOCAB_SIZE,
    seed_shift: int = 55,
):
    index_file = path / "index.txt"
    if not index_file.is_file():
        for i in range(num_files):
            get_test_dataset(
                prefix=path / f"dataset_{i}",
                seed=seed + i * seed_shift,
                num_tokens=num_tokens,
                characters=characters,
                vocab_size=vocab_size,
            )
        index_file.open("w").writelines([str(path / f"dataset_{i}") + "\n" for i in range(num_files)])
