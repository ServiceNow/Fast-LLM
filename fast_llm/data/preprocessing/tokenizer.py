import pathlib
import typing

from fast_llm.config import Config, Field, FieldHint, config_class
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    import numpy as np
    import torch


@config_class()
class TokenizerConfig(Config):
    """
    Configuration for the tokenizer.
    The tokenizer is needed for FIM and dataset preparation.
    """

    path: pathlib.Path = Field(
        default=None,
        desc="Path to the tokenizer file.",
        hint=FieldHint.core,
    )
    bos_token: str | None = Field(
        default=None,
        desc="BOS token to use if the tokenizer doesn't define one; must be an existing token.",
        hint=FieldHint.core,
    )

    def get_tokenizer(self) -> "Tokenizer":
        from fast_llm.data.preprocessing.tokenizer import Tokenizer

        return Tokenizer(self)


class Tokenizer:
    """
    A wrapper around Huggingface (transformers) tokenizer.
    """

    def __init__(self, config: TokenizerConfig):
        from transformers import AutoTokenizer

        log_main_rank(f"> loading tokenizer from {config.path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=config.path,
            errors="replace",
            max_len=None,
            trust_remote_code=True,
            use_fast=True,
        )
        if config.bos_token is not None:
            self.tokenizer.bos_token = config.bos_token
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer does not have an EOS token.")
        if self.tokenizer.bos_token_id is None:
            raise ValueError("Tokenizer does not have an BOS token.")
        self.eod_id = self.tokenizer.eos_token_id
        self.bod_id = self.tokenizer.bos_token_id

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def vocab(self) -> dict[str, int]:
        return self.tokenizer.vocab

    @property
    def inv_vocab(self) -> dict[int, str]:
        return self._inv_vocab

    def tokenize(self, text: str, begin: bool = True, end: bool = True) -> list[int]:
        return (
            ([self.bod_id] if begin else [])
            + self.tokenizer.encode(text, add_special_tokens=False)
            + ([self.eod_id] if end else [])
        )

    def tokenize_with_spans(
        self, text: str, begin: bool = True, end: bool = True, *, spans: list[tuple[int, int]]
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """
        Perform span-aware tokenization and return the tokenized input_ids along with token spans.
        """
        if not spans:
            return self.tokenize(text, begin, end), []
        input_ids, token_splits = self.tokenize_with_splits(
            text, begin, end, text_splits=[split for splits in spans for split in splits]
        )
        return input_ids, [(begin, end) for begin, end in zip(token_splits[::2], token_splits[1::2], strict=True)]

    def tokenize_with_splits(
        self, text: str, begin: bool = True, end: bool = True, *, text_splits: list[int]
    ) -> tuple[list[int], list[int]]:
        Assert.eq(sorted(text_splits), text_splits)
        input_ids = []
        text_splits = [0, *text_splits, len(text_splits)]
        token_splits = []

        for split_begin, split_end in zip(text_splits[:-1], text_splits[1:]):
            input_ids.extend(
                self.tokenize(
                    text[split_begin:split_end], begin=begin and split_begin == 0, end=end and split_end == len(text)
                )
            )
            token_splits.append(len(input_ids))

        return input_ids, token_splits[:-1]

    def detokenize(self, token_ids: "int | list[int] | np.ndarray | torch.Tensor") -> str:
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id
