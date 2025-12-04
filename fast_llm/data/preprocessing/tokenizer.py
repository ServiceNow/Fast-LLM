import functools
import pathlib
import typing

from fast_llm.config import Configurable, Field, FieldHint, config_class
from fast_llm.data.preprocessing.abstract import PreprocessingConfig
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    import numpy as np
    import torch


@config_class(dynamic_type={PreprocessingConfig: "tokenizer"})
class TokenizerConfig(PreprocessingConfig):
    """
    Configuration for the tokenizer.
    The tokenizer is needed for FIM and dataset preparation.
    """

    _abstract = False

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
    max_vocab_size: int | None = Field(
        default=None,
        desc="Constrain output tokens to a specific range. Used for testing.",
        hint=FieldHint.testing,
    )

    def get_tokenizer(self) -> "Tokenizer":
        from fast_llm.data.preprocessing.tokenizer import Tokenizer

        return Tokenizer(self)


class Tokenizer[ConfigType: TokenizerConfig](Configurable[ConfigType]):
    """
    A wrapper around Huggingface (transformers) tokenizer.
    """

    def __init__(self, config: ConfigType):
        super().__init__(config)
        from transformers import AutoTokenizer

        log_main_rank(f"> loading tokenizer from {config.path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._config.path,
            errors="replace",
            max_len=None,
            trust_remote_code=True,
            use_fast=True,
        )
        if self._config.bos_token is not None:
            self.tokenizer.bos_token = self._config.bos_token
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer does not have an EOS token.")
        if self.tokenizer.bos_token_id is None:
            raise ValueError("Tokenizer does not have an BOS token.")
        self.eod_id = self.tokenizer.eos_token_id
        self.bod_id = self.tokenizer.bos_token_id

    @functools.cached_property
    def vocab_size(self) -> int:
        out = len(self.tokenizer)
        if self._config.max_vocab_size is not None:
            out = min(out, self._config.max_vocab_size)
        return out

    @property
    def vocab(self) -> dict[str, int]:
        return self.tokenizer.vocab

    @property
    def inv_vocab(self) -> dict[int, str]:
        return self._inv_vocab

    def tokenize(
        self, text: str, begin: bool = True, end: bool = True, data_type: DataType = DataType.int64
    ) -> "torch.Tensor":
        import torch

        tokens = torch.tensor(
            ([self.bod_id] if begin else [])
            + self.tokenizer.encode(text, add_special_tokens=False)
            + ([self.eod_id] if end else []),
            dtype=data_type.torch,
        )
        if self._config.max_vocab_size is not None:
            tokens %= self._config.max_vocab_size
        return tokens

    def tokenize_with_spans(
        self,
        text: str,
        begin: bool = True,
        end: bool = True,
        *,
        text_spans: list[tuple[int, int]],
        data_type: DataType = DataType.int64,
    ) -> tuple["torch.Tensor", list[tuple[int, int]]]:
        """
        Perform span-aware tokenization and return the tokenized input_ids along with token spans.
        """
        if not text_spans:
            return self.tokenize(text, begin, end, data_type=data_type), []
        input_ids, token_splits = self.tokenize_with_splits(
            text, begin, end, text_splits=[split for splits in text_spans for split in splits], data_type=data_type
        )
        return input_ids, [(begin, end) for begin, end in zip(token_splits[::2], token_splits[1::2], strict=True)]

    def tokenize_with_splits(
        self,
        text: str,
        begin: bool = True,
        end: bool = True,
        *,
        text_splits: list[int],
        data_type: DataType = DataType.int64,
    ) -> tuple["torch.Tensor", list[int]]:
        if not text_splits:
            return self.tokenize(text, begin, end, data_type=data_type), []
        import torch

        Assert.eq(sorted(text_splits), text_splits)
        input_ids = []
        text_splits = [0, *text_splits, len(text)]
        token_splits = []
        total_tokens = 0

        for i, (split_begin, split_end) in enumerate(zip(text_splits[:-1], text_splits[1:])):
            input_ids.append(
                split_tokens := self.tokenize(
                    text[split_begin:split_end],
                    begin and i == 0,
                    end and i == len(text_splits) - 2,
                    data_type=data_type,
                )
            )
            total_tokens += len(split_tokens)
            token_splits.append(total_tokens)

        return torch.cat(input_ids), token_splits[:-1]

    def detokenize(
        self, tokens: "int | list[int] | np.ndarray | torch.Tensor", begin: bool = False, end: bool = False
    ) -> str:
        tokens = self._remove_delimiters(tokens, begin, end)
        return self.tokenizer.decode(tokens)

    def detokenize_with_spans(
        self, tokens: "torch.Tensor", begin: bool = False, end: bool = False, *, token_spans: list[tuple[int, int]]
    ) -> tuple[str, list[tuple[int, int]]]:
        if not token_spans:
            return self.detokenize(tokens, begin, end), []
        text, text_splits = self.detokenize_with_splits(
            tokens, begin, end, token_splits=[split for splits in token_spans for split in splits]
        )
        return text, [(begin, end) for begin, end in zip(text_splits[::2], text_splits[1::2], strict=True)]

    def detokenize_with_splits(
        self, tokens: "torch.Tensor", begin: bool = False, end: bool = False, *, token_splits: list[int]
    ) -> tuple[str, list[int]]:
        if not token_splits:
            return self.detokenize(tokens, begin, end), []
        Assert.eq(sorted(token_splits), token_splits)
        tokens = self._remove_delimiters(tokens, begin, end)
        texts = []
        token_splits = [0, *(token_split - begin for token_split in token_splits), len(tokens)]
        text_splits = []
        total_characters = 0

        for i, (split_begin, split_end) in enumerate(zip(token_splits[:-1], token_splits[1:])):
            texts.append(split_text := self.detokenize(tokens[split_begin:split_end]))
            total_characters += len(split_text)
            text_splits.append(total_characters)

        return "".join(texts), text_splits[:-1]

    def _remove_delimiters(
        self, token_ids: "int | list[int] | np.ndarray | torch.Tensor", begin: bool = False, end: bool = False
    ):
        if begin:
            Assert.eq(token_ids[0], self.bod_id)
            token_ids = token_ids[1:]
        if end:
            Assert.eq(token_ids[-1], self.eod_id)
            token_ids = token_ids[:-1]
        return token_ids

    @property
    def eod(self):
        return self.eod_id
