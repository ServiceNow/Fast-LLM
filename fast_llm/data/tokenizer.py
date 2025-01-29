import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from fast_llm.data.config import SequenceDelimiters, TokenizerConfig
from fast_llm.engine.config_utils.run import log_main_rank


class Tokenizer:
    """
    A wrapper around Huggingface (transformers) tokenizer.
    """

    def __init__(self, config: TokenizerConfig):
        log_main_rank(f"> loading tokenizer from {config.path} ...")
        self.tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=config.path, errors="replace", max_len=None
        )
        self.special_tokens_mode = config.special_tokens_mode
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer does not have an EOS token.")
        self.eod_id = self.tokenizer.eos_token_id
        self.bod_id = self.tokenizer.bos_token_id
        self._inv_vocab = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def vocab(self) -> dict[str, int]:
        return self.tokenizer.vocab

    @property
    def inv_vocab(self) -> dict[int, str]:
        return self._inv_vocab

    def tokenize(self, text: str, beginning_of_text=True, end_of_text=True) -> list[int]:
        if self.special_tokens_mode == SequenceDelimiters.eos_only:
            return self.tokenizer.encode(text, add_special_tokens=False) + ([self.eod_id] if end_of_text else [])
        elif self.special_tokens_mode == SequenceDelimiters.bos_only:
            return ([self.bod_id] if (self.bod_id is not None and beginning_of_text) else []) + self.tokenizer.encode(
                text, add_special_tokens=False
            )
        elif self.special_tokens_mode == SequenceDelimiters.bos_eos:
            return (
                ([self.bod_id] if (self.bod_id is not None and beginning_of_text) else [])
                + self.tokenizer.encode(text, add_special_tokens=False)
                + ([self.eod_id] if end_of_text else [])
            )
        elif self.special_tokens_mode == SequenceDelimiters.no_delimiters:
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            # TODO: How do we handle when beginning_of_text=False or end_of_text=False?
            return self.tokenizer.encode(text)

    def tokenize_with_spans(
        self, text: str, char_spans: list[tuple[int, int]]
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """
        Perform span-aware tokenization and return the tokenized input_ids along with token spans.
        """
        input_ids = []
        token_spans = []
        char_pos = 0
        beginning_of_text = True
        for start, end in char_spans:
            if char_pos < start:
                curr_text = text[char_pos:start]
                tokenized_text = self.tokenize(curr_text, beginning_of_text=beginning_of_text, end_of_text=False)
                beginning_of_text = False
                input_ids.extend(tokenized_text)
            curr_text = text[start : end + 1]
            if end >= len(text) - 1:
                tokenized_text = self.tokenize(curr_text, beginning_of_text=beginning_of_text, end_of_text=True)
            else:
                tokenized_text = self.tokenize(curr_text, beginning_of_text=beginning_of_text, end_of_text=False)
            beginning_of_text = False
            token_spans.append((len(input_ids), len(input_ids) + len(tokenized_text) - 1))
            input_ids.extend(tokenized_text)
            char_pos = end + 1
        if char_pos < len(text):
            curr_text = text[char_pos:]
            tokenized_text = self.tokenize(curr_text, beginning_of_text=beginning_of_text, end_of_text=True)
            input_ids.extend(tokenized_text)
        return input_ids, token_spans

    def detokenize(self, token_ids: int | list[int] | np.ndarray | torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id
