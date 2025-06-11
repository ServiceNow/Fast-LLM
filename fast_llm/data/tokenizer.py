import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from fast_llm.data.config import TokenizerConfig
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

    def _tokenize(self, text: str, begin=True, end=True) -> list[int]:
        return (
            ([self.bod_id] if begin else [])
            + self.tokenizer.encode(text, add_special_tokens=False)
            + ([self.eod_id] if end else [])
        )

    def tokenize(self, text: str, char_spans=None, image_positions=None) -> tuple[list[int], list[tuple[int, int]]]:
        """
        Tokenize the input text and return the tokenized input_ids and if provided, token spans and image positions.
        """
        if not image_positions:
            image_positions = []
        if not char_spans:
            char_spans = []

        image_idx = 0
        char_pos = 0
        token_ids = []
        image_token_positions = []
        token_spans = []
        beginning_of_text = True
        image_position = image_positions[image_idx] if image_idx < len(image_positions) else float("inf")

        for start, end in char_spans:
            # Tokenize all text before the span, with image positions in mind (i.e., break text at image positions).
            while image_position <= start:
                tokenized_text = self._tokenize(text[char_pos:image_position], begin=beginning_of_text, end=False)
                beginning_of_text = False
                token_ids.extend(tokenized_text)
                image_token_positions.append(len(token_ids))
                image_idx += 1
                char_pos = image_position
                image_position = image_positions[image_idx] if image_idx < len(image_positions) else float("inf")
            if char_pos < start:
                tokenized_text = self._tokenize(text[char_pos:start], begin=beginning_of_text, end=False)
                beginning_of_text = False
                token_ids.extend(tokenized_text)
            char_pos = start
            len(token_ids)
            span_length = 0
            token_start = len(token_ids)
            # Tokenize all text before the end of the span
            while image_position <= end:
                tokenized_text = self._tokenize(text[char_pos:image_position], begin=beginning_of_text, end=False)
                beginning_of_text = False
                token_ids.extend(tokenized_text)
                image_token_positions.append(len(token_ids))
                span_length += len(tokenized_text)
                char_pos = image_position
                image_idx += 1
                image_position = image_positions[image_idx] if image_idx < len(image_positions) else float("inf")
            # Tokenize the last part of the span, since there are no more images
            if char_pos < end + 1:
                # end of span is end of text
                tokenized_text = self._tokenize(
                    text[char_pos : end + 1],
                    begin=beginning_of_text,
                    end=(end >= len(text) - 1),
                )
                beginning_of_text = False
                token_ids.extend(tokenized_text)
                span_length += len(tokenized_text)
                char_pos = end + 1
            token_spans.append((token_start, token_start + span_length - 1))

        # Tokenize text remaining after the last span
        while image_position <= len(text):
            image_position = image_positions[image_idx]
            tokenized_text = self._tokenize(text[char_pos:image_position], begin=beginning_of_text, end=False)
            beginning_of_text = False
            token_ids.extend(tokenized_text)
            image_token_positions.append(len(token_ids))
            char_pos = image_position
            image_idx += 1
            image_position = image_positions[image_idx] if image_idx < len(image_positions) else float("inf")
        tokenized_text = self._tokenize(text[char_pos:], begin=beginning_of_text, end=True)
        token_ids.extend(tokenized_text)

        return token_ids, token_spans, image_token_positions

    def detokenize(self, token_ids: int | list[int] | np.ndarray | torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id
