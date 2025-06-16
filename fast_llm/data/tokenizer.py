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

    def tokenize(
        self, text: str, char_spans=None, image_positions=None
    ) -> tuple[list[int], list[tuple[int, int]], list[int]]:
        """
        Tokenize the input text and return the tokenized input_ids, token spans, and image token positions.
        This version simplifies logic by merging all relevant positions, sorting, and tokenizing between them.
        """
        if not image_positions:
            image_positions = []
        if not char_spans:
            char_spans = []

        # Collect all positions with their type
        positions = []
        for pos in image_positions:
            positions.append((pos, "image"))
        for start, end in char_spans:
            positions.append((start, "span_start"))
            positions.append((end + 1, "span_end"))
        # Sort positions by character index. We assume that image and span positions are individually sorted and spans do not overlap
        positions = sorted(positions, key=lambda x: x[0])

        token_ids = []
        token_spans = []
        image_token_positions = []
        char_pos = 0
        current_span_start = None

        for position in positions:
            # We only tokenize if there is at least one character, else we might potentially add begin/end multiple times
            if char_pos < position[0]:
                tokenized_text = self._tokenize(
                    text[char_pos : position[0]], begin=(char_pos == 0), end=position[0] > len(text) - 1
                )
                token_ids.extend(tokenized_text)
            char_pos = position[0]
            # beginning_of_text = False
            if position[1] == "image":
                if position[0] == 0:
                    # image should be after the bos token
                    image_token_positions.append(1)
                else:
                    image_token_positions.append(len(token_ids))
            elif position[1] == "span_start":
                assert (
                    current_span_start is None
                ), "Starting a new span before current has ended, please check for overlapping spans"
                current_span_start = len(token_ids)
            elif position[1] == "span_end":
                assert (
                    current_span_start is not None
                ), "Closing a span that has not started, please check for overlapping spans"
                # spans are inclusive, so we take the index of the last token in the span
                token_spans.append((current_span_start, len(token_ids) - 1))
                current_span_start = None
        # Handle any remaining text after the last position and add EOS token
        if char_pos < len(text):
            tokenized_text = self._tokenize(text[char_pos:], begin=(char_pos == 0), end=True)
            token_ids.extend(tokenized_text)

        return token_ids, token_spans, image_token_positions

    def detokenize(self, token_ids: int | list[int] | np.ndarray | torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id
