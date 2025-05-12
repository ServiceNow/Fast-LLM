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

    def tokenize(self, text: str, image_positions=None, char_spans=None) -> tuple[list[int], list[tuple[int, int]]]:
        """
        Tokenize the input text and return the tokenized input_ids along with token spans.
        """
        # if not image_positions and not char_spans:
        #     return self._tokenize(text), [], []
        if not image_positions:
            image_positions = []
        if not char_spans:
            char_spans = []

        image_idx = 0
        char_pos = 0
        token_ids = []
        image_token_positions = []
        beginning_of_text = True
        for start, end in char_spans:
            image_position = image_positions[image_idx] if image_idx < len(image_positions) else float("inf")
            while image_position <= start:
                tokenized_text = self._tokenize(text[char_pos:image_position], begin=beginning_of_text, end=False)
                beginning_of_text = False
                image_token_positions.append(len(token_ids))
                token_ids.extend(tokenized_text)
                image_idx += 1
                char_pos = image_position
                image_position = image_positions[image_idx] if image_idx < len(image_positions) else float("inf")
            if char_pos < start:
                self._tokenize(text[char_pos:start], begin=beginning_of_text, end=False)
                beginning_of_text = False
                token_ids.extend(tokenized_text)
            char_pos = start
            len(token_ids)
            span_length = 0
            while image_position <= end:
                tokenized_text = self._tokenize(text[char_pos:image_position], begin=beginning_of_text, end=False)
                beginning_of_text = False
                image_token_positions.append(len(token_ids))
                token_ids.extend(tokenized_text)
                span_length += len(tokenized_text)
                char_pos = image_position
                image_idx += 1
                image_position = image_positions[image_idx] if image_idx < len(image_positions) else float("inf")
            if char_pos < end:
                if end >= len(text) - 1:
                    tokenized_text = self._tokenize(text[char_pos : end + 1], begin=beginning_of_text, end=True)
                    beginning_of_text = False
                    token_ids.extend(tokenized_text)
                    span_length += len(tokenized_text)
                    char_pos = end + 1
                else:
                    tokenized_text = self._tokenize(text[char_pos : end + 1], begin=beginning_of_text, end=False)
                    beginning_of_text = False
                    token_ids.extend(tokenized_text)
                    span_length += len(tokenized_text)

    # def tokenize(self, text, image_positions=None):
    #     if not image_positions:
    #         return self._tokenize(text), [], []
    #     image_idx = 0
    #     char_pos = 0
    #     token_ids = []
    #     image_token_positions = []
    #     beginning_of_text = True
    #     while image_idx < len(image_positions):
    #         if image_positions[image_idx] > len(text):
    #             raise ValueError(
    #                 f"Image position {image_positions[image_idx]} is greater than text length {len(text)}"
    #             )
    #         curr_text = text[char_pos : image_positions[image_idx]]
    #         tokenized_text = self._tokenize(
    #             curr_text, begin=beginning_of_text, end=image_positions[image_idx] >= len(text)
    #         )
    #         beginning_of_text = False
    #         token_ids.extend(tokenized_text)
    #         image_token_positions = len(token_ids)
    #         char_pos = image_positions[image_idx]
    #         image_idx += 1
    #     if char_pos < len(text):
    #         curr_text = text[char_pos:]
    #         tokenized_text = self._tokenize(curr_text, begin=beginning_of_text, end=True)
    #         token_ids.extend(tokenized_text)
    #     return token_ids, image_token_positions

    # def tokenize_with_spans(
    #     self, text: str, char_spans: list[tuple[int, int]]
    # ) -> tuple[list[int], list[tuple[int, int]]]:
    #     """
    #     Perform span-aware tokenization and return the tokenized input_ids along with token spans.
    #     """
    #     input_ids = []
    #     token_spans = []
    #     char_pos = 0
    #     beginning_of_text = True
    #     for start, end in char_spans:
    #         if char_pos < start:
    #             curr_text = text[char_pos:start]
    #             tokenized_text = self.tokenize(curr_text, begin=beginning_of_text, end=False)
    #             beginning_of_text = False
    #             input_ids.extend(tokenized_text)
    #         curr_text = text[start : end + 1]
    #         if end >= len(text) - 1:
    #             tokenized_text = self.tokenize(curr_text, begin=beginning_of_text, end=True)
    #         else:
    #             tokenized_text = self.tokenize(curr_text, begin=beginning_of_text, end=False)
    #         beginning_of_text = False
    #         token_spans.append((len(input_ids), len(input_ids) + len(tokenized_text) - 1))
    #         input_ids.extend(tokenized_text)
    #         char_pos = end + 1
    #     if char_pos < len(text):
    #         curr_text = text[char_pos:]
    #         tokenized_text = self.tokenize(curr_text, begin=beginning_of_text, end=True)
    #         input_ids.extend(tokenized_text)
    #     return input_ids, token_spans

    def detokenize(self, token_ids: int | list[int] | np.ndarray | torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id
