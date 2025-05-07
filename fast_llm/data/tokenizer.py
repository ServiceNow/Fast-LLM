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

    def tokenize(self, text, image_positions=None, audio_positions=None):
        image_positions = image_positions or []
        audio_positions = audio_positions or []
        if len(set(image_positions).intersection(audio_positions)) > 0:
            raise ValueError("Image and audio can not have the same position.")
        multimodal_positions = sorted(image_positions + audio_positions)
        if not multimodal_positions:
            return self._tokenize(text), [], []
        multimodel_idx = 0
        char_pos = 0
        token_ids = []
        image_token_positions = []
        audio_token_positions = []
        beginning_of_text = True
        while multimodel_idx < len(multimodal_positions):
            multimodal_char_pos = multimodal_positions[multimodel_idx]
            multimodal_type = "image" if multimodal_char_pos in image_positions else "audio"

            if multimodal_char_pos > len(text):
                raise ValueError(
                    f"{multimodal_type.capitalize()} position {multimodal_char_pos} is greater than text length {len(text)}"
                )
            curr_text = text[char_pos:multimodal_char_pos]
            tokenized_text = self._tokenize(curr_text, begin=beginning_of_text, end=multimodal_char_pos >= len(text))
            beginning_of_text = False
            token_ids.extend(tokenized_text)

            # store multimodal token positions
            if multimodal_type == "image":
                image_token_positions.append(len(token_ids))
            else:
                audio_token_positions.append(len(token_ids))
            char_pos = multimodal_char_pos
            multimodel_idx += 1
        if char_pos < len(text):
            curr_text = text[char_pos:]
            tokenized_text = self._tokenize(curr_text, begin=beginning_of_text, end=True)
            token_ids.extend(tokenized_text)
        return token_ids, image_token_positions, audio_token_positions

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
                tokenized_text = self.tokenize(curr_text, begin=beginning_of_text, end=False)
                beginning_of_text = False
                input_ids.extend(tokenized_text)
            curr_text = text[start : end + 1]
            if end >= len(text) - 1:
                tokenized_text = self.tokenize(curr_text, begin=beginning_of_text, end=True)
            else:
                tokenized_text = self.tokenize(curr_text, begin=beginning_of_text, end=False)
            beginning_of_text = False
            token_spans.append((len(input_ids), len(input_ids) + len(tokenized_text) - 1))
            input_ids.extend(tokenized_text)
            char_pos = end + 1
        if char_pos < len(text):
            curr_text = text[char_pos:]
            tokenized_text = self.tokenize(curr_text, begin=beginning_of_text, end=True)
            input_ids.extend(tokenized_text)
        return input_ids, token_spans

    def detokenize(self, token_ids: int | list[int] | np.ndarray | torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id
