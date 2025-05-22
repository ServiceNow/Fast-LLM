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
        self, text: str, char_spans=None, image_positions=None, audio_positions=None
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """
        Tokenize the input text and return the tokenized input_ids and if provided, token spans and image positions.
        """
        image_positions = image_positions or []
        audio_positions = audio_positions or []
        char_spans = char_spans or []

        if len(set(image_positions).intersection(audio_positions)) > 0:
            raise ValueError("Image and audio can not have the same position.")
        multimodal_positions = sorted(image_positions + audio_positions)

        mm_idx = 0
        char_pos = 0
        token_ids = []
        image_token_positions = []
        audio_token_positions = []
        token_spans = []
        beginning_of_text = True
        multimodal_position = multimodal_positions[mm_idx] if mm_idx < len(multimodal_positions) else float("inf")
        for start, end in char_spans:
            # tokenize text, compute mm token position before span
            while multimodal_position <= start:
                # tokenize text before mm position
                tokenized_text = self._tokenize(text[char_pos:multimodal_position], begin=beginning_of_text, end=False)
                beginning_of_text = False
                token_ids.extend(tokenized_text)

                # update mm token positions
                multimodal_type = "image" if multimodal_position in image_positions else "audio"
                if multimodal_type == "image":
                    image_token_positions.append(len(token_ids))
                else:
                    audio_token_positions.append(len(token_ids))

                # updates
                mm_idx += 1
                char_pos = multimodal_position
                multimodal_position = (
                    multimodal_positions[mm_idx] if mm_idx < len(multimodal_positions) else float("inf")
                )

            # tokenize remaining text before span
            if char_pos < start:
                self._tokenize(text[char_pos:start], begin=beginning_of_text, end=False)
                beginning_of_text = False
                token_ids.extend(tokenized_text)

            char_pos = start
            span_length = 0
            token_start = len(token_ids)

            # tokenize text, compute mm token position within span
            while multimodal_position <= end:
                # tokenize text before mm position
                tokenized_text = self._tokenize(text[char_pos:multimodal_position], begin=beginning_of_text, end=False)
                beginning_of_text = False
                token_ids.extend(tokenized_text)

                # update mm token positions
                multimodal_type = "image" if multimodal_position in image_positions else "audio"
                if multimodal_type == "image":
                    image_token_positions.append(len(token_ids))
                else:
                    audio_token_positions.append(len(token_ids))

                # updates
                span_length += len(tokenized_text)
                char_pos = multimodal_position
                mm_idx += 1
                multimodal_position = (
                    multimodal_positions[mm_idx] if mm_idx < len(multimodal_positions) else float("inf")
                )

            # tokenize remaining text until end of span
            if char_pos < end:
                if end >= len(text) - 1:
                    tokenized_text = self._tokenize(text[char_pos : end + 1], begin=beginning_of_text, end=True)
                else:
                    tokenized_text = self._tokenize(text[char_pos : end + 1], begin=beginning_of_text, end=False)
                beginning_of_text = False
                token_ids.extend(tokenized_text)
                span_length += len(tokenized_text)
                char_pos = end + 1

            # update token spans
            token_spans.append((token_start, token_start + span_length - 1))

        # tokenize text, compute mm token position after all spans
        while multimodal_position <= len(text):
            # tokenize text before mm position
            multimodal_position = multimodal_positions[mm_idx]
            tokenized_text = self._tokenize(text[char_pos:multimodal_position], begin=beginning_of_text, end=False)
            beginning_of_text = False
            token_ids.extend(tokenized_text)

            # update mm token positions
            multimodal_type = "image" if multimodal_position in image_positions else "audio"
            if multimodal_type == "image":
                image_token_positions.append(len(token_ids))
            else:
                audio_token_positions.append(len(token_ids))

            # updates
            char_pos = multimodal_position
            mm_idx += 1
            multimodal_position = multimodal_positions[mm_idx] if mm_idx < len(multimodal_positions) else float("inf")

        # tokenize text after all spans
        tokenized_text = self._tokenize(text[char_pos:], begin=beginning_of_text, end=True)
        token_ids.extend(tokenized_text)

        return token_ids, token_spans, image_token_positions, audio_token_positions

    def detokenize(self, token_ids: int | list[int] | np.ndarray | torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id
