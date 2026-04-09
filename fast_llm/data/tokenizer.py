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
        ) # type: ignore

    def tokenize(
        self, text: str, char_spans=None, image_positions=None, audio_positions=None
    ) -> tuple[list[int], list[tuple[int, int]], list[int], list[int]]:
        """
        Tokenize text that may contain multimodal markers and character spans of interest.

        Approach: collect all "cut points" — mm positions, span boundaries, and end-of-text —
        then make a single left-to-right pass. At each cut point, the text segment since the
        last cut is tokenized and the current token index is recorded in char_to_token. BOS is
        prepended to the first non-empty chunk; EOS is appended to the chunk ending at len(text).

        Token spans are derived directly from the mapping:
            token_span = (char_to_token[s], char_to_token[e+1] - 1)

        Args:
            text: Raw input string.
            char_spans: Character-index ranges (start, end inclusive) to track; returned as
                inclusive token-index ranges in token_spans.
            image_positions: Character positions where an image token will be inserted.
            audio_positions: Character positions where an audio token will be inserted.
                Image and audio positions must be disjoint.

        Returns:
            token_ids: Flat list of token IDs for the full text.
            token_spans: Inclusive token-index ranges corresponding to each input char_span.
            image_token_positions: Token indices at which image markers are to be inserted.
            audio_token_positions: Token indices at which audio markers are to be inserted.
        """
        image_positions = set(image_positions or [])
        audio_positions = set(audio_positions or [])
        char_spans = char_spans or []

        if image_positions & audio_positions:
            raise ValueError("Image and audio cannot have the same position.")

        text_len = len(text)

        # All positions where we need to record a token index or insert an mm marker.
        # Always include text_len so the final chunk is emitted inside the loop.
        cut_points = sorted(
            (image_positions | audio_positions)
            | {s for s, _ in char_spans}
            | {e + 1 for _, e in char_spans}
            | {text_len}
        )

        # BOS and EOS are emitted unconditionally, bracketing everything including mm markers.
        # This ensures mm markers always land between BOS and EOS even for empty text.
        # char_to_token[0] is pre-set to 0 so that spans starting at char 0 include BOS.
        # setdefault in the loop prevents that entry from being overwritten.
        token_ids: list[int] = [self.bod_id]
        image_token_positions: list[int] = []
        audio_token_positions: list[int] = []
        char_to_token: dict[int, int] = {0: 0}  # char index -> token index of first token from text[char:]

        char_pos = 0

        for cut in cut_points:
            # Emit text[char_pos:cut] with no BOS/EOS (handled outside the loop)
            if char_pos < cut:
                tokens = self._tokenize(text[char_pos:cut], begin=False, end=False)
                token_ids.extend(tokens)
                char_pos = cut

            # Record token index at this cut point; setdefault preserves the char 0 → token 0 entry
            char_to_token.setdefault(cut, len(token_ids))

            # Record mm marker position (placeholder; caller inserts actual modality tokens here)
            if cut in image_positions:
                image_token_positions.append(len(token_ids))
            elif cut in audio_positions:
                audio_token_positions.append(len(token_ids))

        token_ids.append(self.eod_id)

        token_spans = [
            (char_to_token[s], char_to_token[e + 1] - 1)
            for s, e in char_spans
        ]

        return token_ids, token_spans, image_token_positions, audio_token_positions

    def detokenize(self, token_ids: int | list[int] | np.ndarray | torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id
