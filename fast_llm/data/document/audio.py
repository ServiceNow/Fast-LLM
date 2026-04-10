import dataclasses
import typing

import torch

from fast_llm.data.document.abstract import Batch, Document
from fast_llm.layers.audio_encoder.config import AudioKwargs
from fast_llm.layers.language_model.config import LanguageModelKwargs


@dataclasses.dataclass(kw_only=True)
class AudioDocument(Document):
    """
    Variable-length audio clips with their positions in the token sequence.

    Each clip is a 1-D float32 waveform (raw audio samples before mel-spectrogram
    extraction).  Positions map each clip to the text-token index that will be
    replaced by the audio encoder's output tokens.
    """

    samples: list[torch.Tensor]  # one float32 tensor per clip, variable length
    positions: torch.Tensor  # int32, shape (num_clips,) — token position per clip

    def __post_init__(self):
        assert len(self.samples) == self.positions.numel(), (
            f"AudioDocument: mismatch between samples ({len(self.samples)}) "
            f"and positions ({self.positions.numel()})"
        )


@dataclasses.dataclass(kw_only=True)
class AudioBatch(Batch, AudioDocument):
    """
    Batched audio documents, produced by ``AudioBatch.from_documents``.
    """

    @classmethod
    def from_documents(
        cls,
        documents: "typing.Sequence[AudioDocument | None]",
        sizes: typing.Iterable[int],
    ) -> "typing.Self | None":
        """
        Merge a list of per-sequence ``AudioDocument`` objects into a single batch,
        adjusting ``positions`` to be relative to each sequence's start offset.
        """
        docs_with_offset = []
        offset = 0
        for document, size in zip(documents, sizes, strict=True):
            if document is not None and len(document.samples) > 0:
                docs_with_offset.append((document, offset))
            offset += size

        if not docs_with_offset:
            return None

        return cls(
            samples=[s for doc, _ in docs_with_offset for s in doc.samples],
            positions=torch.cat([doc.positions + offset for doc, offset in docs_with_offset]),
        )

    def to_kwargs(self) -> dict[str, typing.Any]:
        return {
            AudioKwargs.audio: self.samples,
            AudioKwargs.audio_positions: self.positions,
        }
