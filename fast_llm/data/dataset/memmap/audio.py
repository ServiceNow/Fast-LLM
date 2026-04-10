import typing

import numpy as np
import torch

from fast_llm.data.dataset.memmap.abstract import MemmapIndexedDatasetReader, MemmapWriter
from fast_llm.data.document.audio import AudioDocument

from fast_llm.data.dataset.memmap.config import AudioReaderConfig


class AudioReader[ConfigType: AudioReaderConfig](MemmapIndexedDatasetReader[ConfigType]):
    """
    Reads variable-length audio clips from a memmap buffer.

    Buffer layout (matching ``AudioReaderConfig._expected_buffer_size``):
      [float32 × num_samples]          — concatenated raw audio waveforms
      [int32   × num_clips]            — length (in float32 samples) per clip
      [int32   × num_clips]            — token position of each clip in the LM sequence
      [int32   × (num_documents + 1)]  — CSR clip offsets per document
    """

    def __init__(self, config: AudioReaderConfig, buffer: memoryview):
        super().__init__(config, buffer)

        offset = 0
        # torch.frombuffer requires count > 0; use empty tensors for absent data.
        if config.num_samples > 0:
            self._samples_flat = torch.frombuffer(
                self._buffer, dtype=torch.float32, count=config.num_samples, offset=offset
            )
        else:
            self._samples_flat = torch.empty(0, dtype=torch.float32)
        offset += config.num_samples * torch.float32.itemsize

        if config.num_clips > 0:
            self._clip_lengths = torch.frombuffer(
                self._buffer, dtype=torch.int32, count=config.num_clips, offset=offset
            )
            offset += config.num_clips * torch.int32.itemsize
            self._clip_positions = torch.frombuffer(
                self._buffer, dtype=torch.int32, count=config.num_clips, offset=offset
            )
            offset += config.num_clips * torch.int32.itemsize
        else:
            self._clip_lengths = torch.empty(0, dtype=torch.int32)
            self._clip_positions = torch.empty(0, dtype=torch.int32)

        self._clip_offsets = torch.frombuffer(
            self._buffer, dtype=torch.int32, count=config.num_documents + 1, offset=offset
        )

        # Precompute per-clip start offsets into the flat sample buffer.
        self._sample_offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64), torch.cumsum(self._clip_lengths.to(torch.int64), dim=0)]
        )

    def get_document(self, index: int, begin: int, end: int) -> AudioDocument | None:
        clip_begin = self._clip_offsets[index].item()
        clip_end = self._clip_offsets[index + 1].item()
        if clip_begin == clip_end:
            return None

        positions = self._clip_positions[clip_begin:clip_end].clone()
        samples = []
        for i in range(clip_begin, clip_end):
            s_begin = self._sample_offsets[i].item()
            s_end = self._sample_offsets[i + 1].item()
            samples.append(self._samples_flat[s_begin:s_end].clone())

        return AudioDocument(samples=samples, positions=positions)

    def get_document_sizes(self) -> torch.Tensor:
        return torch.zeros(self._config.num_documents, dtype=torch.int32)

    def get_document_size(self, index: int) -> int:
        return 0

    def get_split(self, begin_ratio: float, end_ratio: float) -> tuple[int, int, dict[str, typing.Any]]:
        return self._config.get_split(begin_ratio, end_ratio)


class AudioWriter(MemmapWriter):
    """Context-manager writer for the audio section of a memmap file."""

    def __enter__(self):
        super().__enter__()
        self._samples: list[float] = []
        self._clip_lengths: list[int] = []
        self._clip_positions: list[int] = []
        self._clip_offsets: list[int] = [0]  # CSR: one entry per document
        return self

    def write(self, document: AudioDocument | None):
        super().write(document)
        if document is None or len(document.samples) == 0:
            self._clip_offsets.append(self._clip_offsets[-1])
            return

        for samples, position in zip(document.samples, document.positions.tolist()):
            self._samples.extend(samples.tolist())
            self._clip_lengths.append(len(samples))
            self._clip_positions.append(int(position))

        self._clip_offsets.append(len(self._clip_lengths))

    def _get_config(self, begin: int, end: int | None):
        from fast_llm.data.dataset.memmap.config import AudioReaderConfig

        num_documents = len(self._clip_offsets) - 1
        num_clips = len(self._clip_lengths)
        num_samples = len(self._samples)

        body_size = (
            num_samples * torch.float32.itemsize
            + num_clips * torch.int32.itemsize  # lengths
            + num_clips * torch.int32.itemsize  # positions
            + (num_documents + 1) * torch.int32.itemsize  # CSR offsets
        )
        if end is None:
            end = begin + len(AudioReaderConfig.header) + body_size + len(AudioReaderConfig.footer)
        return AudioReaderConfig(
            begin=begin,
            end=end,
            num_documents=num_documents,
            num_clips=num_clips,
            num_samples=num_samples,
        )

    @classmethod
    def _get_config_class(cls):
        from fast_llm.data.dataset.memmap.config import AudioReaderConfig

        return AudioReaderConfig

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Write body: parent already wrote header; parent __exit__ will write footer.
            self._stream.write(np.array(self._samples, dtype=np.float32).tobytes())
            self._stream.write(np.array(self._clip_lengths, dtype=np.int32).tobytes())
            self._stream.write(np.array(self._clip_positions, dtype=np.int32).tobytes())
            self._stream.write(np.array(self._clip_offsets, dtype=np.int32).tobytes())

        super().__exit__(exc_type, exc_val, exc_tb)
