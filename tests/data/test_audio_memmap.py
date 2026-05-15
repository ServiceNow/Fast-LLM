"""
Unit tests for the audio memmap reader/writer (AudioReader, AudioWriter, AudioReaderConfig).

Tests the round-trip property: documents written with AudioWriter can be read back
identically with AudioReader.
"""

import io

import torch
import pytest

from fast_llm.data.dataset.memmap.audio import AudioReader, AudioWriter
from fast_llm.data.dataset.memmap.config import AudioReaderConfig
from fast_llm.data.document.audio import AudioDocument


def _make_document(clip_samples: list[list[float]], positions: list[int]) -> AudioDocument:
    return AudioDocument(
        samples=[torch.tensor(s, dtype=torch.float32) for s in clip_samples],
        positions=torch.tensor(positions, dtype=torch.int32),
    )


def _round_trip(documents: list[AudioDocument | None]) -> tuple[AudioReaderConfig, AudioReader]:
    """Write documents to an in-memory buffer, then return the config and reader."""
    buf = io.BytesIO()
    config = AudioWriter.write_dataset(buf, documents)
    data = buf.getvalue()
    mv = memoryview(data)
    reader = config.get_reader(mv)
    return config, reader


def test_audio_memmap_single_document_single_clip():
    """Single document with one clip round-trips correctly."""
    samples = [0.1, 0.2, 0.3, 0.4, 0.5]
    doc = _make_document([samples], [7])
    config, reader = _round_trip([doc])

    assert config.num_documents == 1
    assert config.num_clips == 1
    assert config.num_samples == len(samples)

    result = reader.get_document(0, 0, 1)
    assert result is not None
    assert len(result.samples) == 1
    assert result.samples[0].tolist() == pytest.approx(samples)
    assert result.positions.tolist() == [7]


def test_audio_memmap_single_document_multiple_clips():
    """Single document with two clips round-trips correctly."""
    s0 = [0.1, 0.2]
    s1 = [0.3, 0.4, 0.5]
    doc = _make_document([s0, s1], [3, 9])
    config, reader = _round_trip([doc])

    assert config.num_clips == 2
    assert config.num_samples == len(s0) + len(s1)

    result = reader.get_document(0, 0, 1)
    assert result is not None
    assert len(result.samples) == 2
    assert result.samples[0].tolist() == pytest.approx(s0)
    assert result.samples[1].tolist() == pytest.approx(s1)
    assert result.positions.tolist() == [3, 9]


def test_audio_memmap_multiple_documents():
    """Multiple documents can be read back individually."""
    doc0 = _make_document([[1.0, 2.0]], [0])
    doc1 = _make_document([[3.0], [4.0, 5.0]], [2, 8])
    doc2 = _make_document([[6.0, 7.0, 8.0]], [1])
    config, reader = _round_trip([doc0, doc1, doc2])

    assert config.num_documents == 3
    assert config.num_clips == 1 + 2 + 1
    assert config.num_samples == 2 + 1 + 2 + 3

    r0 = reader.get_document(0, 0, 1)
    assert r0 is not None and r0.samples[0].tolist() == pytest.approx([1.0, 2.0])

    r1 = reader.get_document(1, 0, 1)
    assert r1 is not None
    assert r1.samples[0].tolist() == pytest.approx([3.0])
    assert r1.samples[1].tolist() == pytest.approx([4.0, 5.0])
    assert r1.positions.tolist() == [2, 8]

    r2 = reader.get_document(2, 0, 1)
    assert r2 is not None and r2.samples[0].tolist() == pytest.approx([6.0, 7.0, 8.0])


def test_audio_memmap_none_document_returns_none():
    """Documents written as None produce None on read-back."""
    doc = _make_document([[1.0]], [0])
    config, reader = _round_trip([None, doc, None])

    assert config.num_documents == 3

    assert reader.get_document(0, 0, 1) is None
    assert reader.get_document(2, 0, 1) is None

    r1 = reader.get_document(1, 0, 1)
    assert r1 is not None and r1.samples[0].tolist() == pytest.approx([1.0])


def test_audio_memmap_empty_dataset():
    """Dataset with no documents has all-zero metadata."""
    config, reader = _round_trip([])

    assert config.num_documents == 0
    assert config.num_clips == 0
    assert config.num_samples == 0


def test_audio_memmap_all_none_documents():
    """All-None dataset has zero clips and samples."""
    config, reader = _round_trip([None, None, None])

    assert config.num_documents == 3
    assert config.num_clips == 0
    assert config.num_samples == 0

    for i in range(3):
        assert reader.get_document(i, 0, 1) is None


def test_audio_memmap_variable_clip_lengths():
    """Clips of different lengths within the same document are stored accurately."""
    lengths = [1, 5, 2, 10, 3]
    clips = [list(range(n)) for n in lengths]
    positions = list(range(len(lengths)))
    doc = _make_document(clips, positions)
    config, reader = _round_trip([doc])

    assert config.num_samples == sum(lengths)

    result = reader.get_document(0, 0, 1)
    assert result is not None
    for i, (expected, got) in enumerate(zip(clips, result.samples)):
        assert got.tolist() == pytest.approx(expected), f"clip {i} mismatch"


def test_audio_memmap_positions_preserved():
    """Token positions are stored and retrieved without modification."""
    positions = [0, 15, 42, 1023]
    doc = _make_document([[float(i)] for i in range(len(positions))], positions)
    _, reader = _round_trip([doc])

    result = reader.get_document(0, 0, 1)
    assert result is not None
    assert result.positions.tolist() == positions


def test_audio_memmap_config_buffer_size_consistent():
    """AudioReaderConfig._expected_buffer_size matches actual bytes written."""
    doc0 = _make_document([[1.0, 2.0], [3.0]], [0, 5])
    doc1 = _make_document([[4.0]], [1])
    buf = io.BytesIO()
    config = AudioWriter.write_dataset(buf, [doc0, doc1])

    body_bytes = len(buf.getvalue()) - len(AudioReaderConfig.header) - len(AudioReaderConfig.footer)
    assert body_bytes == config._expected_buffer_size
