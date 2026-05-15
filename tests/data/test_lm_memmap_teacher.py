"""Round-trip tests for the parallel teacher stream in LanguageModelReader / LanguageModelWriter.

The teacher stream is a recursive `LanguageModelDocument` written into the same
shard as the student.  These tests verify that a document bearing a teacher
sub-document survives a write/read round trip with bytewise-identical contents
on both streams, and that legacy documents without a teacher still work.
"""

import io
import pathlib
import tempfile

import torch

from fast_llm.data.dataset.memmap.language_model import LanguageModelReader, LanguageModelWriter
from fast_llm.data.dataset.memmap.memmap import MemmapDataset
from fast_llm.data.document.language_model import LanguageModelDocument
from fast_llm.data.document.range import RangeDocument


def _make_doc(token_count: int, mask_end: int, base: int, teacher_token_count: int | None = None) -> LanguageModelDocument:
    """Build a student document with optional teacher.  Token IDs are deterministic
    (offset by `base`) so the round-trip check can match exactly."""
    teacher = None
    if teacher_token_count is not None:
        teacher = LanguageModelDocument(
            tokens=torch.arange(teacher_token_count, dtype=torch.int64) + base + 10000,
            loss_masking_spans=RangeDocument(
                ranges=[(0, max(0, teacher_token_count - (token_count - mask_end)))]
            ),
        )
    return LanguageModelDocument(
        tokens=torch.arange(token_count, dtype=torch.int64) + base,
        loss_masking_spans=RangeDocument(ranges=[(0, mask_end)]),
        teacher=teacher,
    )


def _round_trip(documents: list[LanguageModelDocument]) -> tuple[LanguageModelReader, MemmapDataset]:
    """Write to a temp file via MemmapDataset.write_dataset, then re-open."""
    tmp = tempfile.NamedTemporaryFile(suffix=".fast_llm_dataset", delete=False)
    tmp.close()
    path = pathlib.Path(tmp.name)
    try:
        MemmapDataset.write_dataset(path, documents, LanguageModelWriter)
        ds = MemmapDataset("test", path)
        return ds.reader, ds
    except Exception:
        path.unlink(missing_ok=True)
        raise


def test_legacy_document_without_teacher_round_trips():
    """No-teacher documents still write/read correctly (teacher field is None)."""
    documents = [_make_doc(20, 12, base=100), _make_doc(15, 8, base=200)]
    reader, ds = _round_trip(documents)
    try:
        assert not reader._config.has_teacher
        for i, doc in enumerate(documents):
            got = ds.get_document(i)
            assert torch.equal(got.tokens, doc.tokens)
            assert got.teacher is None
    finally:
        pathlib.Path(ds._path).unlink(missing_ok=True)


def test_document_with_teacher_round_trips():
    """Teacher tokens / loss_masking_spans survive a write/read round trip and
    have lengths independent of the student."""
    # Two documents with student/teacher length pairs (20, 25) and (30, 22)
    # -- intentionally varying which side is longer to exercise both directions.
    documents = [
        _make_doc(20, 12, base=0, teacher_token_count=25),
        _make_doc(30, 18, base=500, teacher_token_count=22),
    ]
    reader, ds = _round_trip(documents)
    try:
        assert reader._config.has_teacher
        assert len(reader._teacher) == len(documents)
        for i, doc in enumerate(documents):
            got = ds.get_document(i)
            assert torch.equal(got.tokens, doc.tokens)
            assert got.teacher is not None
            assert torch.equal(got.teacher.tokens, doc.teacher.tokens)
            # Spans round-trip exactly (input is list-of-tuples, output is a
            # tensor; compare via list-of-lists).
            def _as_list(ranges):
                return ranges.tolist() if hasattr(ranges, "tolist") else [list(r) for r in ranges]

            assert _as_list(got.teacher.loss_masking_spans.ranges) == _as_list(
                doc.teacher.loss_masking_spans.ranges
            )
            # And the unmasked region count matches across student/teacher (the
            # contract gather-then-KL relies on).
            student_unmasked = len(got.tokens) - doc.loss_masking_spans.ranges[0][1]
            teacher_unmasked = (
                len(got.teacher.tokens) - doc.teacher.loss_masking_spans.ranges[0][1]
            )
            assert student_unmasked == teacher_unmasked, (
                f"doc {i}: student unmasked={student_unmasked} vs teacher unmasked={teacher_unmasked}"
            )
    finally:
        pathlib.Path(ds._path).unlink(missing_ok=True)


def test_teacher_buffer_size_matches_expected():
    """The on-disk size matches `_expected_buffer_size`, which now includes teacher bytes."""
    documents = [_make_doc(20, 12, base=0, teacher_token_count=25)]
    tmp = tempfile.NamedTemporaryFile(suffix=".fast_llm_dataset", delete=False)
    tmp.close()
    path = pathlib.Path(tmp.name)
    try:
        config = MemmapDataset.write_dataset(path, documents, LanguageModelWriter)
        assert config.has_teacher
        # The reader-config validator already enforces `end - begin == expected_buffer_size`
        # at construction time; reaching here means it passed.  Sanity-check that
        # the teacher contributes more than zero bytes.
        assert config.teacher.expected_buffer_size > 0
    finally:
        path.unlink(missing_ok=True)
