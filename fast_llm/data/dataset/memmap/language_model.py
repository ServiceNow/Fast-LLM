import dataclasses
import io
import pathlib
import tempfile
import typing

import torch

from fast_llm.data.dataset.memmap.abstract import MemmapIndexedDatasetReader, MemmapWriter
from fast_llm.data.dataset.memmap.config import LanguageModelReaderConfig, NullReaderConfig
from fast_llm.data.dataset.memmap.patch import PatchReader, PatchWriter
from fast_llm.data.dataset.memmap.range import RangeReader, RangeWriter
from fast_llm.data.dataset.memmap.token import TokenWriter
from fast_llm.data.document.abstract import Document
from fast_llm.data.document.config import ImageNormalizationConfig
from fast_llm.data.document.language_model import LanguageModelDocument
from fast_llm.utils import Assert


class LanguageModelReader[ConfigType: LanguageModelReaderConfig](MemmapIndexedDatasetReader[ConfigType]):
    def __init__(self, config: ConfigType, buffer: memoryview):
        super().__init__(config, buffer)
        # Using `buffer` and not `self._buffer` because nested offsets (`begin`, `end`) are global.
        self._tokens = self._config.tokens.get_reader(buffer)
        self._loss_masking_spans = self._config.loss_masking_spans.get_reader(buffer)
        self._chosen_spans = self._config.chosen_spans.get_reader(buffer)
        self._rejected_spans = self._config.rejected_spans.get_reader(buffer)
        self._image_patches = self._config.image_patches.get_reader(buffer)
        # TODO: ======= Move to model preprocessing ======
        self._image_normalization_config = ImageNormalizationConfig()

    @property
    def num_tokens(self) -> int:
        return self._config.tokens.num_tokens

    def get_document(self, index: int, begin: int, end: int) -> Document:
        image_patches = self._image_patches.get_document(index, begin, end)
        if image_patches is not None:
            image_patches.patches = self._image_normalization_config.normalize(image_patches.patches)
        return LanguageModelDocument(
            **dataclasses.asdict(self._tokens.get_document(index, begin, end)),
            loss_masking_spans=self._loss_masking_spans.get_document(index, begin, end),
            chosen_spans=self._chosen_spans.get_document(index, begin, end),
            rejected_spans=self._rejected_spans.get_document(index, begin, end),
            image_patches=image_patches,
        )

    def get_document_sizes(self) -> torch.Tensor:
        return self._tokens.get_document_sizes()

    def get_document_size(self, index: int) -> int:
        return self._tokens.get_document_size(index)

    def get_split(self, begin_ratio: float, end_ratio: float) -> tuple[int, int, dict[str, typing.Any]]:
        begin_index, end_index, token_metadata = self._tokens.get_split(begin_ratio, end_ratio)
        metadata = {
            "num_tokens": token_metadata["num_tokens"],
            "tokens": token_metadata,
        }
        if isinstance(self._loss_masking_spans, RangeReader):
            metadata["loss_masking_spans"] = self._loss_masking_spans.get_split(begin_index, end_index)
        if isinstance(self._chosen_spans, RangeReader):
            metadata["chosen_spans"] = self._chosen_spans.get_split(begin_index, end_index)
        if isinstance(self._rejected_spans, RangeReader):
            metadata["rejected_spans"] = self._rejected_spans.get_split(begin_index, end_index)
        if isinstance(self._image_patches, PatchReader):
            metadata["image_patches"] = self._image_patches.get_split(begin_index, end_index)

        return begin_index, end_index, metadata


class LanguageModelWriter(MemmapWriter):
    _use_loss_masking_spans: bool
    _use_preference_spans: bool
    _use_image_patches: bool

    def __enter__(self):
        super().__enter__()
        self._directory = tempfile.TemporaryDirectory()
        self._path = pathlib.Path(self._directory.name)
        # We write intermediate results in separate files so we don't need to iterate over the dataset multiple times.
        self._token_writer = TokenWriter(self._path.joinpath("tokens")).__enter__()
        self._loss_masking_span_writer = RangeWriter(self._path.joinpath("loss_masking_spans")).__enter__()
        self._chosen_spans_writer = RangeWriter(self._path.joinpath("chosen_spans")).__enter__()
        self._rejected_spans_writer = RangeWriter(self._path.joinpath("rejected_spans")).__enter__()
        self._image_patches_writer = PatchWriter(self._path.joinpath("image_patches")).__enter__()
        return self

    def write(self, document: LanguageModelDocument):
        super().write(document)
        # Write tokens.
        self._token_writer.write(document)

        use_loss_masking_spans = document.loss_masking_spans is not None
        use_preference_spans = document.chosen_spans is not None
        use_image_patches = document.image_patches is not None
        if hasattr(self, "_use_loss_masking_spans"):
            Assert.eq(self._use_loss_masking_spans, use_loss_masking_spans)
            Assert.eq(self._use_preference_spans, use_preference_spans)
            Assert.eq(self._use_image_patches, use_image_patches)
        else:
            self._use_loss_masking_spans = use_loss_masking_spans
            self._use_preference_spans = use_preference_spans
            self._use_image_patches = use_image_patches

        # Write loss masking spans.
        if use_loss_masking_spans:
            self._loss_masking_span_writer.write(document.loss_masking_spans)

        # Write preference spans.
        if use_preference_spans:
            assert document.rejected_spans is not None
            self._chosen_spans_writer.write(document.chosen_spans)
            self._rejected_spans_writer.write(document.rejected_spans)

        # Write image patches
        if use_image_patches:
            self._image_patches_writer.write(document.image_patches)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._token_writer.__exit__(exc_type, exc_val, exc_tb)
        self._loss_masking_span_writer.__exit__(exc_type, exc_val, exc_tb)
        self._chosen_spans_writer.__exit__(exc_type, exc_val, exc_tb)
        self._rejected_spans_writer.__exit__(exc_type, exc_val, exc_tb)
        self._image_patches_writer.__exit__(exc_type, exc_val, exc_tb)

        if exc_type is None:
            # A dummy config so we can verify the begin and end offsets.
            config = self._get_config(self._begin, None)
            _copy_chunked(self._path.joinpath("tokens"), self._stream, config.tokens.begin, config.tokens.end)

            if self._use_loss_masking_spans:
                _copy_chunked(
                    self._path.joinpath("loss_masking_spans"),
                    self._stream,
                    config.loss_masking_spans.begin,
                    config.loss_masking_spans.end,
                )
            if self._use_preference_spans:
                _copy_chunked(
                    self._path.joinpath("chosen_spans"),
                    self._stream,
                    config.chosen_spans.begin,
                    config.chosen_spans.end,
                )
                _copy_chunked(
                    self._path.joinpath("rejected_spans"),
                    self._stream,
                    config.rejected_spans.begin,
                    config.rejected_spans.end,
                )

            if self._use_image_patches:
                _copy_chunked(
                    self._path.joinpath("image_patches"),
                    self._stream,
                    config.image_patches.begin,
                    config.image_patches.end,
                )

        self._directory.cleanup()
        super().__exit__(exc_type, exc_val, exc_tb)

    @classmethod
    def _get_config_class(cls) -> type[LanguageModelReaderConfig]:
        return LanguageModelReaderConfig

    def _get_config(self, begin: int, end: int | None):
        tokens = self._token_writer.get_config(begin + len(LanguageModelReaderConfig.header))
        offset = tokens.end
        if self._use_loss_masking_spans:
            loss_masking_spans = self._loss_masking_span_writer.get_config(offset)
            offset = loss_masking_spans.end
        else:
            loss_masking_spans = NullReaderConfig()
        if self._use_preference_spans:
            chosen_spans = self._chosen_spans_writer.get_config(offset)
            offset = chosen_spans.end
            rejected_spans = self._rejected_spans_writer.get_config(offset)
            offset = rejected_spans.end
        else:
            chosen_spans = NullReaderConfig()
            rejected_spans = NullReaderConfig()
        if self._use_image_patches:
            image_patches = self._image_patches_writer.get_config(offset)
            offset = image_patches.end
        else:
            image_patches = NullReaderConfig()

        if end is None:
            end = offset + len(LanguageModelReaderConfig.footer)

        return LanguageModelReaderConfig(
            begin=begin,
            end=end,
            tokens=tokens,
            loss_masking_spans=loss_masking_spans,
            chosen_spans=chosen_spans,
            rejected_spans=rejected_spans,
            image_patches=image_patches,
        )


def _copy_chunked(path: pathlib.Path, stream: io.BufferedWriter, expected_begin: int, expected_end: int):
    # Copy temporary file content in chunks of 100 MB.
    Assert.eq(stream.tell(), expected_begin)
    with path.open("rb") as input_stream:
        while data := input_stream.read(100000000):
            stream.write(data)
    Assert.eq(stream.tell(), expected_end)
