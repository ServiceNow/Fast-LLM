import dataclasses
import io
import pathlib
import tempfile
import typing

import torch

from fast_llm.data.dataset.memmap.abstract import MemmapIndexedDatasetReader, MemmapWriter
from fast_llm.data.dataset.memmap.audio import AudioWriter
from fast_llm.data.dataset.memmap.config import LanguageModelReaderConfig, NullReaderConfig
from fast_llm.data.dataset.memmap.patch import PatchWriter
from fast_llm.data.dataset.memmap.range import RangeWriter
from fast_llm.data.dataset.memmap.token import TokenWriter
from fast_llm.data.dataset.memmap.token_data import TokenDataWriter
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
        self._audio = self._config.audio.get_reader(buffer)
        self._advantages = self._config.advantages.get_reader(buffer)
        self._old_log_probabilities = self._config.old_log_probabilities.get_reader(buffer)

    @property
    def num_tokens(self) -> int:
        return self._config.tokens.num_tokens

    def get_document(self, index: int, begin: int, end: int) -> LanguageModelDocument:
        image_patches = self._image_patches.get_document(index, begin, end)
        audio = self._audio.get_document(index, begin, end)
        return LanguageModelDocument(
            **dataclasses.asdict(self._tokens.get_document(index, begin, end)),
            loss_masking_spans=self._loss_masking_spans.get_document(index, begin, end),
            chosen_spans=self._chosen_spans.get_document(index, begin, end),
            rejected_spans=self._rejected_spans.get_document(index, begin, end),
            image_patches=image_patches,
            audio=audio,
            advantages=self._advantages.get_document(index, begin, end),
            old_log_probabilities=self._old_log_probabilities.get_document(index, begin, end),
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
        if self._config.has_loss_masking_spans:
            metadata["loss_masking_spans"] = self._loss_masking_spans.get_split(begin_index, end_index)
        if self._config.has_preference_spans:
            metadata["chosen_spans"] = self._chosen_spans.get_split(begin_index, end_index)
            metadata["rejected_spans"] = self._rejected_spans.get_split(begin_index, end_index)
        if self._config.has_image_patches:
            metadata["image_patches"] = self._image_patches.get_split(begin_index, end_index)
        if self._config.has_audio:
            metadata["audio"] = self._audio.get_split(begin_index, end_index)
        if self._config.has_grpo_data:
            metadata["advantages"] = self._advantages.get_split(begin_index, end_index)
            metadata["old_log_probabilities"] = self._old_log_probabilities.get_split(begin_index, end_index)
        return begin_index, end_index, metadata


class LanguageModelWriter(MemmapWriter):
    _use_loss_masking_spans: bool
    _use_preference_spans: bool
    _use_image_patches: bool
    _use_audio: bool
    _use_grpo_data: bool

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
        self._audio_writer = AudioWriter(self._path.joinpath("audio")).__enter__()
        self._advantages_writer = TokenDataWriter(self._path.joinpath("advantages")).__enter__()
        self._old_log_probabilities_writer = TokenDataWriter(self._path.joinpath("old_log_probabilities")).__enter__()
        return self

    def write(self, document: LanguageModelDocument):
        super().write(document)
        # Write tokens.
        self._token_writer.write(document)

        use_loss_masking_spans = document.loss_masking_spans is not None
        use_preference_spans = document.chosen_spans is not None
        use_image_patches = document.image_patches is not None
        use_audio = document.audio is not None
        use_grpo_data = document.advantages is not None
        if hasattr(self, "_use_loss_masking_spans"):
            Assert.eq(self._use_loss_masking_spans, use_loss_masking_spans)
            Assert.eq(self._use_preference_spans, use_preference_spans)
            Assert.eq(self._use_image_patches, use_image_patches)
            Assert.eq(self._use_audio, use_audio)
            Assert.eq(self._use_grpo_data, use_grpo_data)
        else:
            self._use_loss_masking_spans = use_loss_masking_spans
            self._use_preference_spans = use_preference_spans
            self._use_image_patches = use_image_patches
            self._use_audio = use_audio
            self._use_grpo_data = use_grpo_data

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

        # Write audio clips
        if use_audio:
            self._audio_writer.write(document.audio)

        if use_grpo_data:
            assert document.advantages is not None
            assert document.old_log_probabilities is not None
            self._advantages_writer.write(document.advantages)
            self._old_log_probabilities_writer.write(document.old_log_probabilities)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._token_writer.__exit__(exc_type, exc_val, exc_tb)
        self._loss_masking_span_writer.__exit__(exc_type, exc_val, exc_tb)
        self._chosen_spans_writer.__exit__(exc_type, exc_val, exc_tb)
        self._rejected_spans_writer.__exit__(exc_type, exc_val, exc_tb)
        self._image_patches_writer.__exit__(exc_type, exc_val, exc_tb)
        self._audio_writer.__exit__(exc_type, exc_val, exc_tb)
        self._advantages_writer.__exit__(exc_type, exc_val, exc_tb)
        self._old_log_probabilities_writer.__exit__(exc_type, exc_val, exc_tb)

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
            if self._use_audio:
                _copy_chunked(
                    self._path.joinpath("audio"),
                    self._stream,
                    config.audio.begin,
                    config.audio.end,
                )
            if self._use_grpo_data:
                _copy_chunked(
                    self._path.joinpath("advantages"),
                    self._stream,
                    config.advantages.begin,
                    config.advantages.end,
                )
                _copy_chunked(
                    self._path.joinpath("old_log_probabilities"),
                    self._stream,
                    config.old_log_probabilities.begin,
                    config.old_log_probabilities.end,
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
        if self._use_audio:
            audio = self._audio_writer.get_config(offset)
            offset = audio.end
        else:
            audio = NullReaderConfig()
        if self._use_grpo_data:
            advantages = self._advantages_writer.get_config(offset)
            offset = advantages.end
            old_log_probabilities = self._old_log_probabilities_writer.get_config(offset)
            offset = old_log_probabilities.end
        else:
            advantages = NullReaderConfig()
            old_log_probabilities = NullReaderConfig()
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
            audio=audio,
            advantages=advantages,
            old_log_probabilities=old_log_probabilities,
        )


def _copy_chunked(path: pathlib.Path, stream: io.BufferedWriter, expected_begin: int, expected_end: int):
    # Copy temporary file content in chunks of 100 MB.
    Assert.eq(stream.tell(), expected_begin)
    with path.open("rb") as input_stream:
        while data := input_stream.read(100000000):
            stream.write(data)
    Assert.eq(stream.tell(), expected_end)
