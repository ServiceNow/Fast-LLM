import io
import logging
import pathlib
import tempfile
import typing
import warnings

import torch

from fast_llm.config import Field, config_class
from fast_llm.data.preprocessing.abstract import NullPreprocessingConfig
from fast_llm.data.preprocessing.image_patch import ImageNormalizationConfig, ImagePatchConfig
from fast_llm.data.preprocessing.language_model import LanguageModelPreprocessingConfig
from fast_llm.data.sample.abstract import (
    Batch,
    MemmapIndexDatasetReaderConfig,
    MemmapIndexedDatasetReader,
    MemmapReaderBaseConfig,
    MemmapWriter,
    NullReaderConfig,
    Sample,
)
from fast_llm.data.sample.patch import EmptyPatchReader, PatchBatch, PatchReaderConfig, PatchSample, PatchWriter
from fast_llm.data.sample.range import EmptyRangeReader, RangeBatch, RangeReaderConfig, RangeSample, RangeWriter
from fast_llm.data.sample.token import TokenBatch, TokenReaderConfig, TokenSample, TokenWriter
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class LanguageModelSample(Sample):
    def __init__(
        self,
        tokens: TokenSample,
        loss_masking_spans: RangeSample | None = None,
        chosen_spans: RangeSample | None = None,
        rejected_spans: RangeSample | None = None,
        image_patches: PatchSample | None = None,
    ):
        self.tokens = tokens
        self.loss_masking_spans = loss_masking_spans
        self.chosen_spans = chosen_spans
        self.rejected_spans = rejected_spans
        self.image_patches = image_patches

    @classmethod
    def from_documents(cls, documents: typing.Iterable[typing.Self]) -> typing.Self:
        return cls(
            TokenSample.from_documents([document.tokens for document in documents]),
            _merge_optional(RangeSample.from_documents, [document.loss_masking_spans for document in documents]),
            _merge_optional(RangeSample.from_documents, [document.chosen_spans for document in documents]),
            _merge_optional(RangeSample.from_documents, [document.rejected_spans for document in documents]),
            _merge_optional(PatchSample.from_documents, [document.image_patches for document in documents]),
        )

    def crop(self, begin: int, end: int) -> typing.Self:
        return self.__class__(
            self.tokens.crop(begin, end),
            _crop_optional(self.loss_masking_spans, begin, end),
            _crop_optional(self.chosen_spans, begin, end),
            _crop_optional(self.rejected_spans, begin, end),
            _crop_optional(self.image_patches, begin, end),
        )

    def __len__(self) -> int:
        return len(self.tokens)

    def get_padding(self, size: int) -> typing.Self:
        return LanguageModelSample(
            self.tokens.get_padding(size),
            None if self.loss_masking_spans is None else self.loss_masking_spans.get_padding(size),
            None if self.chosen_spans is None else self.chosen_spans.get_padding(size),
            None if self.rejected_spans is None else self.rejected_spans.get_padding(size),
            None if self.image_patches is None else self.image_patches.get_padding(size),
        )


class LanguageModelBatch(Batch):
    def __init__(
        self,
        tokens: TokenBatch,
        loss_masking_spans: RangeBatch | None = None,
        chosen_spans: RangeBatch | None = None,
        rejected_spans: RangeBatch | None = None,
        image_patches: PatchBatch | None = None,
    ):
        self.tokens = tokens
        self.loss_masking_spans = loss_masking_spans
        self.chosen_spans = chosen_spans
        self.rejected_spans = rejected_spans
        self.image_patches = image_patches

    @classmethod
    def from_samples(cls, samples: typing.Iterable[LanguageModelSample]) -> typing.Self:
        return cls(
            TokenBatch.from_samples([sample.tokens for sample in samples]),
            _merge_optional(RangeBatch.from_samples, [sample.loss_masking_spans for sample in samples]),
            _merge_optional(RangeBatch.from_samples, [sample.chosen_spans for sample in samples]),
            _merge_optional(RangeBatch.from_samples, [sample.rejected_spans for sample in samples]),
            _merge_optional(PatchBatch.from_samples, [sample.image_patches for sample in samples]),
        )

    def crop(self, begin: int, end: int) -> typing.Self:
        return self.__class__(
            self.tokens.crop(begin, end),
            _crop_optional(self.loss_masking_spans, begin, end),
            _crop_optional(self.chosen_spans, begin, end),
            _crop_optional(self.rejected_spans, begin, end),
            _crop_optional(self.image_patches, begin, end),
        )

    def to_device_(self, device: "torch.device | str"):
        self.tokens.to_device_(device)
        if self.loss_masking_spans is not None:
            self.loss_masking_spans.to_device_(device)
        if self.chosen_spans is not None:
            self.chosen_spans.to_device_(device)
        if self.rejected_spans is not None:
            self.rejected_spans.to_device_(device)
        if self.image_patches is not None:
            self.image_patches.to_device_(device)


def _merge_optional[T](fn: typing.Callable[[typing.Iterable], T], args: typing.Iterable) -> T | None:
    return None if any(arg is None for arg in args) else fn(args)


def _crop_optional[T: Sample | Batch](sample_or_batch: T, begin: int, end: int) -> T | None:
    return None if sample_or_batch is None else sample_or_batch.crop(begin, end)


@config_class(dynamic_type={MemmapReaderBaseConfig: "language_model"})
class LanguageModelReaderConfig(MemmapIndexDatasetReaderConfig):
    _abstract = False
    header: typing.ClassVar[bytes] = b"lm begin"
    footer: typing.ClassVar[bytes] = b"lm end"
    tokens: TokenReaderConfig = Field()
    # Using dynamic type for optional readers for enabling/disabling
    loss_masking_spans: MemmapReaderBaseConfig = Field()
    chosen_spans: MemmapReaderBaseConfig = Field()
    rejected_spans: MemmapReaderBaseConfig = Field()
    image_patches: MemmapReaderBaseConfig = Field()

    def _validate(self) -> None:
        super()._validate()
        if isinstance(self.preprocessing, NullPreprocessingConfig):
            # Address missing config, mostly for backward compatibility.
            # TODO: We can't tell which dataset this comes from.
            logger.warning(
                f"Preprocessing configuration not specified for dataset reader, generating partial configuration from known parameters."
            )
            if isinstance(self.image_patches, PatchReaderConfig):
                Assert.eq(len(patch_shape := self.image_patches.patch_shape), 3)
                image_patches = ImagePatchConfig(height=patch_shape[1], width=patch_shape[2])
            else:
                image_patches = NullPreprocessingConfig()
            self.preprocessing = LanguageModelPreprocessingConfig(
                vocab_size=0,
                image_patches=image_patches,
                use_loss_masking_spans=isinstance(self.loss_masking_spans, RangeReaderConfig),
                use_preference_spans=isinstance(self.chosen_spans, RangeReaderConfig),
            )
        # TODO: Avoid duplicated information.
        Assert.custom(
            isinstance,
            self.loss_masking_spans,
            RangeReaderConfig if self.preprocessing.use_loss_masking_spans else NullReaderConfig,
        )
        Assert.custom(
            isinstance,
            self.chosen_spans,
            RangeReaderConfig if self.preprocessing.use_preference_spans else NullReaderConfig,
        )
        Assert.custom(
            isinstance,
            self.rejected_spans,
            RangeReaderConfig if self.preprocessing.use_preference_spans else NullReaderConfig,
        )
        if self.preprocessing.use_image_patches:
            Assert.custom(isinstance, self.image_patches, PatchReaderConfig)
            Assert.eq(self.image_patches.patch_shape, self.preprocessing.image_patches.patch_shape)
            Assert.eq(self.image_patches.data_type, DataType.uint8)
        else:
            Assert.custom(isinstance, self.image_patches, NullReaderConfig)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def num_tokens(self) -> int:
        return self.tokens.num_tokens

    @property
    def reader_class(self) -> "type[LanguageModelReader]":
        return LanguageModelReader

    @property
    def writer_class(self) -> "type[LanguageModelWriter]":
        return LanguageModelWriter

    @property
    def _expected_buffer_size(self) -> int:
        return (
            self.tokens.expected_buffer_size
            + self.loss_masking_spans.expected_buffer_size
            + self.chosen_spans.expected_buffer_size
            + self.rejected_spans.expected_buffer_size
            + self.image_patches.expected_buffer_size
        )


class LanguageModelReader[ConfigType: LanguageModelReaderConfig](MemmapIndexedDatasetReader[ConfigType]):
    _model_preprocessing: LanguageModelPreprocessingConfig

    def __init__(
        self,
        config: ConfigType,
        buffer: memoryview,
        model_preprocessing: LanguageModelPreprocessingConfig | None = None,
    ):
        super().__init__(config, buffer, model_preprocessing)
        self._config.preprocessing.check_compatibility(self._model_preprocessing)
        # Using `buffer` and not `self._buffer` because nested offsets (`begin`, `end`) are global.
        self._tokens = self._config.tokens.get_reader(buffer)

        if self._model_preprocessing.use_loss_masking_spans:
            if isinstance(self._config.loss_masking_spans, NullReaderConfig):
                # TODO: We can't tell which dataset this comes from.
                warnings.warn(
                    f"The model uses loss masking spans, but the dataset does not specify any."
                    " Assuming empty span lists."
                )
                self._loss_masking_spans = EmptyRangeReader(
                    RangeReaderConfig(begin=0, end=0, num_documents=0, num_ranges=0), buffer
                )
            else:
                self._loss_masking_spans = self._config.loss_masking_spans.get_reader(buffer)

        if self._model_preprocessing.use_preference_spans:
            self._chosen_spans = self._config.chosen_spans.get_reader(buffer)
            self._rejected_spans = self._config.rejected_spans.get_reader(buffer)

        if self._model_preprocessing.use_image_patches:
            model_image_preprocessing: ImagePatchConfig = self._model_preprocessing.image_patches
            if isinstance(self._config.image_patches, NullReaderConfig):
                warnings.warn(
                    f"The model uses image patches, but the dataset does not specify any."
                    " Assuming empty patch lists."
                )
                self._image_patches = EmptyPatchReader(
                    PatchReaderConfig(
                        begin=0,
                        end=0,
                        num_documents=0,
                        num_patches=0,
                        num_patch_groups=0,
                        patch_shape=model_image_preprocessing.patch_shape,
                        data_type=DataType.uint8,
                    ),
                    buffer,
                )
            else:
                self._image_patches = self._config.image_patches.get_reader(buffer)

            # TODO: Make this configurable. (Add to `model_preprocessing`?)
            self._image_normalization_config = ImageNormalizationConfig()

    @property
    def num_tokens(self) -> int:
        return self._config.tokens.num_tokens

    def get_document(self, index: int, begin: int, end: int) -> Sample:
        if self._model_preprocessing.use_image_patches:
            image_patches = self._image_patches.get_document(index, begin, end)
            image_patches.patches = self._image_normalization_config.normalize(image_patches.patches)
        else:
            image_patches = None
        return LanguageModelSample(
            self._tokens.get_document(index, begin, end),
            (
                self._loss_masking_spans.get_document(index, begin, end)
                if self._model_preprocessing.use_loss_masking_spans
                else None
            ),
            (
                self._chosen_spans.get_document(index, begin, end)
                if self._model_preprocessing.use_preference_spans
                else None
            ),
            (
                self._rejected_spans.get_document(index, begin, end)
                if self._model_preprocessing.use_preference_spans
                else None
            ),
            image_patches,
        )

    def get_document_sizes(self) -> torch.Tensor:
        return self._tokens.get_document_sizes()

    def get_document_size(self, index: int) -> int:
        return self._tokens.get_document_size(index)


class LanguageModelWriter(MemmapWriter):
    _preprocessing_config: LanguageModelPreprocessingConfig

    def __enter__(self):
        super().__enter__()
        self._size_cumsum = [0]
        self._data_type = None

        self._directory = tempfile.TemporaryDirectory()
        self._path = pathlib.Path(self._directory.name)
        # We write intermediate results in separate files so we don't need to iterate over the dataset multiple times.
        self._token_writer = TokenWriter(self._path.joinpath("tokens")).__enter__()
        if self._preprocessing_config.use_loss_masking_spans:
            self._loss_masking_span_writer = RangeWriter(self._path.joinpath("loss_masking_spans")).__enter__()
        if self._preprocessing_config.use_preference_spans:
            self._chosen_spans_writer = RangeWriter(self._path.joinpath("chosen_spans")).__enter__()
            self._rejected_spans_writer = RangeWriter(self._path.joinpath("rejected_spans")).__enter__()
        if self._preprocessing_config.use_image_patches:
            self._image_patches_writer = PatchWriter(self._path.joinpath("image_patches")).__enter__()
        return self

    def write(self, document: LanguageModelSample):
        super().write(document)
        # Write tokens.
        self._token_writer.write(document.tokens)

        # Write loss masking spans.
        if self._preprocessing_config.use_loss_masking_spans:
            assert document.loss_masking_spans is not None
            self._loss_masking_span_writer.write(document.loss_masking_spans)

        # Write preference spans.
        if self._preprocessing_config.use_preference_spans:
            assert document.chosen_spans is not None
            assert document.rejected_spans is not None
            self._chosen_spans_writer.write(document.chosen_spans)
            self._rejected_spans_writer.write(document.rejected_spans)

        # Write image patches
        if self._preprocessing_config.use_image_patches:
            assert document.image_patches is not None
            self._image_patches_writer.write(document.image_patches)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._token_writer.__exit__(exc_type, exc_val, exc_tb)
        if self._preprocessing_config.use_loss_masking_spans:
            self._loss_masking_span_writer.__exit__(exc_type, exc_val, exc_tb)
        if self._preprocessing_config.use_preference_spans:
            self._chosen_spans_writer.__exit__(exc_type, exc_val, exc_tb)
            self._rejected_spans_writer.__exit__(exc_type, exc_val, exc_tb)
        if self._preprocessing_config.use_image_patches:
            self._image_patches_writer.__exit__(exc_type, exc_val, exc_tb)

        if exc_type is None:
            # A dummy config so we can verify the begin and end offsets.
            config = self._get_config(self._begin, None)
            _copy_chunked(self._path.joinpath("tokens"), self._stream, config.tokens.begin, config.tokens.end)

            if self._preprocessing_config.use_loss_masking_spans:
                _copy_chunked(
                    self._path.joinpath("loss_masking_spans"),
                    self._stream,
                    config.loss_masking_spans.begin,
                    config.loss_masking_spans.end,
                )
            if self._preprocessing_config.use_preference_spans:
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

            if self._preprocessing_config.use_image_patches:
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
        if self._preprocessing_config.use_loss_masking_spans:
            loss_masking_spans = self._loss_masking_span_writer.get_config(offset)
            offset = loss_masking_spans.end
        else:
            loss_masking_spans = NullReaderConfig()
        if self._preprocessing_config.use_preference_spans:
            chosen_spans = self._chosen_spans_writer.get_config(offset)
            offset = chosen_spans.end
            rejected_spans = self._rejected_spans_writer.get_config(offset)
            offset = rejected_spans.end
        else:
            chosen_spans = NullReaderConfig()
            rejected_spans = NullReaderConfig()
        if self._preprocessing_config.use_image_patches:
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
            preprocessing=self._preprocessing_config,
        )


def _copy_chunked(path: pathlib.Path, stream: io.BufferedWriter, expected_begin: int, expected_end: int):
    # Copy temporary file content in chunks of 100 MB.
    Assert.eq(stream.tell(), expected_begin)
    with path.open("rb") as input_stream:
        while data := input_stream.read(100000000):
            stream.write(data)
    Assert.eq(stream.tell(), expected_end)
