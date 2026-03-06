import functools
import io
import logging
import math
import pathlib
import typing

from fast_llm.config import Config, Field, FieldHint, config_class
from fast_llm.data.dataset.config import IndexedDatasetConfig, SampledDatasetConfig
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert, get_unique

if typing.TYPE_CHECKING:
    pass

    from fast_llm.data.dataset.indexed import IndexedDataset
    from fast_llm.data.dataset.memmap.abstract import (
        MemmapIndexedDatasetReader,
        MemmapReader,
        MemmapWriter,
        NullMemmapReader,
    )
    from fast_llm.data.dataset.memmap.language_model import LanguageModelReader, LanguageModelWriter
    from fast_llm.data.dataset.memmap.patch import PatchReader, PatchWriter
    from fast_llm.data.dataset.memmap.range import RangeReader, RangeWriter
    from fast_llm.data.dataset.memmap.token import TokenReader, TokenWriter
    from fast_llm.data.document.abstract import Document

logger = logging.getLogger(__name__)


@config_class(dynamic_type={SampledDatasetConfig: "memmap"})
class MemmapDatasetConfig[DocumentType: Document](IndexedDatasetConfig[DocumentType]):
    _abstract: typing.ClassVar[bool] = False
    path: pathlib.Path = Field(
        default=None,
        desc="The path to the dataset, excluding the `.bin` or `.idx` suffix.",
        hint=FieldHint.core,
    )

    def build(self) -> "IndexedDataset[DocumentType]":
        name = str(self.path).replace("/", "__")
        if self.path.is_file():
            from fast_llm.data.dataset.memmap.memmap import MemmapDataset

            return MemmapDataset[DocumentType](name, self.path)
        elif self.path.with_suffix(".bin").is_file() and self.path.with_suffix(".idx").is_file():
            logger.warning(
                "Using the legacy memmap dataset format."
                " This format is deprecated and will be removed in a future release."
                " Please recreate the dataset in the new memmap format."
            )
            from fast_llm.data.dataset.gpt.legacy_memmap import LegacyMemmapDataset

            return LegacyMemmapDataset[DocumentType](name, self.path)
        else:
            raise FileNotFoundError(self.path)


@config_class(registry=True)
class MemmapReaderBaseConfig(Config):
    """
    Configuration for a memmap reader or reader-like object.
    Note: `MemmapDataset` requires a `MemmapIndexedDatasetReader`.
      Other readers need to be nested within a `MemmapIndexedDatasetReader`
    Note: Reader configs are not typical configs, and do not need to be located in a separate `config.py` file.
    """

    _abstract = True

    @classmethod
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        if cls is MemmapReaderBaseConfig and cls.get_subclass(default.get("type")) is None:
            # Default subclass, necessary for loading configs where some components could be absent.
            return NullReaderConfig._from_dict(default, strict)
        return super()._from_dict(default, strict=strict)

    def get_reader(self, buffer: memoryview) -> "MemmapReader|None":
        raise NotImplementedError()

    @property
    def expected_buffer_size(self) -> int:
        """
        The expected buffer size in bytes, including header and footer. Used for self-validation.
        """
        raise NotImplementedError()

    def get_metadata(self) -> dict[str, typing.Any]:
        raise NotImplementedError()

    @classmethod
    def blend_metadata(cls, metadata: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
        raise NotImplementedError()


@config_class(dynamic_type={MemmapReaderBaseConfig: "none"})
class NullReaderConfig(MemmapReaderBaseConfig):
    """
    Configuration for a dynamically disabled reader.
    """

    _abstract = False

    def get_reader(self, buffer: memoryview) -> "NullMemmapReader":
        from fast_llm.data.dataset.memmap.abstract import NullMemmapReader

        return NullMemmapReader(self)

    @property
    def expected_buffer_size(self) -> int:
        return 0


@config_class(registry=True)
class MemmapReaderConfig(MemmapReaderBaseConfig):
    """
    Configuration for a standard memmap reader.
    """

    # Data location in the file.
    begin: int = Field()
    end: int = Field()
    # Constant strings for alignment safety.
    header: typing.ClassVar[bytes]
    footer: typing.ClassVar[bytes]

    @property
    def reader_class(self) -> "type[MemmapReader]":
        raise NotImplementedError()

    def get_reader(self, buffer: memoryview) -> "MemmapReader":
        return self.reader_class(self, buffer)

    @property
    def expected_buffer_size(self) -> int:
        """
        The expected buffer size in bytes, including header and footer. Used for self-validation.
        """
        return self._expected_buffer_size + len(self.header) + len(self.footer)

    @property
    def _expected_buffer_size(self) -> int:
        """
        The expected buffer size in bytes, excluding header and footer. Used for self-validation.
        """
        raise NotImplementedError()

    @property
    def writer_class(self) -> "type[MemmapWriter]":
        raise NotImplementedError()

    def get_writer(self, stream: io.BufferedWriter) -> "MemmapWriter":
        return self.writer_class(stream)

    def _validate(self):
        super()._validate()
        Assert.eq(self.end - self.begin, self.expected_buffer_size)


@config_class()
class PatchReaderBaseConfig(MemmapReaderBaseConfig):
    _abstract = False
    patch_shape: tuple[int, ...] = Field()
    data_type: DataType = Field()

    @property
    def patch_size(self) -> int:
        return math.prod(self.patch_shape)

    @property
    def grid_dims(self) -> int:
        return len(self.patch_shape) - 1


@config_class(dynamic_type={MemmapReaderBaseConfig: "patch"})
class PatchReaderConfig(PatchReaderBaseConfig, MemmapReaderConfig):
    header: typing.ClassVar[bytes] = b"patch begin"
    footer: typing.ClassVar[bytes] = b"patch end"
    num_documents: int = Field()
    num_patches: int = Field()
    num_patch_groups: int = Field()

    def __len__(self) -> int:
        return self.num_documents

    @property
    def reader_class(self) -> "type[PatchReader]":
        from fast_llm.data.dataset.memmap.patch import PatchReader

        return PatchReader

    @property
    def writer_class(self) -> "type[PatchWriter]":
        from fast_llm.data.dataset.memmap.patch import PatchWriter

        return PatchWriter

    @property
    def _expected_buffer_size(self) -> int:
        import torch

        return (
            self.num_patches * self.patch_size * self.data_type.torch.itemsize
            + ((1 + self.grid_dims) * self.num_patches + self.num_patch_groups + 2 * self.num_documents + 2)
            * torch.int32.itemsize
        )

    def get_metadata(self) -> dict[str, typing.Any]:
        return {
            "num_documents": self.num_documents,
            "num_patches": self.num_patches,
            "num_patch_groups": self.num_patch_groups,
            "num_pixels": self.patch_size * self.num_patches,
            "patch_shape": self.patch_shape,
            "data_type": str(self.data_type),
        }

    @classmethod
    def blend_metadata(cls, metadata: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
        return {
            "num_documents": sum(metadata_["num_documents"] for metadata_ in metadata),
            "num_patches": sum(metadata_["num_patches"] for metadata_ in metadata),
            "num_patch_groups": sum(metadata_["num_patch_groups"] for metadata_ in metadata),
            "num_pixels": sum(metadata_["num_pixels"] for metadata_ in metadata),
            "patch_shape": get_unique(metadata_["patch_shape"] for metadata_ in metadata),
            "data_type": get_unique(metadata_["data_type"] for metadata_ in metadata),
        }


@config_class()
class RangeReaderBaseConfig(MemmapReaderBaseConfig):
    _abstract = False


@config_class(dynamic_type={MemmapReaderBaseConfig: "range"})
class RangeReaderConfig(RangeReaderBaseConfig, MemmapReaderConfig):
    header: typing.ClassVar[bytes] = b"range begin"
    footer: typing.ClassVar[bytes] = b"range end"
    num_documents: int = Field()
    num_ranges: int = Field()

    @property
    def reader_class(self) -> "type[RangeReader]":
        from fast_llm.data.dataset.memmap.range import RangeReader

        return RangeReader

    @property
    def writer_class(self) -> "type[RangeWriter]":
        from fast_llm.data.dataset.memmap.range import RangeWriter

        return RangeWriter

    @property
    def _expected_buffer_size(self) -> int:
        import torch

        return self.num_ranges * torch.int32.itemsize * 2 + (self.num_documents + 1) * torch.int32.itemsize

    def get_metadata(self) -> dict[str, typing.Any]:
        return {
            "num_documents": self.num_documents,
            "num_ranges": self.num_ranges,
        }

    @classmethod
    def blend_metadata(cls, metadata: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
        return {
            "num_documents": sum(metadata_["num_documents"] for metadata_ in metadata),
            "num_ranges": sum(metadata_["num_ranges"] for metadata_ in metadata),
        }


@config_class(dynamic_type={MemmapReaderBaseConfig: "token"})
class TokenReaderConfig(MemmapReaderConfig):
    _abstract = False
    header: typing.ClassVar[bytes] = b"token begin"
    footer: typing.ClassVar[bytes] = b"token end"
    num_documents: int = Field()
    num_tokens: int = Field()
    data_type: DataType = Field()

    def __len__(self) -> int:
        return self.num_documents

    @property
    def reader_class(self) -> "type[TokenReader]":
        from fast_llm.data.dataset.memmap.token import TokenReader

        return TokenReader

    @property
    def writer_class(self) -> "type[TokenWriter]":
        from fast_llm.data.dataset.memmap.token import TokenWriter

        return TokenWriter

    @property
    def _expected_buffer_size(self) -> int:
        import torch

        return self.num_tokens * self.data_type.torch.itemsize + (self.num_documents + 1) * torch.int64.itemsize

    def get_metadata(self) -> dict[str, typing.Any]:
        return {
            "num_tokens": self.num_tokens,
            "num_documents": self.num_documents,
            "data_type": str(self.data_type),
        }

    @classmethod
    def blend_metadata(cls, metadata: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
        return {
            "num_tokens": sum(metadata_["num_tokens"] for metadata_ in metadata),
            "num_documents": sum(metadata_["num_documents"] for metadata_ in metadata),
            "data_type": get_unique(metadata_["data_type"] for metadata_ in metadata),
        }


@config_class()
class MemmapIndexDatasetReaderConfig(MemmapReaderConfig):
    """
    Configuration for a standard memmap reader matching the indexed dataset interface, i.e.,
    consisting of a list of documents of known lengths.
    """

    def __len__(self) -> int:
        raise NotImplementedError()

    @property
    def num_tokens(self) -> int:
        raise NotImplementedError()

    @property
    def reader_class(self) -> "type[MemmapIndexedDatasetReader]":
        raise NotImplementedError()

    def get_reader(self, buffer: memoryview) -> "MemmapIndexedDatasetReader":
        return self.reader_class(self, buffer)

    def get_metadata(self) -> dict[str, typing.Any]:
        return {"num_tokens": self.num_tokens}

    @classmethod
    def blend_metadata(cls, metadata: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
        return {"num_tokens": sum(metadata_["num_tokens"] for metadata_ in metadata)}


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
        if self.has_image_patches:
            Assert.eq(len(self.patch_shape), 3)
            Assert.eq(self.image_patches.data_type, DataType.uint8)

    def __len__(self) -> int:
        return len(self.tokens)

    @functools.cached_property
    def has_loss_masking_spans(self) -> bool:
        return isinstance(self.loss_masking_spans, RangeReaderConfig)

    @functools.cached_property
    def has_preference_spans(self) -> bool:
        return isinstance(self.chosen_spans, RangeReaderConfig)

    @functools.cached_property
    def has_image_patches(self) -> bool:
        return isinstance(self.image_patches, PatchReaderConfig)

    @functools.cached_property
    def patch_shape(self) -> tuple[int, int, int]:
        assert self.has_image_patches
        return self.image_patches.patch_shape

    @property
    def num_tokens(self) -> int:
        return self.tokens.num_tokens

    @property
    def reader_class(self) -> "type[LanguageModelReader]":
        from fast_llm.data.dataset.memmap.language_model import LanguageModelReader

        return LanguageModelReader

    @property
    def writer_class(self) -> "type[LanguageModelWriter]":
        from fast_llm.data.dataset.memmap.language_model import LanguageModelWriter

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

    def get_metadata(self) -> dict[str, typing.Any]:
        out = super().get_metadata()
        out["tokens"] = self.tokens.get_metadata()
        if not isinstance(self.loss_masking_spans, NullReaderConfig):
            out["loss_masking_spans"] = self.loss_masking_spans.get_metadata()
        if not isinstance(self.chosen_spans, NullReaderConfig):
            out["chosen_spans"] = self.chosen_spans.get_metadata()
        if not isinstance(self.rejected_spans, NullReaderConfig):
            out["rejected_spans"] = self.rejected_spans.get_metadata()
        if not isinstance(self.image_patches, NullReaderConfig):
            out["image_patches"] = self.image_patches.get_metadata()
        return out

    @classmethod
    def blend_metadata(cls, metadata: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
        out = super().blend_metadata(metadata)
        out["tokens"] = TokenReaderConfig.blend_metadata([metadata_["tokens"] for metadata_ in metadata])
        if "loss_masking_spans" in metadata[0]:
            out["loss_masking_spans"] = RangeReaderConfig.blend_metadata(
                [metadata_["loss_masking_spans"] for metadata_ in metadata]
            )
        if "chosen_spans" in metadata[0]:
            out["chosen_spans"] = RangeReaderConfig.blend_metadata(
                [metadata_["chosen_spans"] for metadata_ in metadata]
            )
        if "rejected_spans" in metadata[0]:
            out["image_patches"] = RangeReaderConfig.blend_metadata(
                [metadata_["image_patches"] for metadata_ in metadata]
            )
        if "image_patches" in metadata[0]:
            out["image_patches"] = PatchReaderConfig.blend_metadata(
                [metadata_["image_patches"] for metadata_ in metadata]
            )
        return out
