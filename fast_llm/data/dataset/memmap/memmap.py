import json
import pathlib
import typing

import numpy as np
import torch

from fast_llm.data.dataset.indexed import IndexedDataset
from fast_llm.data.dataset.memmap.abstract import MemmapIndexedDatasetReader, MemmapWriter
from fast_llm.data.dataset.memmap.config import MemmapIndexDatasetReaderConfig
from fast_llm.data.document.abstract import (
    Document,
)

FILE_HEADER = b"fast_llm_prepared_dataset"


class MemmapDataset[DocumentType: Document](IndexedDataset[DocumentType]):
    """
    A memory map dataset, which handles lazy loading of a pre-processed dataset.
    """

    def __init__(self, name: str, path: pathlib.Path | str):
        self._init(name, path)

    def _init(self, name: str, path: pathlib.Path | str) -> None:
        super().__init__()
        self._name = name
        self._path = path

        path = pathlib.Path(path) if isinstance(path, str) else path
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            assert stream.read(len(FILE_HEADER)) == FILE_HEADER, f"Invalid file header in {path}."
            config_offset = int.from_bytes(stream.read(8), signed=False)
            assert config_offset + 4 <= file_size, f"Config offset {config_offset} out of range for {path}."
            stream.seek(config_offset)
            config_size = int.from_bytes(stream.read(4), signed=False)
            assert config_offset + 4 + config_size <= file_size, f"Config size {config_size} out of range for {path}."
            config_bytes = stream.read(config_size)
            reader_config = MemmapIndexDatasetReaderConfig.from_dict(json.loads(config_bytes.decode("utf-8")))

        self._memmap = np.memmap(self._path, mode="c")
        self._reader = reader_config.get_reader(memoryview(self._memmap))

    def __getstate__(self) -> tuple[str, pathlib.Path]:
        # We pass the reader config to force its import in data loader workers.
        return self._name, self._path

    def __setstate__(self, state: tuple[str, pathlib.Path]):
        self._init(*state)

    def __del__(self):
        if hasattr(self, "_memmap"):
            self._memmap._mmap.close()  # noqa
            del self._memmap

    def get_document(self, index: int, begin: int = 0, end: int | None = None) -> DocumentType:
        if end is None:
            end = self._reader.get_document_size(index)
        return self._reader.get_document(index, begin, end)

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._reader)

    @property
    def num_tokens(self) -> int:
        return self._reader.num_tokens

    def get_document_sizes(self) -> torch.Tensor:
        return self._reader.get_document_sizes()

    def get_document_size(self, index: int) -> int:
        return self._reader.get_document_size(index)

    @property
    def reader(self) -> MemmapIndexedDatasetReader:
        return self._reader

    @classmethod
    def write_dataset(
        cls,
        path: pathlib.Path,
        documents: typing.Iterable[Document],
        writer_class: type[MemmapWriter],
    ) -> MemmapIndexDatasetReaderConfig:
        # TODO: Match `writer_class` with `DocumentType`?
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            # Write the file type header.
            stream.write(FILE_HEADER)
            # Leave space for a pointer to the reader config.
            # We write the config  at the end since we don't know it yet.
            start = stream.tell()
            stream.seek(start + 8)
            # Write the data.
            reader_config = writer_class.write_dataset(stream, documents)
            # Write the reader config.
            config_offset = stream.tell()
            reader_config_bytes = json.dumps(reader_config.to_dict()).encode("utf-8")
            stream.write(len(reader_config_bytes).to_bytes(4, signed=False))
            stream.write(reader_config_bytes)
            # Write a pointer to the reader config.
            stream.seek(start)
            stream.write(config_offset.to_bytes(8, signed=False))
        return reader_config
