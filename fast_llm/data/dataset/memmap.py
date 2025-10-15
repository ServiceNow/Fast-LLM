import json
import pathlib
import typing

import numpy as np
import torch

from fast_llm.data.dataset.indexed import IndexedDataset
from fast_llm.data.sample.abstract import MemmapReader, Sample
from fast_llm.data.sample.config import MemmapIndexDatasetReaderConfig

FILE_HEADER = b"fast_llm_prepared_dataset"


class MemmapDataset(IndexedDataset):
    """
    A memory map dataset, which handles lazy loading of a pre-processed dataset.
    """

    def __init__(
        self,
        name: str,
        path: pathlib.Path | str,
    ):
        self._init(name, path)

    def _init(self, name: str, path: pathlib.Path | str) -> None:
        super().__init__()
        self._name = name
        self._path = path

        with self._path.open("rb") as stream:
            # Very file type.
            assert stream.read(len(FILE_HEADER)) == FILE_HEADER
            # Go to reader configs.
            stream.seek(int.from_bytes(stream.read(4), signed=False))
            # Read the reader config.
            reader_config = MemmapIndexDatasetReaderConfig.from_dict(
                json.loads(stream.read(int.from_bytes(stream.read(4), signed=False)).decode("utf-8"))
            )

        self._memmap = np.memmap(self._path, mode="r")
        # TODO: ===== Check num_documents, num_tokens ======
        self._reader = reader_config.get_reader(memoryview(self._memmap))

    def __getstate__(self) -> tuple[str, pathlib.Path]:
        return (self._name, self._path)

    def __setstate__(self, state: tuple[str, pathlib.Path]):
        self._init(*state)

    def __del__(self):
        if hasattr(self, "_memmap"):
            self._memmap._mmap.close()  # noqa
            del self._memmap

    def get(
        self,
        index: int,
        begin: int,
        end: int,
    ) -> Sample:
        return self._reader.get(index, begin, end)

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return self._reader

    # TODO: ====== needed? ======
    # @property
    # def num_tokens(self) -> int:
    #    return self._reader.num_tokens

    def get_document_sizes(self) -> torch.Tensor:
        return self._reader.get_document_sizes()

    def get_document_size(self, index: int) -> int:
        return self._reader.get_document_size(index)

    @classmethod
    def write_dataset(cls, path: pathlib.Path, documents: typing.Iterable[Sample], reader_class: type[MemmapReader]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            # Write the file type header.
            stream.write(FILE_HEADER)
            # Leave space for a pointer to the reader config.
            # We write the config  at the end since we don't know it yet.
            start = stream.tell()
            stream.seek(start + 4)
            # Write the data.
            reader_config = reader_class.write(documents, stream)
            # Write the reader config.
            config_offset = stream.tell()
            reader_config_bytes = json.dumps(reader_config.to_dict()).encode("utf-8")
            stream.write(len(reader_config_bytes).to_bytes(4, signed=False))
            stream.write(reader_config_bytes)
            # Write a pointer to the reader config.
            stream.seek(start)
            stream.write(config_offset.to_bytes(4, signed=False))
