import json
import pathlib
import typing

import numpy as np
import torch

from fast_llm.data.dataset.config import SamplingParameters
from fast_llm.data.dataset.indexed import IndexedDataset
from fast_llm.data.preprocessing.abstract import PreprocessingConfig
from fast_llm.data.sample.abstract import MemmapIndexDatasetReaderConfig, MemmapWriter, Sample

FILE_HEADER = b"fast_llm_prepared_dataset"


class MemmapDataset[SampleType: Sample](IndexedDataset[SampleType]):
    """
    A memory map dataset, which handles lazy loading of a pre-processed dataset.
    """

    def __init__(
        self,
        name: str,
        path: pathlib.Path | str,
        preprocessing: PreprocessingConfig,
    ):
        self._init(name, path, preprocessing)

    def _init(self, name: str, path: pathlib.Path | str, preprocessing: PreprocessingConfig) -> None:
        super().__init__()
        self._name = name
        self._path = path
        self._preprocessing = preprocessing

        with self._path.open("rb") as stream:
            # Very file type.
            assert stream.read(len(FILE_HEADER)) == FILE_HEADER
            # Go to reader configs.
            stream.seek(int.from_bytes(stream.read(8), signed=False))
            # Read the reader config.
            reader_config = MemmapIndexDatasetReaderConfig.from_dict(
                json.loads(stream.read(int.from_bytes(stream.read(4), signed=False)).decode("utf-8"))
            )

        reader_config.preprocessing.check_compatibility(self._preprocessing)

        self._memmap = np.memmap(self._path, mode="r")
        # TODO: ====== Forward preprocessing config so the reader reads just what we need.
        self._reader = reader_config.get_reader(memoryview(self._memmap))

    def __getstate__(self) -> tuple[str, pathlib.Path, dict, MemmapIndexDatasetReaderConfig]:
        # We pass the reader config to force its import in data loader workers.
        return self._name, self._path, self._preprocessing.to_dict(), self._reader.config

    def __setstate__(self, state: tuple[str, pathlib.Path, dict, MemmapIndexDatasetReaderConfig]):
        name, path, preprocessing, _ = state
        self._init(name, path, PreprocessingConfig.from_dict(preprocessing))

    def __del__(self):
        if hasattr(self, "_memmap"):
            self._memmap._mmap.close()  # noqa
            del self._memmap

    def get_document(
        self, index: int, begin: int = 0, end: int | None = None, parameters: SamplingParameters | None = None
    ) -> SampleType:
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

    @classmethod
    def write_dataset(
        cls,
        path: pathlib.Path,
        documents: typing.Iterable[Sample],
        writer_class: type[MemmapWriter],
        preprocessing_config: PreprocessingConfig | None = None,
    ) -> MemmapIndexDatasetReaderConfig:
        # TODO: Match `writer_class` with `SampleType`?
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            # Write the file type header.
            stream.write(FILE_HEADER)
            # Leave space for a pointer to the reader config.
            # We write the config  at the end since we don't know it yet.
            start = stream.tell()
            stream.seek(start + 8)
            # Write the data.
            reader_config = writer_class.write_dataset(stream, documents, preprocessing_config)
            # Write the reader config.
            config_offset = stream.tell()
            reader_config_bytes = json.dumps(reader_config.to_dict()).encode("utf-8")
            stream.write(len(reader_config_bytes).to_bytes(4, signed=False))
            stream.write(reader_config_bytes)
            # Write a pointer to the reader config.
            stream.seek(start)
            stream.write(config_offset.to_bytes(8, signed=False))
        return reader_config
