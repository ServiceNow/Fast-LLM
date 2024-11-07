import numpy as np

from fast_llm.data.gpt.config import GPTRawDataset
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert, padded_cumsum


class GPTDatasetSlice(GPTRawDataset):
    """
    A GPT dataset, which reads samples from (a split of) a `MMapIndexedDataset` pointing to a GPT dataset.
    """

    def __init__(
        self,
        name: str,
        dataset: GPTRawDataset,
        begin: int | None = None,
        end: int | None = None,
    ):
        self._name = name
        self._dataset = dataset
        self._begin = 0 if begin is None else begin
        self._end = len(dataset) if end is None else end

        # Checks
        try:
            Assert.geq(self._begin, 0)
            Assert.in_range_incl(self._end, self._begin + 1, len(dataset))
        except Exception as e:
            raise AssertionError(f"Invalid document indices for dataset {name} with length {len(dataset)}") from e

    def __getitem__(self, index: int):
        """
        Get the sample (document) with the given index (in the split dataset).
        """
        return self.get(index)

    def get(self, document: int, offset: int = 0, length: int | None = None):
        """
        Get the sample (document) with the given index (in the dataset slice),
        optionally sub-sampled to a specific offset (starting point) and maximum length
        (end = min(offset + length, sample_length).
        """
        return self._dataset.get(document + self._begin, offset, length)

    @property
    def num_documents(self):
        return self._end - self._begin

    @property
    def num_tokens(self):
        return np.sum(self._dataset.document_sizes[self._begin : self._end])

    @property
    def name(self):
        return self._name

    @classmethod
    def from_splits(cls, dataset: GPTRawDataset, phase_split: dict[PhaseType, float]):
        """
        Create a set of GPT datasets from a MMapIndexedDataset,
        each containing approximately the requested proportion of the total tokens.
        """
        split_probs = list(phase_split.values())
        Assert.eq(sum(split_probs), 1)
        num_documents = dataset.num_documents
        splits = [round(x) for x in padded_cumsum(split_probs) * num_documents]
        return {
            phase: GPTDatasetSlice(f"{dataset.name}_{phase.value}", dataset, split_begin, split_end)
            for phase, split_begin, split_end in zip(phase_split, splits[:-1], splits[1:])
        }