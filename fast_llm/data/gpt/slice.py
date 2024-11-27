from fast_llm.data.config import SamplableSplitDataset
from fast_llm.data.gpt.dataset import GPTIndexedDataset
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert, normalize_probabilities, padded_cumsum


class GPTDatasetSlice(GPTIndexedDataset):
    """
    A GPT dataset, which reads samples from (a split of) a `MMapIndexedDataset` pointing to a GPT dataset.
    """

    def __init__(
        self,
        name: str,
        dataset: GPTIndexedDataset,
        begin: int | None = None,
        end: int | None = None,
    ):
        self._name = name
        self._dataset = dataset
        self._begin = 0 if begin is None else begin
        dataset_documents = dataset.num_documents
        self._end = dataset_documents if end is None else end

        # Checks
        try:
            Assert.geq(self._begin, 0)
            Assert.in_range_incl(self._end, self._begin + 1, dataset_documents)
        except Exception as e:
            raise AssertionError(f"Invalid document indices for dataset {name} with length {dataset_documents}") from e

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

    def get_document_sizes(self):
        # TODO: This can be really big.
        return self._dataset.get_document_sizes()[self._begin : self._end]

    @property
    def name(self):
        return self._name

    @classmethod
    def from_splits(cls, dataset: GPTIndexedDataset, phase_split: dict[PhaseType, float]):
        """
        Create a set of GPT datasets from a MMapIndexedDataset,
        each containing approximately the requested proportion of the total tokens.
        """
        probabilities = normalize_probabilities(list(phase_split.values()))
        splits = [round(x) for x in padded_cumsum(probabilities) * dataset.num_documents]
        return SamplableSplitDataset[GPTDatasetSlice](
            f"{dataset.name}_split",
            {
                phase: GPTDatasetSlice(f"{dataset.name}_{phase.value}", dataset, split_begin, split_end)
                for phase, split_begin, split_end in zip(phase_split, splits[:-1], splits[1:])
            },
        )
