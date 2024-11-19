import abc
import typing

import numpy as np

from fast_llm.data.config import SamplableDataset
from fast_llm.data.gpt.config import GPTSamplingConfig

if typing.TYPE_CHECKING:
    from fast_llm.data.gpt.data import GPTData


try:
    from fast_llm.csrc.data import build_sample_idx  # noqa

    _extension_available = True
except ImportError:
    _extension_available = False


class GPTIndexedDataset(SamplableDataset):
    """
    A GPT dataset containing a list of unsampled, unprocessed samples.
    TODO: Move sampling responsibility here?
    """

    def get(self, document: int, offset: int = 0, length: int | None = None):
        pass

    @property
    def num_documents(self) -> int:
        """
        Number of documents in the dataset.
        Can be calculated from document sizes but may be overridden if there is a better method.
        """
        return len(self.get_document_sizes())

    @property
    def num_tokens(self) -> int:
        """
        Number of tokens in the dataset.
        Can be calculated from document sizes but may be overridden if there is a better method.
        """
        return self.get_document_sizes().sum()

    @abc.abstractmethod
    def get_document_sizes(self) -> "np.ndarray":
        """
        The size of each document in the dataset.
        The resulting array could be very large, so this method should be called cautiously,
        and derived classes should try to avoid holding the whole array im memory.
        """

    def sample(self, config: GPTSamplingConfig, data: "GPTData"):
        from fast_llm.data.gpt.sampled import GPTSampledIndexedDataset

        return GPTSampledIndexedDataset(self, config, data)
