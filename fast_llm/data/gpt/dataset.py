import abc
import math

import numpy as np
import numpy.random

from fast_llm.data.config import Dataset
from fast_llm.utils import Assert

try:
    from fast_llm.csrc.data import build_sample_idx  # noqa

    _extension_available = True
except ImportError:
    _extension_available = False


class GPTIndexedDataset(Dataset):
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

    def sample(self, num_samples: int, sequence_length: int, np_rng: numpy.random.RandomState, verbose: bool):
        """
        Create a `GPTSampledDataset` with the requested parameters.
        """
        document_sizes = self.get_document_sizes()
        num_documents = len(document_sizes)
        num_tokens = document_sizes.sum()

        num_epochs = math.ceil((sequence_length * num_samples + 1) / num_tokens)
        # For the last epoch, decide whether include the entire epoch
        # in the global shuffle or not.
        # Get the number of samples for the last epoch
        main_epochs_samples = ((num_epochs - 1) * num_tokens - 1) // sequence_length
        last_epoch_samples = num_samples - main_epochs_samples
        samples_per_epoch = (num_tokens - 1) // sequence_length
        # If we have less than 80% of the samples for the last epoch, separate out the epoch and treat it differently.
        # Note: the 80% number is just based on common sense and can be adjusted if needed.
        separate_last_epoch = num_epochs > 1 and last_epoch_samples < 0.8 * samples_per_epoch

        doc_idx = np.tile(np.arange(num_documents, dtype=np.int32), num_epochs)
        if separate_last_epoch:
            np_rng.shuffle(doc_idx[:-num_documents])
            np_rng.shuffle(doc_idx[-num_documents:])
        else:
            np_rng.shuffle(doc_idx)

        assert _extension_available, (
            "The C++ extension for dataset sampling is missing." " Please make sure Fast-LLM is installed correctly."
        )

        sample_idx = build_sample_idx(document_sizes, doc_idx, sequence_length, num_epochs, num_tokens, verbose)

        # shuffle-idx.
        # -1 is due to data structure used to retrieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        total_size = sample_idx.shape[0] - 1
        # TODO: Isn't the dataset already shuffled above?
        shuffle_idx = np.arange(
            0, total_size, dtype=np.int64 if total_size >= (np.iinfo(np.uint32).max - 1) else np.uint32
        )
        if separate_last_epoch:
            np_rng.shuffle(shuffle_idx[:main_epochs_samples])
            np_rng.shuffle(shuffle_idx[main_epochs_samples:])
        else:
            np_rng.shuffle(shuffle_idx)

        Assert.geq(len(shuffle_idx), num_samples)
        # TODO: The doc and sample idx are way bigger than needed when sampling for << 1 epoch.
        return doc_idx, sample_idx, shuffle_idx[:num_samples]
