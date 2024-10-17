import abc
import logging
import pathlib
import time

import numpy as np
import torch.utils.data

from fast_llm.core.distributed import ProcessGroup, safe_barrier
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.utils import Assert

try:
    from fast_llm.csrc.data import build_blending_indices  # noqa

    _extension_available = True
except ImportError:
    _extension_available = False

logger = logging.getLogger(__name__)


class Dataset(abc.ABC):
    """
    A generic dataset class compatible with torch.utils.data.Dataset but with a slightly different signature.
    """

    @abc.abstractmethod
    def __getitem__(self, index: int):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        """
        A name for the dataset to facilitate identification and debugging.
        """


class RawDataset(Dataset):  # noqa
    """
    A raw dataset class containing a list of unsampled, unprocessed samples, i.e., matching what is stored on disk.
    (Excluding off-line processing prior to training.)
    Functionally identical to a `Dataset`, but renamed for clarity.
    """


class SampledDataset(Dataset):  # noqa
    """
    A sampled dataset class containing a prepared list of samples to be indexed sequentially (as-is) during training.
    (See the `Sampler` class below.)
    Functionally identical to a `Dataset`, but renamed for clarity.
    """


class BlendedDataset(SampledDataset):
    """
    A blended sampling of multiple sampled datasets, where each dataset is sampled with the provided probability.
    The sampling order of each dataset is respected, but there is no strict guarantee
    on the total number of samples from each dataset.
    The sampling exactly matches Megatron-LM with matching parameters.
    """

    def __init__(
        self,
        datasets: list[SampledDataset],
        weights: list[float],
        *,
        name: str = "blended",
        num_samples: int,
        cache_dir: pathlib.Path | None = None,
        group: ProcessGroup | None = None,
        verbose: bool = True,
        data_sample_warn_time_ms: float = 1000,
    ):
        self._datasets = datasets
        self._name = name
        self._num_samples = num_samples
        self._weights = weights
        self._data_sample_warn_time_ms = data_sample_warn_time_ms

        if cache_dir is None:
            self._dataset_idx_filename, self._sample_idx_filename = None, None
            self._dataset_index, self._sample_index = self._build_blending_indices(verbose and len(datasets) <= 20)
        else:
            self._dataset_idx_filename = cache_dir / (self._name + "_blending_dataset_idx.npy")
            self._sample_idx_filename = cache_dir / (self._name + "_blending_sample_idx.npy")

            # Build the indexed mapping if it doesn't exist.
            # TODO: This only works if the dataset location is accessible by all job.
            if (group is None or group.rank() == 0) and not (
                self._dataset_idx_filename.is_file() and self._sample_idx_filename.is_file()
            ):
                dataset_index, sample_index = self._build_blending_indices(verbose and len(datasets) <= 20)
                np.save(self._dataset_idx_filename, dataset_index)
                np.save(self._sample_idx_filename, sample_index)

            safe_barrier(group, self._name)
            self._load_mappings(verbose)

    def __getstate__(self):
        return (
            self._datasets,
            self._name,
            self._num_samples,
            self._data_sample_warn_time_ms,
            self._dataset_index if self._dataset_idx_filename is None else self._dataset_idx_filename,
            self._sample_index if self._sample_idx_filename is None else self._sample_idx_filename,
        )

    def __setstate__(self, state):
        (
            self._datasets,
            self._name,
            self._num_samples,
            self._data_sample_warn_time_ms,
            dataset_index,
            sample_index,
        ) = state
        if isinstance(dataset_index, pathlib.Path):
            self._dataset_idx_filename, self._sample_idx_filename = dataset_index, sample_index
            self._load_mappings(False)
        else:
            self._dataset_idx_filename, self._sample_idx_filename = None, None
            self._dataset_index, self._sample_index = dataset_index, sample_index

    def _load_mappings(self, verbose):
        if verbose:
            log_main_rank(lambda: f" > loading blending dataset index mapping from {self._dataset_idx_filename}")
        self._dataset_index = np.load(self._dataset_idx_filename, mmap_mode="r")
        if verbose:
            log_main_rank(lambda: f" > loading blending dataset index mapping from {self._sample_idx_filename}")
        self._sample_index = np.load(self._sample_idx_filename, mmap_mode="r")

    def __len__(self):
        return self._num_samples

    def _build_blending_indices(self, verbose: bool):
        assert _extension_available, "Please run `make -C ./fast_llm/csrc/` first."
        Assert.lt(len(self._datasets), 32767)
        dataset_index = np.zeros(self._num_samples, dtype=np.int16)
        dataset_sample_index = np.zeros(self._num_samples, dtype=np.int64)
        build_blending_indices(
            dataset_index,
            dataset_sample_index,
            self._weights,
            len(self._datasets),
            self._num_samples,
            verbose,
        )
        available_samples_per_dataset = np.array([len(dataset) for dataset in self._datasets])
        sampled_per_dataset = np.bincount(dataset_index)
        # Oversampling is extremely unlikely but still possible.
        if not (sampled_per_dataset <= available_samples_per_dataset[: len(sampled_per_dataset)]).all():
            raise RuntimeError(
                "Failed to build blending indices:"
                + "".join(
                    [
                        f"\n  Dataset {i} {self._datasets[i].name} available {sampled}, "
                        f"sampled {available_samples_per_dataset[i]}"
                        for i, sampled in enumerate(sampled_per_dataset)
                        if sampled > available_samples_per_dataset[i]
                    ]
                )
            )
        return dataset_index, dataset_sample_index

    def __getitem__(self, idx):
        start_time = time.perf_counter()
        dataset_index = self._dataset_index[idx]
        dataset = self._datasets[dataset_index]
        sample_index = self._sample_index[idx]
        try:
            sample = dataset[sample_index]
            sample_time = (time.perf_counter() - start_time) * 1000
            if sample_time > self._data_sample_warn_time_ms:
                logger.warning(
                    f"Sample {sample_index} from dataset {dataset_index} ({dataset.name})"
                    f" took {sample_time:,.2f} ms to load"
                )
            return sample

        except Exception:
            logger.error(f"Failed to get sample {sample_index} from dataset {dataset_index} ({dataset.name})")
            raise

    @property
    def name(self):
        return self._name


class Sampler(torch.utils.data.Sampler):
    """
    A distributed sampler generating indices for a `SampledDataset` (i.e., the natural numbers).
    To be used as the `batch_sampler` of a `torch.utils.data.DataLoader`.
    """

    def __init__(self, total_samples, begin_index, micro_batch_size, data_rank, data_parallel):
        self._total_samples = total_samples
        self._begin_index = begin_index
        self._batch_size = micro_batch_size * data_parallel
        self._start_idx = data_rank * micro_batch_size
        self._end_idx = (data_rank + 1) * micro_batch_size

    def __len__(self):
        return self._total_samples

    def __iter__(self):
        for idx in range(self._begin_index, self._total_samples - self._batch_size + 1, self._batch_size):
            yield list(range(idx + self._start_idx, idx + self._end_idx))
