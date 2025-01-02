import logging
import pathlib
import time
import typing

import numpy as np

from fast_llm.core.distributed import safe_barrier
from fast_llm.data.data.config import SamplingConfig
from fast_llm.data.dataset.abstract import PhaseSplits, SampledDataset, SplitDataset
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.utils import Assert, normalize_probabilities

if typing.TYPE_CHECKING:
    from fast_llm.data.data.gpt.data import GPTData

try:
    from fast_llm.csrc.data import build_blending_indices  # noqa

    _extension_available = True
except ImportError:
    _extension_available = False

logger = logging.getLogger(__name__)


class BlendedDataset(SampledDataset):
    """
    A blended sampling of multiple sampled datasets, where each dataset is sampled with the provided probability.
    The sampling order of each dataset is respected, but there is no strict guarantee
    on the total number of samples from each dataset.
    The sampling exactly matches Megatron-LM with matching parameters.
    """

    def __init__(
        self,
        name: str,
        datasets_and_weights: list[tuple[SampledDataset, float]],
        sampling_config: SamplingConfig,
        # TODO: Generalize
        data: "GPTData",
    ):
        self._name = name
        assert len(datasets_and_weights) > 0
        self._datasets, weights = zip(*datasets_and_weights)
        self._weights = normalize_probabilities(weights)
        self._num_samples = sampling_config.num_samples
        self._data_sample_warn_time_ms = data.config.data_sample_warn_time_ms

        if sampling_config.cache_directory is None:
            self._dataset_idx_filename, self._sample_idx_filename = None, None
            self._dataset_index, self._sample_index = self._build_blending_indices(
                sampling_config.verbose and len(self._datasets) <= 20
            )
        else:
            group = data.distributed.world_group
            self._dataset_idx_filename = sampling_config.cache_directory / (self._name + "_blending_dataset_idx.npy")
            self._sample_idx_filename = sampling_config.cache_directory / (self._name + "_blending_sample_idx.npy")

            # Build the indexed mapping if it doesn't exist.
            # TODO: This only works if the dataset location is accessible by all job.
            if (group is None or group.rank() == 0) and not (
                self._dataset_idx_filename.is_file() and self._sample_idx_filename.is_file()
            ):
                dataset_index, sample_index = self._build_blending_indices(
                    sampling_config.verbose and len(self._datasets) <= 20
                )
                sampling_config.cache_directory.mkdir(exist_ok=True, parents=True)
                np.save(self._dataset_idx_filename, dataset_index)
                np.save(self._sample_idx_filename, sample_index)

            safe_barrier(group, self._name)
            self._load_mappings(sampling_config.verbose)

    @classmethod
    def apply(
        cls,
        name: str,
        datasets_and_weights: list[(SplitDataset[SampledDataset], float)],
        sampling_configs: PhaseSplits[SamplingConfig],
        data: "GPTData",
    ):
        Assert.leq(set(sampling_configs), set.union(*[set(dataset) for dataset, _ in datasets_and_weights]))
        return SplitDataset[BlendedDataset](
            name,
            {
                phase: BlendedDataset(
                    f"{name}_{phase.value}",
                    [
                        (dataset[phase], weight)
                        for dataset, weight in datasets_and_weights
                        if phase in dataset and weight > 0
                    ],
                    sampling_config,
                    data,
                )
                for phase, sampling_config in sampling_configs.items()
            },
        )

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
        assert _extension_available, (
            "The C++ extension for dataset blending is missing." " Please make sure Fast-LLM is installed correctly."
        )
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
