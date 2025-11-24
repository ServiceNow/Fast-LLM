import logging
import time
import typing

from fast_llm.data.dataset.abstract import SampledDataset

try:
    from fast_llm.csrc.data import build_blending_indices  # noqa

    _extension_available = True
except ImportError:
    _extension_available = False

logger = logging.getLogger(__name__)


class DatasetMonitor(SampledDataset):
    """
    A blended sampling of multiple sampled datasets, where each dataset is sampled with the provided probability.
    The sampling order of each dataset is respected, but there is no strict guarantee
    on the total number of samples from each dataset.
    The sampling exactly matches Megatron-LM with matching parameters.
    """

    def __init__(
        self,
        dataset: SampledDataset,
        data_sample_warn_time_ms: float,
    ):
        self._dataset = dataset
        self._data_sample_warn_time_ms = data_sample_warn_time_ms

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx) -> typing.Any:
        start_time = time.perf_counter()
        try:
            sample = self._dataset[idx]
            sample_time = (time.perf_counter() - start_time) * 1000
            if sample_time > self._data_sample_warn_time_ms:
                logger.warning(
                    f"Sample {idx} from dataset {self._dataset.name})" f" took {sample_time:,.2f} ms to load"
                )
            return sample

        except Exception:
            logger.error(f"Failed to get sample {idx} from dataset {self._dataset.name}")
            raise

    @property
    def name(self) -> str:
        return self._dataset.name
