import typing

import torch.utils.data


class SampledDatasetIterator(torch.utils.data.Sampler):
    """
    A distributed sampler generating indices for a `SampledDataset` (i.e., the natural numbers).
    To be used as the `batch_sampler` of a `torch.utils.data.DataLoader`.
    """

    def __init__(self, total_samples, begin_index, micro_batch_size, data_rank, data_parallel):
        super().__init__()
        self._total_samples = total_samples
        self._begin_index = begin_index
        self._batch_size = micro_batch_size * data_parallel
        self._start_idx = data_rank * micro_batch_size
        self._end_idx = (data_rank + 1) * micro_batch_size

    def __len__(self) -> int:
        return self._total_samples

    def __iter__(self) -> typing.Iterator[list[int]]:
        for idx in range(self._begin_index, self._total_samples - self._batch_size + 1, self._batch_size):
            yield list(range(idx + self._start_idx, idx + self._end_idx))
