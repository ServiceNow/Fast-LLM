import itertools
import typing

import torch.utils.data

from fast_llm.core.distributed import broadcast_object


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


class DistributedDataLoaderWrapper:
    """
    Wraps a regular dataloader so that only the process group leader
    loads data, and then broadcasts the batch to other ranks in the group.
    """

    def __init__(
        self,
        data_loader: torch.utils.data.dataloader.DataLoader,
        process_group: torch.distributed.ProcessGroup | None,
    ):
        self._data_loader = data_loader
        self._rank = 0 if process_group is None else process_group.rank()
        self._process_group = process_group

    def __iter__(self):
        if self._rank == 0:
            self._iterator = iter(self._data_loader)
        else:
            self._iterator = itertools.repeat(None)
        if self._process_group is None:
            return self._iterator
        return self

    def __next__(self):
        # TODO:
        # Instead of broadcasting a general object, make this iterator yield an actual Batch class.
        # Implement `get_state_dict` and `from_state_dict` in the Batch class so that we can
        # efficiently broadcast tensors directly. This avoids using `broadcast_object` on the
        # entire Batch object, which is inefficient for tensors because it serializes
        # (pickles) them before sending.

        try:
            data = next(self.iterator)  # may raise StopIteration
        except Exception as e:
            data = e
        data = broadcast_object(data, self._process_group, 0)

        if isinstance(data, Exception):
            raise data

        return data
