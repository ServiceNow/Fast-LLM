import torch.distributed
import torch.utils.data.dataloader

from fast_llm.core.distributed import broadcast_object


class DistributedDataLoaderWrapper:
    """
    Wraps a regular dataloader so that only the process group leader
    loads data, and then broadcasts the batch to other ranks in the group.
    """

    def __init__(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader | None,
        rank: int,
        process_group: torch.distributed.ProcessGroup | None,
    ):
        self.dataloader = dataloader
        self.rank = rank
        self.process_group = process_group

        assert (self.rank == 0 and self.dataloader is not None) or (self.rank > 0 and self.dataloader is None)

    def __iter__(self):
        if self.rank == 0:
            self.iterator = iter(self.dataloader)
        if self.process_group is None:
            return self.iterator
        return self

    def __next__(self):
        # TODO:
        # Instead of broadcasting a general object, make this iterator yield actual batches.
        # Add batch data to a state dict or a dedicated Batch class, so we can efficiently
        # broadcast tensors directly. This avoids using `broadcast_object` on entire objects,
        # which is inefficient for tensors since it serializes them (pickles) before sending.
        if self.rank == 0:
            try:
                data = next(self.iterator)  # may raise StopIteration
            except Exception as e:
                data = e
            data = broadcast_object(data, self.process_group, 0)
        else:
            data = broadcast_object(None, self.process_group, 0)

        if isinstance(data, Exception):
            raise data

        return data
