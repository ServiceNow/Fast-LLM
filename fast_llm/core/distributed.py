"""
Basic distributed ops from torch.distributed, and adaptations to support our usage:
* Bypass the group tracking mess and use groups as plain, isolated pytorch objects.
* Use local ranks (to a given group) exclusively, and avoid unnecessary back and forth between local and global ranks.
Import all necessary content from torch.distributed here to keep track of their usages and simplify future changes.
Todo: Move all core methods elsewhere (functional?).
"""

import contextlib
import logging

import torch
from torch.distributed import (  # noqa
    ProcessGroup,
    ReduceOp,
    all_gather,
    all_gather_into_tensor,
    all_reduce,
    reduce_scatter,
    reduce_scatter_tensor,
)

logger = logging.getLogger(__name__)


def broadcast(tensor, src, group, async_op=False):
    """Same as torch.distributed.broadcast, but without the complication of going through the global rank."""
    assert group is not None
    opts = torch.distributed.BroadcastOptions()
    opts.rootRank = src
    opts.rootTensor = 0
    work = group.broadcast([tensor], opts)
    if async_op:
        return work
    else:
        work.wait()


def check_parallel_match(tensor: torch.Tensor, group: ProcessGroup | None, name: str):
    # A utility function to check for tensor-parallel (or other) mismatches.
    all_tensors = tensor.new_empty((group.size(),) + tensor.shape)
    all_gather_into_tensor(all_tensors, tensor, group)
    mismatches = (all_tensors != tensor).any(dim=0)
    num_mismatches = mismatches.sum().item()
    if num_mismatches > 0:
        logger.error(f"MISMATCH {name} {num_mismatches} / {tensor.numel()}")


def safe_barrier(group: ProcessGroup | None, value: int | str = 1):
    if group:
        hashed = hash(value) % 2**32
        out = allreduce_scalar(hashed, dtype=torch.int64, group=group)
        if out != hashed * group.size():
            raise RuntimeError(f"Desync detected for barrier {value} ({out}!={hashed*group.size()})")


def allreduce_scalar(
    value,
    dtype: torch.dtype = torch.float64,
    group: torch.distributed.ProcessGroup | None = None,
    op=ReduceOp.SUM,
):
    if group:
        value = torch.full([1], value, dtype=dtype, device=torch.cuda.current_device())
        torch.distributed.all_reduce(value, op=op, group=group)
        return value.item()
    else:
        return value


def broadcast_scalar(
    value,
    dtype: torch.dtype = torch.float64,
    group: torch.distributed.ProcessGroup | None = None,
    src: int = 0,
):
    if not group:
        return value
    tensor = torch.empty([1], dtype=dtype, device=torch.device(torch.cuda.current_device()))
    if group.rank() == src:
        tensor.fill_(value)
    broadcast(tensor, src, group)
    return tensor.item()


def send(tensor: torch.Tensor, dst: int, group: ProcessGroup, async_op=False, tag: int = 0):
    assert group is not None
    work = group.send([tensor], dst, tag)
    if async_op:
        return work
    else:
        work.wait()


def recv(tensor: torch.Tensor, src: int, group: ProcessGroup, async_op=False, tag: int = 0):
    assert group is not None
    work = group.recv([tensor], src, tag)
    if async_op:
        return work
    else:
        work.wait()


@contextlib.contextmanager
def set_generator(generator: torch.Generator):
    """Use the generator as default, for ops that don't support a generator argument."""
    default_generator: torch.Generator = torch.cuda.default_generators[torch.cuda.current_device()]
    assert generator is not default_generator
    old_state = default_generator.get_state()
    default_generator.set_state(generator.get_state())
    try:
        yield
    finally:
        generator.set_state(default_generator.get_state())
        default_generator.set_state(old_state)
