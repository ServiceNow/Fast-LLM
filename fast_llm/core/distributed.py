"""
Basic distributed ops from torch.distributed, and adaptations to support our usage:
* Bypass the group tracking mess and use groups as plain, isolated pytorch objects.
* Use local ranks (to a given group) exclusively, and avoid unnecessary back and forth between local and global ranks.
Import all necessary content from torch.distributed here to keep track of their usages and simplify future changes.
Todo: Move all core methods elsewhere (functional?).
"""

import contextlib
import datetime
import io
import logging
import pickle
import typing

import torch
import torch.monitor
from torch._C._distributed_c10d import Work
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


def add_ephemeral_timeout(group: ProcessGroup, timeout: float | None = None) -> None:
    if group is not None and timeout is not None:
        # TODO: Only works for nccl?
        group._add_ephemeral_timeout(datetime.timedelta(seconds=timeout))


def broadcast(
    tensor: torch.Tensor, src: int, group: ProcessGroup, async_op=False, timeout: float | None = None
) -> Work | None:
    """Same as torch.distributed.broadcast, but without the complication of going through the global rank."""
    assert group is not None
    opts = torch.distributed.BroadcastOptions()
    opts.rootRank = src
    opts.rootTensor = 0
    add_ephemeral_timeout(group, timeout)
    work = group.broadcast([tensor], opts)
    if async_op:
        return work
    else:
        work.wait()
        return None


def check_parallel_match(tensor: torch.Tensor, group: ProcessGroup | None, name: str) -> None:
    # A utility function to check for tensor-parallel (or other) mismatches.
    all_tensors = tensor.new_empty((group.size(),) + tensor.shape)
    all_gather_into_tensor(all_tensors, tensor, group)

    mismatches = (all_tensors != tensor).any(dim=0)
    num_mismatches = mismatches.sum().item()
    if num_mismatches > 0:
        num_nans = tensor.isnan().sum().item()
        logger.error(
            f"MISMATCH {name} {num_mismatches:,} / {tensor.numel():,}"
            + ("" if num_nans == 0 else f" [{num_nans:,} nans detected locally]")
        )


def safe_barrier(group: ProcessGroup | None, value: int | str = 1, timeout: float | None = None) -> None:
    if group:
        hashed = hash(value) % 2**32
        out = allreduce_scalar(hashed, dtype=torch.int64, group=group, timeout=timeout)
        if out != hashed * group.size():
            raise RuntimeError(f"Desync detected for barrier {value} ({out}!={hashed*group.size()})")


def allreduce_scalar(
    value: float | int,
    dtype: torch.dtype = torch.float64,
    group: torch.distributed.ProcessGroup | None = None,
    op=ReduceOp.SUM,
    timeout: float | None = None,
) -> float | int:
    if group:
        value = torch.full([1], value, dtype=dtype, device=torch.cuda.current_device())
        add_ephemeral_timeout(group, timeout)
        torch.distributed.all_reduce(value, op=op, group=group)
        return value.item()
    else:
        return value


def broadcast_scalar(
    value: float | int,
    dtype: torch.dtype = torch.float64,
    group: torch.distributed.ProcessGroup | None = None,
    src: int = 0,
    timeout: float | None = None,
) -> float | int:
    if not group:
        return value
    tensor = torch.empty([1], dtype=dtype, device=torch.device(torch.cuda.current_device()))
    if group.rank() == src:
        tensor.fill_(value)
    broadcast(tensor, src, group, timeout=timeout)
    return tensor.item()


def send(tensor: torch.Tensor, dst: int, group: ProcessGroup, async_op=False, tag: int = 0) -> Work | None:
    assert group is not None
    work = group.send([tensor], dst, tag)
    if async_op:
        return work
    else:
        work.wait()
        return None


def recv(tensor: torch.Tensor, src: int, group: ProcessGroup, async_op=False, tag: int = 0) -> Work | None:
    assert group is not None
    work = group.recv([tensor], src, tag)
    if async_op:
        return work
    else:
        work.wait()
        return None


@contextlib.contextmanager
def set_generator(generator: torch.Generator) -> typing.Generator[None, None, None]:
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


def gather(
    tensor: torch.Tensor,
    gather_list: list[torch.Tensor] | None = None,
    group: ProcessGroup | None = None,
    async_op: bool = False,
    dst: int = 0,
):
    assert group is not None
    opts = torch.distributed.GatherOptions()
    opts.rootRank = dst
    work = group.gather([gather_list] if dst == group.rank() else [], [tensor], opts)

    if async_op:
        return work
    elif work is not None:
        work.wait()
        return None


def scatter(
    tensor: torch.Tensor,
    scatter_list: list[torch.Tensor] | None = None,
    group: ProcessGroup | None = None,
    async_op: bool = False,
    src: int = 0,
):
    assert group is not None
    opts = torch.distributed.ScatterOptions()
    opts.rootRank = src
    opts.asyncOp = async_op
    work = group.scatter(
        [tensor if not tensor.is_complex() else torch.view_as_real(tensor)],
        [[t if not t.is_complex() else torch.view_as_real(t) for t in scatter_list]] if src == group.rank() else [],
        opts,
    )
    if async_op:
        return work
    elif work is not None:
        work.wait()
        return None


def _object_to_tensor(obj: typing.Any) -> torch.Tensor:
    f = io.BytesIO()
    pickle.Pickler(f).dump(obj)
    return torch.tensor(torch.UntypedStorage.from_buffer(f.getvalue(), dtype=torch.uint8), dtype=torch.uint8)


def _tensor_to_object(tensor: torch.Tensor) -> typing.Any:
    return pickle.Unpickler(io.BytesIO(tensor.numpy(force=True).tobytes())).load()


def gather_object(
    obj: typing.Any,
    group: ProcessGroup | None = None,
    dst: int = 0,
) -> list[typing.Any] | None:
    assert group is not None
    group_rank = group.rank()
    group_size = group.size()
    device = torch.cuda.current_device()

    obj_tensor = _object_to_tensor(None if group_rank == dst else obj)
    sizes = torch.full([group.size()], len(obj_tensor), dtype=torch.int64, device=device)
    all_gather_into_tensor(sizes, sizes[group.rank()], group=group)
    sizes = sizes.tolist()
    max_size = max(sizes)

    input_tensor = torch.empty(max_size, dtype=torch.uint8, device=device)

    if group_rank == dst:
        output_tensors = list(torch.empty(max_size * group_size, dtype=torch.uint8, device=device).chunk(group_size))
        gather(input_tensor, output_tensors, dst=dst, group=group)
        return [
            obj if rank_ == dst else _tensor_to_object(tensor[:size])
            for rank_, (tensor, size) in enumerate(zip(output_tensors, sizes, strict=True))
        ]
    else:
        input_tensor[: obj_tensor.numel()].copy_(obj_tensor)
        gather(input_tensor, None, dst=dst, group=group)
        return None


def scatter_object(
    scatter_object_input_list: typing.Optional[list[typing.Any]] = None,
    group: ProcessGroup | None = None,
    src: int = 0,
) -> typing.Any:
    assert group is not None
    group_rank = group.rank()
    group_size = group.size()
    device = torch.cuda.current_device()

    if group_rank == src:
        tensor_list = [
            _object_to_tensor(None if rank_ == src else obj) for rank_, obj in enumerate(scatter_object_input_list)
        ]
        sizes = [tensor.numel() for tensor in tensor_list]
        max_size = max(sizes)
        size_tensor = torch.tensor([[size, max_size] for size in sizes], dtype=torch.int64, device=device)
        scatter(size_tensor[group_rank], list(size_tensor.unbind()), src=src, group=group)
        scatter_list = list(torch.empty(max_size * group_size, dtype=torch.uint8, device=device).chunk(group_size))
        for scatter_tensor, tensor, size in zip(scatter_list, tensor_list, sizes, strict=True):
            scatter_tensor[:size].copy_(tensor)
        scatter(scatter_list[src], scatter_list, src=src, group=group)
        return scatter_object_input_list[src]
    else:
        size_tensor = torch.empty(2, dtype=torch.int64, device=device)
        scatter(size_tensor, None, src=src, group=group)
        size, max_size = size_tensor.tolist()
        output_tensor = torch.empty(max_size, dtype=torch.uint8, device=device)
        scatter(output_tensor, None, src=src, group=group)
        return _tensor_to_object(output_tensor[:size])
