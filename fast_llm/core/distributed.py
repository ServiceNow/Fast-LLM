"""
Basic distributed ops from torch.distributed, and adaptations to support our usage:
* Bypass the group tracking mess and use groups as plain, isolated pytorch objects.
* Use local ranks (to a given group) exclusively, and avoid unnecessary back and forth between local and global ranks.
Import all necessary content from torch.distributed here to keep track of their usages and simplify future changes.
Todo: Move all core methods elsewhere (functional?).
"""

import collections
import contextlib
import datetime
import io
import itertools
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


def _check_single_tensor(param, param_name) -> None:
    """Check that the parameter ``param_name`` is a single tensor."""
    if not isinstance(param, torch.Tensor):
        raise TypeError(
            f"""Invalid function argument. Expected parameter `{param_name}` of type torch.Tensor
             but got {type(param)} instead."""
        )


def _check_tensor_list(param, param_name) -> None:
    """Check that the parameter ``param_name`` is a list of tensors."""
    if not isinstance(param, list):
        raise TypeError(
            f"""Invalid function argument. Expected parameter `{param_name}` of type List[torch.Tensor]
             but got {type(param)} instead."""
        )
    elif not all(isinstance(p, torch.Tensor) for p in param):
        raise TypeError(
            f"""Invalid function argument. Expected parameter `{param_name}` of type List[torch.Tensor]
             but got {type(param)} with elements of type {[type(p) for p in param]}."""
        )


def _as_iterable(obj) -> collections.abc.Iterable:
    return obj if isinstance(obj, list) else (obj,)


def _ensure_all_tensors_same_dtype(*tensors) -> None:
    last_dtype = None
    for tensor in itertools.chain.from_iterable(map(_as_iterable, tensors)):
        tensor_dtype = tensor.dtype
        # Mixing complex and its element type is allowed
        if tensor_dtype.is_complex:
            tensor_dtype = torch.float32 if tensor_dtype == torch.complex64 else torch.complex128

        if last_dtype is None:
            last_dtype = tensor_dtype
        else:
            if last_dtype != tensor_dtype:
                raise ValueError(
                    "Invalid usage of tensors with different dtypes" f"Found {last_dtype} and  {tensor.dtype}"
                )


def _rank_not_in_group(group: typing.Optional[ProcessGroup]) -> bool:
    """Check if the current process's rank is not in a given group."""
    if group is None:
        return False
    return group == torch.distributed.GroupMember.NON_GROUP_MEMBER


def _warn_not_in_group(op_name) -> None:
    # TODO: get global rank
    global_rank = -1
    logger.warning(f"Running {op_name} on global rank {global_rank} which does not " "belong to the given group.")


_pickler = pickle.Pickler
_unpickler = pickle.Unpickler


def _object_to_tensor(obj, device, group):
    with torch.monitor._WaitCounter("pytorch.wait_counter.c10d._object_to_tensor").guard():
        f = io.BytesIO()
        _pickler(f).dump(obj)
        byte_storage = torch.ByteStorage._from_buffer(f.getvalue())  # type: ignore[attr-defined]
        # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
        # Otherwise, it will casue 100X slowdown.
        # See: https://github.com/pytorch/pytorch/issues/65696
        byte_tensor = torch.ByteTensor(byte_storage).to(device)

        local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
        return byte_tensor, local_size


def _tensor_to_object(tensor, tensor_size, group):
    with torch.monitor._WaitCounter("pytorch.wait_counter.c10d._tensor_to_object").guard():
        tensor = tensor.cpu()
        buf = tensor.numpy().tobytes()[:tensor_size]
        return _unpickler(io.BytesIO(buf)).load()


def _validate_output_list_for_rank(my_rank, dst, gather_list):
    if dst == my_rank:
        if not gather_list:
            raise ValueError("Argument ``gather_list`` must be specified on destination rank.")
    elif gather_list:
        raise ValueError("Argument ``gather_list`` must NOT be specified on non-destination ranks.")


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


def recv(tensor: torch.Tensor, src: int, group: ProcessGroup, async_op=False, tag: int = 0) -> Work | None:
    assert group is not None
    work = group.recv([tensor], src, tag)
    if async_op:
        return work
    else:
        work.wait()


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
    gather_list: typing.Optional[list[torch.Tensor]] = None,
    group: typing.Optional[ProcessGroup] = None,
    async_op: bool = False,
    group_dst: typing.Optional[int] = None,
):
    _check_single_tensor(tensor, "tensor")

    # Parameter ``gather_list`` may be left unspecified on non-dst ranks.
    if gather_list:
        _check_tensor_list(gather_list, "gather_list")
    else:
        gather_list = []
    _ensure_all_tensors_same_dtype(tensor, gather_list)
    assert group is not None
    if _rank_not_in_group(group):
        _warn_not_in_group("gather")
        return
    if group_dst is None:
        group_dst = 0
    my_group_rank = group.rank()
    _validate_output_list_for_rank(my_group_rank, group_dst, gather_list)
    output_tensors = [gather_list] if group_dst == my_group_rank else []
    input_tensors = [tensor]

    opts = torch.distributed.GatherOptions()
    opts.rootRank = group_dst
    # Absent in ver 2.6
    # opts.asyncOp = async_op
    work = group.gather(output_tensors, input_tensors, opts)

    if async_op:
        return work
    elif work is not None:  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


def scatter(
    tensor: torch.Tensor,
    scatter_list: typing.Optional[list[torch.Tensor]] = None,
    group: typing.Optional[ProcessGroup] = None,
    async_op: bool = False,
    group_src: typing.Optional[int] = None,
):
    _check_single_tensor(tensor, "tensor")
    # Parameter ``scatter_list`` may be left unspecified on non-src ranks.
    if scatter_list:
        _check_tensor_list(scatter_list, "scatter_list")
    else:
        scatter_list = []
    _ensure_all_tensors_same_dtype(tensor, scatter_list)
    assert group is not None
    if group_src is None:
        group_src = 0
    if _rank_not_in_group(group):
        _warn_not_in_group("scatter")
        return
    scatter_list = [t if not t.is_complex() else torch.view_as_real(t) for t in scatter_list]
    tensor = tensor if not tensor.is_complex() else torch.view_as_real(tensor)

    my_group_rank = group.rank()
    if group_src == my_group_rank:
        if not scatter_list:
            raise ValueError("Argument ``scatter_list`` must be specified on source rank.")
        input_tensors = [scatter_list]
        output_tensors = [tensor]
    else:
        if scatter_list:
            raise ValueError("Argument ``scatter_list`` must NOT be specified on non-source ranks.")
        input_tensors = []
        output_tensors = [tensor]

    opts = torch.distributed.ScatterOptions()
    opts.rootRank = group_src
    opts.asyncOp = async_op
    work = group.scatter(output_tensors, input_tensors, opts)

    if async_op:
        return work
    elif work is not None:  # Backward compatible with backends that don't sync at CPP level
        work.wait()
    # Otherwise, the backend has sync'ed at CPP level


def gather_object(
    current_device: torch.device | str,
    obj: typing.Any,
    object_gather_list: typing.Optional[list[typing.Any]] = None,
    group: typing.Optional[ProcessGroup] = None,
    group_dst: typing.Optional[int] = None,
):
    assert group is not None
    if group_dst is None:
        group_dst = 0
    if _rank_not_in_group(group):
        _warn_not_in_group("gather_object")
        return

    # Ensure object_gather_list is specified appropriately.
    my_group_rank = group.rank()
    _validate_output_list_for_rank(my_group_rank, group_dst, object_gather_list)
    input_tensor, local_size = _object_to_tensor(obj, current_device, group)

    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = group.size()
    object_sizes_tensor = torch.zeros(group_size, dtype=torch.long, device=current_device)
    object_size_list = [object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)]
    # Allgather tensor sizes. An all-gather is needed here despite this being a
    # gather, since each rank needs to broadcast a tensor of the same (maximal)
    # size.
    all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    # Avoid populating output tensors if the result won't be gathered on this rank.
    if my_group_rank == group_dst:
        coalesced_output_tensor = torch.empty(max_object_size * group_size, dtype=torch.uint8, device=current_device)
        # Output tensors are nonoverlapping views of coalesced_output_tensor
        output_tensors = [
            coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)] for i in range(group_size)
        ]
    # All ranks call gather with equal-sized tensors.
    gather(
        input_tensor,
        gather_list=output_tensors if my_group_rank == group_dst else None,  # type: ignore[possibly-undefined]
        group_dst=group_dst,
        group=group,
    )
    if my_group_rank != group_dst:
        return

    assert object_gather_list is not None, "Must provide object_gather_list on dst rank"
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        tensor_size = object_size_list[i]
        object_gather_list[i] = _tensor_to_object(tensor, tensor_size, group)


def scatter_object_list(
    pg_device: torch.device | str,
    scatter_object_output_list: list[typing.Any],
    scatter_object_input_list: typing.Optional[list[typing.Any]] = None,
    group: typing.Optional[ProcessGroup] = None,
    group_src: typing.Optional[int] = None,
):
    assert group is not None
    if group_src is None:
        group_src = 0
    if _rank_not_in_group(group):
        _warn_not_in_group("scatter_object_list")
        return

    if not isinstance(scatter_object_output_list, list) or len(scatter_object_output_list) < 1:
        raise ValueError("Expected argument scatter_object_output_list to be a list of size at least 1.")

    my_group_rank = group.rank()
    if my_group_rank == group_src:
        if scatter_object_input_list is None:
            raise ValueError("source rank must provide non-None scatter_object_input_list")
        tensor_list, tensor_sizes = zip(
            *[_object_to_tensor(obj, pg_device, group) for obj in scatter_object_input_list]
        )
        tensor_list, tensor_sizes = list(tensor_list), list(tensor_sizes)

        # Src rank broadcasts the maximum tensor size. This is because all ranks are
        # expected to call into scatter() with equal-sized tensors.
        max_tensor_size = max(tensor_sizes)  # type: ignore[possibly-undefined]
        for tensor in tensor_list:  # type: ignore[possibly-undefined]
            tensor.resize_(max_tensor_size)
    else:
        max_tensor_size = torch.tensor([0], dtype=torch.long, device=pg_device)
    broadcast(max_tensor_size, src=group_src, group=group)

    # Scatter actual serialized objects
    output_tensor = torch.empty(max_tensor_size.item(), dtype=torch.uint8, device=pg_device)
    scatter(
        output_tensor,
        scatter_list=None if my_group_rank != group_src else tensor_list,  # type: ignore[possibly-undefined]
        group_src=group_src,
        group=group,
    )

    # Scatter per-object sizes to trim tensors when deserializing back to object
    obj_tensor_size = torch.tensor([0], dtype=torch.long, device=pg_device)
    scatter(
        obj_tensor_size,
        scatter_list=None if my_group_rank != group_src else tensor_sizes,  # type: ignore[possibly-undefined]
        group_src=group_src,
        group=group,
    )

    # Deserialize back to object
    scatter_object_output_list[0] = _tensor_to_object(output_tensor, obj_tensor_size, group)
