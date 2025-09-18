import logging
import math
import typing

from fast_llm.engine.distributed.config import DistributedConfig, DistributedDim
from fast_llm.utils import Assert, div

if typing.TYPE_CHECKING:
    import torch

    from fast_llm.core.distributed import ProcessGroup
    from fast_llm.engine.distributed.distributed import Distributed

logger = logging.getLogger(__name__)


class TensorDim:
    """
    Describes a simple, atomic dimension of a tensor and its size.
    The dimension may be parallelized along a distributed dimension `parallel_dim`,
    in which case its actual (local) `size` will differ from its `global_size`.

    TensorDim's are used to represent the metadata of tensors through `TensorMeta`.

    This class also serves as a base for more complex tensor dimensions.
    """

    def __init__(self, name: str, global_size: int | None, parallel_dim: DistributedDim | None = None):
        # TODO: Handle None for unknown sizes?
        self._name = name
        self._global_size = global_size
        self._size = self._global_size if parallel_dim is None else div(global_size, parallel_dim.size)
        self._parallel_dim = parallel_dim

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"name={self._name},"
            f" size={self._size},"
            f" global_size={self._global_size},"
            f" parallel_dim={self._parallel_dim}"
            f")"
        )

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        return self._size

    @property
    def global_size(self) -> int:
        return self._global_size

    @property
    def is_parallel(self) -> bool:
        return self._parallel_dim is not None and self._parallel_dim.size > 1

    @property
    def parallel_dim(self) -> DistributedDim | None:
        # TODO: Make more flexible for derived classes?
        return self._parallel_dim

    @property
    def parallel_group(self) -> "ProcessGroup|None":
        # TODO: Make more flexible for derived classes?
        return None if self._parallel_dim is None else self._parallel_dim.group

    def replace_parallel_dim(self, distributed_dim: DistributedDim) -> typing.Self:
        """
        Create a copy of the tensor dimension, where the parallel dimension is replaced by `distributed_dim`,
        but the local size remains the same.

        Used in`TensorMeta.replace_tensor_parallel_dim`.
        """
        assert self.is_parallel
        return TensorDim(self.name, self.size * distributed_dim.size, distributed_dim)

    def local_to_global(self, tensor: "torch.Tensor", dim: int = 0) -> "torch.Tensor":
        """
        Partially reconstruct a global tensor from local `tensor` slices whose dimension `dim` is described by `self`.
        If the dimension is parallelized, this amounts to gathering along dimension `dim`
        and parallel dimension `parallel_dim`, otherwise return the input tensor.
        The method needs to be called my all members of the parallel group using their appropriate local slice.

        Used in`TensorMeta.local_to_global`,
        which iterates over the tensor dimensions to fully reconstruct the global tensor.
        """
        if self.is_parallel:
            from fast_llm.core.ops import gather_op

            return gather_op(tensor, self.parallel_group, dim)
        else:
            return tensor

    def local_to_global_partial(
        self, tensor: "torch.Tensor", dim: int = 0, fill_value: float | int = -1
    ) -> "torch.Tensor":
        """
        Partially reconstruct a global tensor from a local `tensor` whose dimension `dim` is described by `self`.
        Unlike `local_to_global`, this method does not need to be called from a distributed setting.
        Instead, entries from other ranks are populated with `fill_value`.

        Used in`TensorMeta.local_to_global_partial`,
        which iterates over the tensor dimensions to fully reconstruct the global tensor.
        """
        if self.is_parallel:
            output = tensor.new_full((*tensor.shape[:dim], self.parallel_dim.size, *tensor.shape[dim:]), fill_value)
            output.narrow(dim, self.parallel_dim.rank, 1).copy_(tensor.unsqueeze(dim)).squeeze(dim)
            return output.flatten(dim, dim + 1)
        else:
            return tensor

    def global_to_local(self, tensor: "torch.Tensor", dim: int = 0, expand: bool = False) -> "torch.Tensor":
        """
        Partially recover a local tensor slice from a global `tensor` whose dimension `dim` is described by `self`.
        If the dimension is parallel, this amounts to taking the `rank`th chunk of size `size` along dimension `dim`
        and parallel dimension `self.parallel_dim`, otherwise return the input tensor.

        Used in`TensorMeta.local_to_global`,
        which iterates over the tensor dimensions to fully reconstruct the local tensor.
        """
        return (
            tensor.chunk(self.parallel_dim.size, dim)[self.parallel_dim.rank]
            if self.parallel_dim is not None and self.parallel_dim.size > 1
            else tensor
        )


class CompositeTensorDim(TensorDim):
    """
    A composite tensor dimension that represent multiple dimensions flattened into ones.
    Typically happens for flattened view or higher-dimensional tensors, or tensors that can be expanded as such.
    If one of the composed dimensions -- other than the first one -- is parallelized,
    this is **not** equivalent to an atomic `TensorDim` of the  same size,
    as the relation between local and global tensors is different.

    At most one of the sub-dimensions may be parallelized. TODO: Allow for more than one?
    """

    def __init__(self, name: str, tensor_dims: tuple[TensorDim, ...]):
        parallel_dim = None
        for dim, tensor_dim in enumerate(tensor_dims):
            if tensor_dim.parallel_dim is not None:
                assert parallel_dim is None
                parallel_dim = tensor_dim.parallel_dim
                self._parallel_dim_index = dim

        super().__init__(
            name=name,
            global_size=math.prod(dim.global_size for dim in tensor_dims),
            parallel_dim=parallel_dim,
        )
        self._tensor_dims = tensor_dims

    def replace_parallel_dim(self, distributed_dim: DistributedDim) -> typing.Self:
        """
        Create a copy of the tensor dimension, where the parallel dimension is replaced by `distributed_dim`,
        but the local size remains the same.
        """
        assert self._parallel_dim_index is not None
        dims = list(self._tensor_dims)
        dims[self._parallel_dim_index] = dims[self._parallel_dim_index].replace_parallel_dim(distributed_dim)
        return CompositeTensorDim(self.name, tuple(dims))

    def local_to_global(self, tensor: "torch.Tensor", dim: int = 0) -> "torch.Tensor":
        """
        Partially reconstruct a global tensor from local `tensor` slices whose dimension `dim` is described by `self`.
        """
        tensor = tensor.unflatten(dim, [tensor_dim.size for tensor_dim in self._tensor_dims])
        for i, tensor_dim in enumerate(self._tensor_dims):
            tensor = tensor_dim.local_to_global(tensor, dim + i)

        return tensor.flatten(dim, dim + len(self._tensor_dims) - 1)

    def local_to_global_partial(
        self, tensor: "torch.Tensor", dim: int = 0, fill_value: float | int = -1
    ) -> "torch.Tensor":
        """
        Partially reconstruct a global tensor from a local `tensor` whose dimension `dim` is described by `self`,
        populating other ranks with `fill_value`.
        """
        tensor = tensor.unflatten(dim, [tensor_dim.size for tensor_dim in self._tensor_dims])
        for i, tensor_dim in enumerate(self._tensor_dims):
            tensor = tensor_dim.local_to_global_partial(tensor, dim + i)

        return tensor.flatten(dim, dim + len(self._tensor_dims) - 1)

    def global_to_local(self, tensor: "torch.Tensor", dim: int = 0, expand: bool = False) -> "torch.Tensor":
        """
        Partially recover a local tensor slice from a global `tensor` whose dimension `dim` is described by `self`.
        """
        tensor = tensor.unflatten(dim, [tensor_dim.global_size for tensor_dim in self._tensor_dims])
        for i, tensor_dim in reversed(list(enumerate(self._tensor_dims))):
            tensor = tensor_dim.global_to_local(tensor, dim + i)
        return tensor if expand else tensor.flatten(dim, dim + len(self._tensor_dims) - 1)


class ConcatenatedTensorDim(TensorDim):
    """
    A complex tensor dimension that results from concatenating tensors.

    All sub-dimensions should have the same `parallel_dim` (may be None). TODO: Allow for more complex scenarios?
    """

    def __init__(self, name: str, tensor_dims: tuple[TensorDim, ...]):
        parallel_dim = tensor_dims[0].parallel_dim
        for dim, tensor_dim in enumerate(tensor_dims[1:]):
            # TODO: Allow more flexibility?
            Assert.is_(tensor_dim.parallel_dim, parallel_dim)

        super().__init__(
            name=name,
            global_size=sum(dim.global_size for dim in tensor_dims),
            parallel_dim=parallel_dim,
        )
        self._tensor_dims = tensor_dims

    def replace_parallel_dim(self, distributed_dim: DistributedDim) -> typing.Self:
        """
        Create a copy of the tensor dimension, where the parallel dimension is replaced by `distributed_dim`,
        but the local size remains the same.
        """
        assert self.is_parallel
        return ConcatenatedTensorDim(
            self.name, tuple(tensor_dim.replace_parallel_dim(distributed_dim) for tensor_dim in self._tensor_dims)
        )

    def local_to_global(self, tensor: "torch.Tensor", dim: int = 0) -> "torch.Tensor":
        """
        Partially reconstruct a global tensor from local `tensor` slices whose dimension `dim` is described by `self`.
        """
        import torch

        return (
            torch.concatenate(
                [
                    tensor_dim.local_to_global(tensor_, dim)
                    for tensor_, tensor_dim in zip(
                        tensor.split([tensor_dim.size for tensor_dim in self._tensor_dims], dim),
                        self._tensor_dims,
                        strict=True,
                    )
                ],
                dim,
            )
            if self.is_parallel
            else tensor
        )

    def local_to_global_partial(
        self, tensor: "torch.Tensor", dim: int = 0, fill_value: float | int = -1
    ) -> "torch.Tensor":
        """
        Partially reconstruct a global tensor from a local `tensor` whose dimension `dim` is described by `self`,
        populating other ranks with `fill_value`.
        """
        import torch

        return (
            torch.concatenate(
                [
                    tensor_dim.local_to_global_partial(tensor_, dim)
                    for tensor_, tensor_dim in zip(
                        tensor.split([tensor_dim.size for tensor_dim in self._tensor_dims], dim),
                        self._tensor_dims,
                        strict=True,
                    )
                ],
                dim,
            )
            if self.is_parallel
            else tensor
        )

    def global_to_local(self, tensor: "torch.Tensor", dim: int = 0, expand: bool = False) -> "torch.Tensor":
        """
        Partially recover a local tensor slice from a global `tensor` whose dimension `dim` is described by `self`.
        """
        if self.is_parallel and expand:
            raise NotImplementedError()
        import torch

        return (
            torch.concatenate(
                [
                    tensor_dim.global_to_local(tensor_, dim)
                    for tensor_, tensor_dim in zip(
                        tensor.split([tensor_dim.global_size for tensor_dim in self._tensor_dims], dim),
                        self._tensor_dims,
                        strict=True,
                    )
                ],
                dim,
            )
            if self.is_parallel
            else tensor
        )


class DefaultDimNames:
    # Scalar
    scalar = "scalar"


class TensorSpace:
    _is_setup: bool = False
    _distributed: "Distributed"

    def __init__(self, distributed_config: DistributedConfig):
        self._distributed_config = distributed_config
        self._tensor_dims: dict[str, TensorDim] = {}
        self.add_tensor_dim(TensorDim(DefaultDimNames.scalar, 1))

    def setup(self, distributed: "Distributed") -> None:
        assert not self._is_setup
        if distributed.config is not self._distributed_config:
            distributed.config.compare(self._distributed_config, ValueError)
        self._is_setup = True
        self._distributed = distributed

    @property
    def distributed_config(self) -> DistributedConfig:
        return self._distributed_config

    @property
    def distributed(self) -> "Distributed":
        assert self._is_setup
        return self._distributed

    def add_tensor_dim(self, tensor_dim: TensorDim) -> None:
        if tensor_dim.name in self._tensor_dims:
            Assert.eq(tensor_dim, self._tensor_dims[tensor_dim.name])
        else:
            if tensor_dim.parallel_dim is not None:
                assert (
                    tensor_dim.parallel_dim.name in self._distributed_config.distributed_dims
                ), tensor_dim.parallel_dim.name
                Assert.eq(
                    tensor_dim.parallel_dim.__dict__,
                    self._distributed_config.distributed_dims[tensor_dim.parallel_dim.name].__dict__,
                )
            self._tensor_dims[tensor_dim.name] = tensor_dim

    def __getitem__(self, name: str) -> TensorDim:
        return self._tensor_dims[name]
