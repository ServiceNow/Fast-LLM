import logging
import math
import typing

from fast_llm.engine.distributed.config import DistributedDim
from fast_llm.utils import Assert, div

if typing.TYPE_CHECKING:
    import torch

    from fast_llm.core.distributed import ProcessGroup

logger = logging.getLogger(__name__)


class TensorDim:
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
        assert self.is_parallel
        return TensorDim(self.name, self.size * distributed_dim.size, distributed_dim)

    def local_to_global(self, tensor: "torch.Tensor", dim: int = 0) -> "torch.Tensor":
        if self.is_parallel:
            from fast_llm.core.ops import gather_op

            return gather_op(tensor, self.parallel_group, dim)
        else:
            return tensor

    def local_to_global_partial(
        self, tensor: "torch.Tensor", dim: int = 0, fill_value: float | int = -1
    ) -> "torch.Tensor":
        if self.is_parallel:
            output = tensor.new_full((*tensor.shape[:dim], self.parallel_dim.size, *tensor.shape[dim:]), fill_value)
            output.narrow(dim, self.parallel_dim.rank, 1).copy_(tensor.unsqueeze(dim)).squeeze(dim)
            return output.flatten(dim, dim + 1)
        else:
            return tensor

    def global_to_local(self, tensor: "torch.Tensor", dim: int = 0, expand: bool = False) -> "torch.Tensor":
        return (
            tensor.chunk(self.parallel_dim.size, dim)[self.parallel_dim.rank]
            if self.parallel_dim is not None and self.parallel_dim.size > 1
            else tensor
        )


class CompositeTensorDim(TensorDim):
    def __init__(self, name: str, tensor_dims: tuple[TensorDim, ...]):
        parallel_dim = None
        for dim, tensor_dim in enumerate(tensor_dims):
            if tensor_dim.parallel_dim is not None:
                # TODO: Allow more than one parallel subdim?
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
        assert self._parallel_dim_index is not None
        dims = list(self._tensor_dims)
        dims[self._parallel_dim_index] = dims[self._parallel_dim_index].replace_parallel_dim(distributed_dim)
        return CompositeTensorDim(self.name, tuple(dims))

    def local_to_global(self, tensor: "torch.Tensor", dim: int = 0) -> "torch.Tensor":
        tensor = tensor.unflatten(dim, [tensor_dim.size for tensor_dim in self._tensor_dims])
        for i, tensor_dim in enumerate(self._tensor_dims):
            tensor = tensor_dim.local_to_global(tensor, dim + i)

        return tensor.flatten(dim, dim + len(self._tensor_dims) - 1)

    def local_to_global_partial(
        self, tensor: "torch.Tensor", dim: int = 0, fill_value: float | int = -1
    ) -> "torch.Tensor":
        tensor = tensor.unflatten(dim, [tensor_dim.size for tensor_dim in self._tensor_dims])
        for i, tensor_dim in enumerate(self._tensor_dims):
            tensor = tensor_dim.local_to_global_partial(tensor, dim + i)

        return tensor.flatten(dim, dim + len(self._tensor_dims) - 1)

    def global_to_local(self, tensor: "torch.Tensor", dim: int = 0, expand: bool = False) -> "torch.Tensor":
        tensor = tensor.unflatten(dim, [tensor_dim.global_size for tensor_dim in self._tensor_dims])
        for i, tensor_dim in reversed(list(enumerate(self._tensor_dims))):
            tensor = tensor_dim.global_to_local(tensor, dim + i)
        return tensor if expand else tensor.flatten(dim, dim + len(self._tensor_dims) - 1)


class ConcatenatedTensorDim(TensorDim):
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
        assert self.is_parallel
        return ConcatenatedTensorDim(
            self.name, tuple(tensor_dim.replace_parallel_dim(distributed_dim) for tensor_dim in self._tensor_dims)
        )

    def local_to_global(self, tensor: "torch.Tensor", dim: int = 0) -> "torch.Tensor":
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


scalar_dim = TensorDim("scalar", 1)
