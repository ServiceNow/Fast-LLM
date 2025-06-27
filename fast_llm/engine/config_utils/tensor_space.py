import math
import typing

from fast_llm.engine.distributed.config import DistributedConfig, DistributedDim
from fast_llm.utils import Assert, div

if typing.TYPE_CHECKING:
    from fast_llm.core.distributed import ProcessGroup
    from fast_llm.engine.distributed.distributed import Distributed


class TensorDim:
    def __init__(self, name: str, global_size: int | None, parallel_dim: DistributedDim | None = None):
        # TODO: Handle None for unknown sizes?
        self._name = name
        self._global_size = global_size
        self._size = self._global_size if parallel_dim is None else div(global_size, parallel_dim.size)
        self._parallel_dim = parallel_dim

    def __repr__(self) -> str:
        return (
            f"TensorDim("
            f"name={self._name},"
            f" size={self._size},"
            f" global_size={self._global_size},"
            f" parallel_dim={None if self.parallel_dim is None else self._parallel_dim}"
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
    def expanded_shape(self) -> tuple[int, ...]:
        return (self._size,)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def global_size(self) -> int:
        return self._global_size

    @property
    def global_expanded_shape(self) -> tuple[int, ...]:
        return (self._size if self._parallel_dim is None else self._size * self._parallel_dim.size,)

    @property
    def parallel_dim(self) -> DistributedDim | None:
        return self._parallel_dim

    @property
    def parallel_dim_index(self) -> int | None:
        return None if self._parallel_dim is None else 0

    @property
    def parallel_group(self) -> "ProcessGroup|None":
        return None if self._parallel_dim is None else self._parallel_dim.group


class CompositeTensorDim(TensorDim):
    def __init__(self, name: str, dims: tuple[TensorDim, ...]):
        # TODO: Recursive composition??
        parallel_dims = [(i, dim.parallel_dim) for i, dim in enumerate(dims) if dim.parallel_dim]
        Assert.leq(len(parallel_dims), 1)

        super().__init__(
            name=name,
            global_size=math.prod(dim.global_size for dim in dims),
            parallel_dim=parallel_dims[0][1] if parallel_dims else None,
        )
        self._dims = dims
        self._parallel_dim_index = (
            sum(dim.ndim for dim in self._dims[: parallel_dims[0][0]])
            + self._dims[parallel_dims[0][0]].parallel_dim_index
            if parallel_dims
            else None
        )

    @property
    def dims(self) -> tuple[TensorDim, ...]:
        return self._dims

    @property
    def ndim(self) -> int:
        return sum(dim.ndim for dim in self._dims)

    @property
    def expanded_shape(self) -> tuple[int, ...]:
        return sum((dim.expanded_shape for dim in self._dims), ())

    @property
    def global_expanded_shape(self) -> tuple[int, ...]:
        return sum((dim.global_expanded_shape for dim in self._dims), ())

    @property
    def parallel_dim_index(self) -> int | None:
        return self._parallel_dim_index


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

    def add_tensor_dim(self, dim: TensorDim) -> None:
        if isinstance(dim, CompositeTensorDim):
            for dim_ in dim.dims:
                Assert.incl(dim_.name, self._tensor_dims)
                Assert.eq(dim_, self._tensor_dims[dim_.name])
        if dim.name in self._tensor_dims:
            Assert.eq(dim, self._tensor_dims[dim.name])
        else:
            if dim.parallel_dim is not None:
                assert dim.parallel_dim.name in self._distributed_config.distributed_dims, dim.parallel_dim.name
                Assert.eq(
                    dim.parallel_dim.__dict__,
                    self._distributed_config.distributed_dims[dim.parallel_dim.name].__dict__,
                )
            self._tensor_dims[dim.name] = dim

    def get_tensor_dim(self, name: str) -> TensorDim:
        return self._tensor_dims[name]
