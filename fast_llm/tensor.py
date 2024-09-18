import math
import typing

import torch
import torch._dynamo  # noqa

from fast_llm.core.distributed import ReduceOp
from fast_llm.core.ops import gather_op, reduce_op
from fast_llm.distributed import Distributed, DistributedConfig, DistributedDim, DistributedDimNames
from fast_llm.functional.triton.pointwise import triton_add, triton_copy
from fast_llm.utils import Assert, div


class _SafeTensorSliceMeta(type):
    def __instancecheck__(self, instance):
        # Good enough for our purpose.
        return not isinstance(instance, torch.Tensor)


class SafeTensorSlice(metaclass=_SafeTensorSliceMeta):
    """
    A mock class for safetensors slices since the actual class is not exposed.
    TODO: Find the actual class.
    """

    def __init__(self):
        raise NotImplementedError()

    def get_shape(self) -> list[int]:
        pass

    def __getitem__(self, item) -> torch.Tensor:
        pass


class TensorDim:
    def __init__(self, name: str, global_size: int | None, parallel_dim: DistributedDim | None = None):
        # TODO: Handle None for unknown sizes?
        self._name = name
        self._global_size = global_size
        self._size = self._global_size if parallel_dim is None else div(global_size, parallel_dim.size)
        self._parallel_dim = parallel_dim

    def __repr__(self):
        return (
            f"TensorDim("
            f"name={self._name},"
            f" size={self._size},"
            f" global_size={self._global_size},"
            f" parallel_dim={None if self.parallel_dim is None else self._parallel_dim}"
            f")"
        )

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size

    @property
    def expanded_shape(self):
        return torch.Size([self._size])

    @property
    def ndim(self):
        return 1

    @property
    def global_size(self):
        return self._global_size

    @property
    def global_expanded_shape(self):
        return torch.Size([self._size if self._parallel_dim is None else self._size * self._parallel_dim.size])

    @property
    def parallel_dim(self):
        return self._parallel_dim

    @property
    def parallel_dim_index(self):
        return None if self._parallel_dim is None else 0

    @property
    def parallel_group(self):
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
    def dims(self):
        return self._dims

    @property
    def ndim(self):
        return sum(dim.ndim for dim in self._dims)

    @property
    def expanded_shape(self):
        return sum((dim.expanded_shape for dim in self._dims), torch.Size())

    @property
    def global_expanded_shape(self):
        return sum((dim.global_expanded_shape for dim in self._dims), torch.Size())

    @property
    def parallel_dim_index(self):
        return self._parallel_dim_index


class DefaultDimNames:
    # Scalar
    scalar = "scalar"


class TensorSpace:
    _is_setup: bool = False
    _distributed: Distributed

    def __init__(self, distributed_config: DistributedConfig):
        self._distributed_config = distributed_config.validate()
        self._tensor_dims: dict[str, TensorDim] = {}
        self.add_tensor_dim(TensorDim(DefaultDimNames.scalar, 1))

    def setup(self, distributed: Distributed):
        assert distributed.config is self._distributed_config
        assert not self._is_setup
        self._is_setup = True
        self._distributed = distributed

    @property
    def distributed_config(self):
        return self._distributed_config

    @property
    def distributed(self):
        assert self._is_setup
        return self._distributed

    def add_tensor_dim(self, dim: TensorDim):
        if isinstance(dim, CompositeTensorDim):
            for dim_ in dim.dims:
                Assert.incl(dim_.name, self._tensor_dims)
                Assert.eq(dim_, self._tensor_dims[dim_.name])
        if dim.name in self._tensor_dims:
            Assert.eq(dim, self._tensor_dims[dim.name])
        else:
            if dim.parallel_dim is not None:
                assert dim.parallel_dim.name in self._distributed_config.distributed_dims, dim.parallel_dim.name
                Assert.eq(dim.parallel_dim, self._distributed_config.distributed_dims[dim.parallel_dim.name])
            self._tensor_dims[dim.name] = dim

    def get_tensor_dim(self, name: str):
        return self._tensor_dims[name]


def validate_tensor(tensor: torch.Tensor, other: torch.Tensor, device: torch.device | None = None):
    Assert.custom(isinstance, tensor, torch.Tensor)
    Assert.eq(tensor.shape, other.shape)
    Assert.eq(tensor.dtype, other.dtype)
    if device is not None:
        Assert.eq(tensor.device, device)
    elif other.device.type != "meta":
        Assert.eq(tensor.device, other.device)
    return tensor


class TensorMeta(torch.Tensor):
    """
    A subclass for tensor metadata.
    """

    def __init__(
        self,
        # The actual tensor (must be on the `meta` device)
        data: torch.Tensor,
        *,
        # A name for the tensor, for identification and debugging.
        tensor_name: str,
        dims: tuple[TensorDim, ...],
        # Reductions to be applied to reconstruct the global tensor.
        reductions: tuple[tuple[DistributedDim, ReduceOp], ...] = (),
    ):
        # The tensor is already initialized, this is object.__init__
        super().__init__()
        Assert.eq(data.device.type, "meta")
        Assert.eq(self.shape, tuple(dim.size for dim in dims))
        self.tensor_name = tensor_name
        self.dims = dims
        # The `names` attribute is half-implemented and breaks things, so we use `dim_names` instead.
        self.dim_names = tuple(dim.name for dim in dims)
        self._reductions = reductions
        for dim, op in reductions:
            assert isinstance(dim, DistributedDim), dim

    def __new__(
        cls,
        data: torch.Tensor,
        *,
        tensor_name: str,
        dims: tuple[TensorDim, ...],
        reductions: tuple[tuple[DistributedDim, ReduceOp], ...] = (),
    ):
        return super().__new__(
            cls,
            data,
        )

    @property
    def is_tensor_parallel(self):
        # TODO: Avoid hard-coded assumptions on tensor parallel.
        return any(
            dim.parallel_dim is not None and dim.parallel_dim.name == DistributedDimNames.tensor for dim in self.dims
        )

    def __repr__(self, *, tensor_contents=()):
        return super().__repr__(
            tensor_contents=", ".join((self.tensor_name, f"dims={self.dim_names}", *tensor_contents))
        )

    @classmethod
    def from_dims(
        cls,
        dims: tuple[TensorDim, ...],
        *,
        tensor_name: str = "",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        return cls(
            torch.empty(
                [dim.size for dim in dims],
                device="meta",
                dtype=dtype,
            ),
            tensor_name=tensor_name,
            dims=dims,
            **kwargs,
        )

    @classmethod
    def from_tensor_space(
        cls,
        dim_names: tuple[str, ...],
        tensor_space: TensorSpace,
        *,
        tensor_name: str = "",
        dtype: torch.dtype = torch.float32,
        reductions: tuple[tuple[str, ReduceOp], ...] = (),
        **kwargs,
    ):
        dims = tuple(tensor_space.get_tensor_dim(dim_name) for dim_name in dim_names)
        if reductions:
            # kwarg not available for ParameterMeta, so we only provide if necessary.
            kwargs["reductions"] = tuple(
                (tensor_space.distributed_config.get_distributed_dim(name), op) for name, op in reductions
            )
        return cls.from_dims(dims, tensor_name=tensor_name, dtype=dtype, **kwargs)

    @property
    def global_shape(self):
        return torch.Size([dim.global_size for dim in self.dims])

    def local_to_global(
        self,
        tensor: torch.Tensor,
        *,
        distributed: Distributed,
    ):
        # Tensors are always either split or duplicated in the tensor-parallel direction.
        # TODO: Avoid hard-coded assumptions on duplication
        is_first_rank = distributed.config.tensor_rank == 0
        modified = False
        for i, dim in enumerate(self.dims):
            if dim.parallel_group is not None:
                tensor = gather_op(
                    tensor.unflatten(i, dim.expanded_shape), dim.parallel_group, i + dim.parallel_dim_index
                ).flatten(i, i + len(dim.expanded_shape) - 1)
                is_first_rank, modified = is_first_rank and dim.parallel_group.rank() == 0, True

        for distributed_dim, op in self._reductions:
            if distributed_dim.group is not None:
                if not modified:
                    # Avoid modifying the input in-place
                    tensor = tensor.clone()
                tensor = reduce_op(tensor, distributed_dim.group, op=op)
                is_first_rank, modified = is_first_rank and distributed_dim.group.rank() == 0, True
        return tensor, is_first_rank

    def global_to_local(
        self,
        tensor: torch.Tensor | SafeTensorSlice,
    ):
        """
        Recover the tensor-parallel slice of a tensor. Support lazy-loaded safetensor slices.
        """
        # Take a trivial slice to convert safetensor slices.
        tensor_ = tensor[:]
        assert not self._reductions

        for i, dim in enumerate(self.dims):
            if dim.parallel_dim is not None and dim.parallel_dim.size > 1:
                tensor_ = (
                    tensor_.unflatten(i, dim.global_expanded_shape)
                    .chunk(dim.parallel_dim.size, i + dim.parallel_dim_index)[dim.parallel_dim.rank]
                    .flatten(i, i + len(dim.expanded_shape) - 1)
                )

        return tensor_.view(self.shape)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # This prevents pytorch from returning broken TensorMeta instances.
        types = [torch.Tensor if issubclass(t, TensorMeta) else t for t in types]
        return torch.Tensor.__torch_function__(func, types, args, kwargs)

    @property
    def memory_usage(self):
        return self.numel() * self.element_size()

    def validate(self, tensor: torch.Tensor, device: torch.device | None = None):
        return validate_tensor(tensor, self, device)


class ParameterMeta(TensorMeta):
    def __init__(
        self,
        data: torch.Tensor,
        *,
        tensor_name: str = "",
        dims: tuple[TensorDim, ...],
        init_method: typing.Callable[["ParameterMeta", torch.Tensor, torch.Generator], torch.Tensor] | None = None,
        weight_decay: bool = True,
        # Pass a list to split the parameter in contiguous (dim=0) chunks of equal size for optimization.
        lr_scale: float | None | tuple[float | None, ...] = None,
        requires_grad: bool = True,
        allow_sequence_tensor_parallel: bool = True,
        auto_grad_accumulation: bool = True,
        allow_no_grad: bool = False,
    ):
        super().__init__(data, tensor_name=tensor_name, dims=dims)
        self.param_init_method = init_method
        self.param_weight_decay = weight_decay
        self._is_param = True
        self.param_grad_is_zero = False
        self.requires_grad = requires_grad
        # Almost all parameters are either tensor-parallel or process tensor-sequence-parallel inputs.
        # Except for position embedding weights
        self.sequence_tensor_parallel = allow_sequence_tensor_parallel and not self.is_tensor_parallel
        # If true, grad accumulation is handled automatically by copying or adding to the grad_buffer.
        # Can be disabled to allow for a more efficient implementation that accumulates directly to it.
        self.auto_grad_accumulation = auto_grad_accumulation
        # Disable the check that gradients have been computed for this parameter before the gradient reduction,
        # to support cases where gradients may not always be computed (ex. MOE layers).
        self.allow_no_grad = allow_no_grad

        self.lr_scale = lr_scale if isinstance(lr_scale, tuple) else (lr_scale,)
        # Ensure the parameter is split in chunks of equal size.
        Assert.multiple(self.dims[0].size, len(self.lr_scale))

    def __new__(
        cls,
        data: torch.Tensor,
        *,
        tensor_name: str = "",
        dims: tuple[TensorDim, ...],
        init_method: typing.Callable,
        weight_decay: bool = True,
        lr_scale: float | None | tuple[float | None, ...] = None,
        allow_sequence_tensor_parallel: bool = True,
        auto_grad_accumulation: bool = True,
        allow_no_grad: bool = False,
    ):
        return super().__new__(
            cls,
            data,
            tensor_name=tensor_name,
            dims=dims,
        )

    def __repr__(self, *, tensor_contents=()):
        return super().__repr__(
            tensor_contents=(f"wd={self.param_weight_decay}", f"lr_scale={self.lr_scale}", *tensor_contents)
        )

    def init_parameter(self, tensor: torch.Tensor, distributed: Distributed):
        assert self.param_init_method is not None
        if distributed.config.tensor_parallel == 1 or distributed.config.reproducible_init:
            generator = distributed.pp_init_generator
        else:
            generator = distributed.tp_init_generator if self.is_tensor_parallel else distributed.pp_init_generator
        self.param_init_method(self, tensor, generator)

    def save(self):
        return {
            "name": self.tensor_name,
            "dim_names": self.dim_names,
            "shape": tuple(self.shape),
            "weight_decay": self.param_weight_decay,
            "sequence_tensor_parallel": self.sequence_tensor_parallel,
            "requires_grad": self.requires_grad,
            "tensor_parallel": self.is_tensor_parallel,
            "allow_no_grad": self.allow_no_grad,
            "lr_scale": self.lr_scale,
        }

    def load(self, state):
        current = self.save()
        Assert.eq(state, current)


def param_get_and_unset_is_zero(param: torch.Tensor):
    is_zero = param.param_grad_is_zero
    param.param_grad_is_zero = False
    return is_zero


def accumulate_gradient(param: torch.Tensor, grad: torch.Tensor):
    if param_get_and_unset_is_zero(param):
        triton_copy(grad, param.grad_buffer)  # noqa
    else:
        triton_add(grad, param.grad_buffer, out=param.grad_buffer)  # noqa


def init_fill_(value):
    def init_(meta: ParameterMeta, tensor: torch.Tensor, generator: torch.Generator):  # noqa
        return tensor.fill_(value)

    return init_


init_zeros_ = init_fill_(0.0)
init_ones_ = init_fill_(1.0)


def init_normal_(mean=0.0, std=1.0):
    def init_(meta: ParameterMeta, tensor: torch.Tensor, generator: torch.Generator):  # noqa
        return tensor.normal_(mean, std, generator=generator)

    return init_


def init_uniform_(low=0.0, high=1.0):
    def init_(meta: ParameterMeta, tensor: torch.Tensor, generator: torch.Generator):  # noqa
        return tensor.uniform_(low, high, generator=generator)  # noqa

    return init_
