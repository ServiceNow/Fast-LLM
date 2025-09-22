import functools
import logging
import typing

import torch

from fast_llm.core.distributed import ReduceOp
from fast_llm.core.ops import reduce_op
from fast_llm.engine.config_utils.initialization import Initialization, Initializer, LambdaInitializer
from fast_llm.engine.config_utils.tensor_dim import ConcatenatedTensorDim, TensorDim
from fast_llm.engine.distributed.config import DistributedDim, DistributedDimNames
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.functional.triton.pointwise import triton_add, triton_copy
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class _SafeTensorSliceMeta(type):
    def __instancecheck__(self, instance) -> bool:
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


def validate_tensor(tensor: torch.Tensor, other: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
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
        **kwargs,
    ):
        return super().__new__(
            cls,
            data,
        )

    @functools.cached_property
    def tensor_parallel_dim_index(self) -> int | None:
        # TODO: Avoid hard-coded assumptions on tensor parallel.
        indexes = [
            i
            for i, dim in enumerate(self.dims)
            if dim.parallel_dim is not None and dim.parallel_dim.name == DistributedDimNames.tensor
        ]
        assert len(indexes) <= 1, indexes
        return indexes[0] if indexes else None

    @functools.cached_property
    def is_tensor_parallel(self) -> bool:
        return self.tensor_parallel_dim_index is not None

    @functools.cached_property
    def tensor_parallel_size(self) -> int:
        return self.dims[self.tensor_parallel_dim_index].parallel_dim.size if self.is_tensor_parallel else 1

    @functools.cached_property
    def tensor_parallel_rank(self) -> int:
        return self.dims[self.tensor_parallel_dim_index].parallel_dim.rank if self.is_tensor_parallel else 0

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
        **kwargs: typing.Any,
    ) -> typing.Self:
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

    @property
    def global_shape(self) -> torch.Size:
        return torch.Size([dim.global_size for dim in self.dims])

    def local_to_global(self, tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Reconstruct a global tensor from its distributed slices. Support lazy-loaded safetensor slices.
        Returns a view of the input tensor (or the input tensor itself) when possible.
        """
        if tensor.ndim == 0:
            tensor = tensor[None]
        Assert.eq(tensor.shape, self.shape)
        # Tensors are always either split or duplicated in the tensor-parallel direction.
        # TODO: Avoid hard-coded assumptions on duplication
        is_first_rank, modified = True, False

        for dim, tensor_dim in enumerate(self.dims):
            if tensor_dim.is_parallel:
                tensor = tensor_dim.local_to_global(tensor, dim)
                is_first_rank &= tensor_dim.parallel_dim.rank == 0
                modified = True

        for distributed_dim, op in self._reductions:
            if distributed_dim.group is not None:
                if not modified:
                    # Avoid modifying the input in-place
                    tensor = tensor.clone()
                tensor = reduce_op(tensor, distributed_dim.group, op=op)
                is_first_rank, modified = is_first_rank and distributed_dim.group.rank() == 0, True
        Assert.eq(tensor.shape, self.global_shape)
        return tensor, is_first_rank

    def local_to_global_partial(self, tensor: torch.Tensor, fill_value: float | int = -1) -> torch.Tensor:
        """
        Construct a tensor of shape `self.global_shape` that contains its local slice at the appropriate location,
        i.e. for which `self.global_to_local(self.local_to_global_partial(tensor)) == tensor`.
        Other entries are filled with `fill_value`.
        Returns a view of the input tensor (or the input tensor itself) when possible.
        """
        if tensor.ndim == 0:
            tensor = tensor[None]
        Assert.eq(tensor.shape, self.shape)
        assert not self._reductions
        for dim, tensor_dim in enumerate(self.dims):
            if tensor_dim.is_parallel:
                tensor = tensor_dim.local_to_global_partial(tensor, dim, fill_value)

        Assert.eq(tensor.shape, self.global_shape)
        return tensor

    def global_to_local(self, tensor: torch.Tensor | SafeTensorSlice) -> torch.Tensor:
        """
        Select the local slice of a global tensor. Support lazy-loaded safetensor slices.
        Returns a view of the input tensor (or the input tensor itself) when possible.
        """
        # Take a trivial slice to convert safetensor slices.
        tensor = tensor[:]
        assert not self._reductions
        if tensor.ndim == 0:
            tensor = tensor[None]
        Assert.eq(tensor.shape, self.global_shape, msg=self)

        for dim, tensor_dim in reversed(list(enumerate(self.dims))):
            tensor = tensor_dim.global_to_local(tensor, dim)

        Assert.eq(tensor.shape, self.shape, msg=self)
        return tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # This prevents pytorch from returning broken TensorMeta instances.
        types = [torch.Tensor if issubclass(t, TensorMeta) else t for t in types]
        return torch.Tensor.__torch_function__(func, types, args, kwargs)

    @property
    def memory_usage(self) -> int:
        return self.numel() * self.element_size()

    def validate(self, tensor: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        return validate_tensor(tensor, self, device)

    def replace_tensor_parallel_dim(self, distributed_dim: DistributedDim) -> "TensorMeta":
        # Replace the tensor-parallel `DistributedDim` in `meta`.
        # Note: This will turn `ParameterMeta` into `TensorMeta`
        if not self.is_tensor_parallel:
            return self
        dims = list(self.dims)
        dims[self.tensor_parallel_dim_index] = dims[self.tensor_parallel_dim_index].replace_parallel_dim(
            distributed_dim
        )
        return TensorMeta(self, tensor_name=self.tensor_name, dims=tuple(dims), reductions=self._reductions)


class ParameterMeta(TensorMeta):
    def __init__(
        self,
        data: torch.Tensor,
        *,
        tensor_name: str = "",
        dims: tuple[TensorDim, ...],
        init_method: "Initialization | typing.Callable[[ParameterMeta, torch.Tensor, torch.Generator], None] | None" = None,
        weight_decay: bool = True,
        # Pass a list to split the parameter in contiguous (dim=0) chunks of equal size for optimization.
        lr_scale: float | None = None,
        requires_grad: bool = True,
        allow_sequence_tensor_parallel: bool = True,
        allow_no_grad: bool = False,
    ):
        super().__init__(data, tensor_name=tensor_name, dims=dims)
        if isinstance(init_method, Initialization):
            init_method = init_method.get_initializer()
        elif init_method is not None:
            # Support non-wrapped callables for convenience.
            assert callable(init_method)
            init_method = LambdaInitializer(init_method)
        self._param_init_method: Initializer | None = init_method
        self._param_weight_decay = weight_decay
        self._is_param = True
        self.param_grad_is_zero = False
        # Almost all parameters are either tensor-parallel or process tensor-sequence-parallel inputs.
        # Except for position embedding weights
        self.sequence_tensor_parallel = allow_sequence_tensor_parallel and not self.is_tensor_parallel
        # Disable the check that gradients have been computed for this parameter before the gradient reduction,
        # to support cases where gradients may not always be computed (ex. MOE layers).
        self.allow_no_grad = allow_no_grad

        self._lr_scale = lr_scale
        self.requires_grad = requires_grad and self._lr_scale != 0

    @property
    def lr_scale(self) -> float | None:
        return self._lr_scale

    @property
    def param_weight_decay(self) -> bool:
        return self._param_weight_decay

    def __repr__(self, *, tensor_contents=()) -> str:
        return super().__repr__(
            tensor_contents=(f"wd={self._param_weight_decay}", f"lr_scale={self._lr_scale}", *tensor_contents)
        )

    def init_parameter(self, tensor: torch.Tensor, distributed: Distributed) -> None:
        assert self._param_init_method is not None
        if (
            distributed.config.tensor_parallel == 1
            or distributed.config.reproducible_init
            or self._param_init_method.requires_global_initialization
        ):
            generator = distributed.pp_init_generator
        else:
            generator = distributed.tp_init_generator if self.is_tensor_parallel else distributed.pp_init_generator
        self._param_init_method(self, tensor, generator)

    @property
    def requires_global_initialization(self) -> bool:
        return self._param_init_method.requires_global_initialization

    def save(self) -> dict[str, typing.Any]:
        return {
            "name": self.tensor_name,
            "dim_names": self.dim_names,
            "shape": tuple(self.shape),
            "weight_decay": self._param_weight_decay,
            "sequence_tensor_parallel": self.sequence_tensor_parallel,
            "requires_grad": self.requires_grad,
            "tensor_parallel": self.is_tensor_parallel,
            "allow_no_grad": self.allow_no_grad,
            "lr_scale": self._lr_scale,
        }

    def load(self, state: dict[str, typing.Any]) -> None:
        current = self.save()
        Assert.eq(state, current)

    @property
    def metas_for_grad(self) -> tuple["ParameterMeta", ...]:
        return (self,)


class ConcatenatedTensorMeta(TensorMeta):
    def __init__(
        self,
        data: torch.Tensor,
        *,
        metas: tuple[ParameterMeta, ...],
        dim_index: int,
        _concatenate_check: bool = False,
        **kwargs,
    ):
        super().__init__(data, **kwargs)
        if not _concatenate_check:
            raise RuntimeError(
                f"Please instantiate {type(self).__name__} tensors through {type(self).__name__}.from_dict()"
            )
        self.metas = metas
        self.dim_index = dim_index

    @classmethod
    def from_metas(
        cls,
        metas: tuple[ParameterMeta, ...],
        *,
        tensor_name: str = "",
        dim_index: int = 0,
        dim_name: str | None = None,
        **kwargs,
    ):
        for meta in metas:
            # TODO: Support recursion?
            assert not isinstance(meta, ConcatenatedTensorMeta)
        for meta in metas[1:]:
            Assert.eq(meta.ndim, metas[0].ndim)
            for index, dim in enumerate(meta.dims):
                if index != dim_index:
                    Assert.is_(dim, metas[0].dims[index])
            Assert.eq(meta.dtype, metas[0].dtype)
            Assert.eq(meta._reductions, metas[0]._reductions)

        dims = list(metas[0].dims)
        if dim_name is None:
            dim_name = f"concatenated_{'_'.join([meta.dims[dim_index].name for meta in metas])}"
        dims[dim_index] = ConcatenatedTensorDim(dim_name, tuple(meta.dims[dim_index] for meta in metas))
        return cls.from_dims(
            tuple(dims),
            tensor_name=tensor_name,
            dtype=metas[0].dtype,
            _concatenate_check=True,
            metas=metas,
            dim_index=dim_index,
            **kwargs,
        )

    def split_tensor(self, tensor: torch.Tensor) -> list[tuple[torch.Tensor, "TensorMeta"]]:
        return [
            (tensor_, meta_)
            for tensor_, meta_ in zip(
                self.dims[self.dim_index].split_tensor(tensor, self.dim_index, global_=True), self.metas, strict=True
            )
        ]


class ConcatenatedParameterMeta(ConcatenatedTensorMeta, ParameterMeta):
    def __init__(
        self,
        data: torch.Tensor,
        *,
        split_gradients: bool = False,
        **kwargs,
    ):
        super().__init__(data, **kwargs)
        self._split_gradients = split_gradients
        if self._split_gradients:
            Assert.eq(self.dim_index, 0)

    @classmethod
    def from_metas(
        cls,
        metas: tuple[ParameterMeta, ...],
        *,
        tensor_name: str = "",
        dim_index: int = 0,
        dim_name: str | None = None,
        **kwargs,
    ):
        for meta in metas:
            assert isinstance(meta, ParameterMeta)
        split_gradients = False
        # TODO: Support more varying attributes.
        for meta in metas[1:]:
            if meta._lr_scale != metas[0]._lr_scale or meta._param_weight_decay != metas[0]._param_weight_decay:
                Assert.eq(dim_index, 0)
                split_gradients = True
            Assert.eq(meta.requires_grad, metas[0].requires_grad)
            Assert.eq(meta.sequence_tensor_parallel, metas[0].sequence_tensor_parallel)
            Assert.eq(meta.allow_no_grad, metas[0].allow_no_grad)

        return super().from_metas(
            metas,
            tensor_name=tensor_name,
            dim_index=dim_index,
            dim_name=dim_name,
            init_method=None,  # TODO
            weight_decay=metas[0]._param_weight_decay,
            lr_scale=None,
            requires_grad=metas[0].requires_grad,
            allow_sequence_tensor_parallel=metas[0].sequence_tensor_parallel,
            split_gradients=split_gradients,
            **kwargs,
        )

    @property
    def lr_scale(self) -> float | None:
        if self._split_gradients:
            raise RuntimeError()
        return super().lr_scale

    @property
    def param_weight_decay(self) -> bool:
        if self._split_gradients:
            raise RuntimeError()
        return super().param_weight_decay

    def init_parameter(self, tensor: torch.Tensor, distributed: Distributed) -> None:
        for tensor_, meta_ in self.split_tensor(tensor):
            meta_.init_parameter(tensor_, distributed)

    @property
    def requires_global_initialization(self) -> bool:
        return True

    @property
    def metas_for_grad(self) -> tuple["ParameterMeta", ...]:
        return self.metas if self._split_gradients else super().metas_for_grad


def param_get_and_unset_is_zero(param: torch.Tensor) -> bool:
    is_zero = param.param_grad_is_zero
    param.param_grad_is_zero = False
    return is_zero


def accumulate_gradient(param: torch.Tensor, grad: torch.Tensor) -> None:
    if param_get_and_unset_is_zero(param):
        triton_copy(grad, param.grad_buffer)  # noqa
    else:
        triton_add(grad, param.grad_buffer, out=param.grad_buffer)  # noqa
