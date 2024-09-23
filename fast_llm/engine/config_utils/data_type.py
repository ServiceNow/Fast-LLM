import enum
import typing

from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    import numpy as np
    import torch
    from triton import language as tl


class DataType(str, enum.Enum):
    """
    An enum to represent data types independently of third party libraries,
    so we can swap them more easily and allow for lazy imports.
    """

    float64 = "float64"
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"
    int64 = "int64"
    int32 = "int32"
    int16 = "int16"
    int8 = "int8"
    uint8 = "uint8"

    @classmethod
    def _missing_(cls, dtype: str) -> "DataType":
        # Handle alternate names and prefixes.
        Assert.custom(isinstance, dtype, str)
        dtype_split = dtype.rsplit(".", 1)
        if len(dtype_split) == 2:
            prefix, dtype = dtype_split
            Assert.incl(prefix, _KNOWN_DATA_TYPE_PREFIXES)
            return DataType(dtype)
        if dtype in _DTYPE_ALT_NAME_MAP_INV:
            return _DTYPE_ALT_NAME_MAP_INV[dtype]
        raise ValueError()

    __fast_llm_argparse_type__ = str

    @classmethod
    def from_torch(cls, dtype: "torch.dtype") -> "DataType":
        if not _TORCH_DTYPE_MAP_INV:
            _set_torch_dtype_map()
        return _TORCH_DTYPE_MAP_INV[dtype]

    @classmethod
    def from_numpy(cls, dtype: "np.dtype") -> "DataType":
        if not _NUMPY_DTYPE_MAP_INV:
            _set_numpy_dtype_map()
        return _NUMPY_DTYPE_MAP_INV[dtype]

    @classmethod
    def from_triton(cls, dtype: "tl.dtype") -> "DataType":
        if not _TRITON_DTYPE_MAP_INV:
            _set_triton_dtype_map()
        return _TRITON_DTYPE_MAP_INV[dtype]

    @property
    def torch(self) -> "torch.dtype":
        if not _TORCH_DTYPE_MAP:
            _set_torch_dtype_map()
        return _TORCH_DTYPE_MAP[self]

    @property
    def numpy(self) -> "np.dtype":
        if not _NUMPY_DTYPE_MAP:
            _set_numpy_dtype_map()
        return _NUMPY_DTYPE_MAP[self]

    @property
    def triton(self) -> "tl.dtype":
        if not _TRITON_DTYPE_MAP:
            _set_triton_dtype_map()
        return _TRITON_DTYPE_MAP[self]


_KNOWN_DATA_TYPE_PREFIXES = {"DataType", "numpy", "np", "torch", "triton.language", "tl"}

_DTYPE_ALT_NAME_MAP_INV = {
    "fp64": DataType.float64,
    "fp32": DataType.float32,
    "fp16": DataType.float16,
    "bf16": DataType.bfloat16,
}

_TORCH_DTYPE_MAP: dict[DataType, "torch.dtype"] = {}
_TORCH_DTYPE_MAP_INV: dict["torch.dtype", DataType] = {}


def _set_torch_dtype_map():
    import torch

    global _TORCH_DTYPE_MAP, _TORCH_DTYPE_MAP_INV

    _TORCH_DTYPE_MAP = {
        DataType.float64: torch.float64,
        DataType.float32: torch.float32,
        DataType.float16: torch.float16,
        DataType.bfloat16: torch.bfloat16,
        DataType.int64: torch.int64,
        DataType.int32: torch.int32,
        DataType.int16: torch.int16,
        DataType.int8: torch.int8,
        DataType.uint8: torch.uint8,
    }
    _TORCH_DTYPE_MAP_INV = {y: x for x, y in _TORCH_DTYPE_MAP.items()}


_NUMPY_DTYPE_MAP: dict[DataType, "np.dtype"] = {}
_NUMPY_DTYPE_MAP_INV: dict["np.dtype", DataType] = {}


def _set_numpy_dtype_map():
    import numpy as np

    global _NUMPY_DTYPE_MAP, _NUMPY_DTYPE_MAP_INV

    _NUMPY_DTYPE_MAP = {
        DataType.float64: np.float64,
        DataType.float32: np.float32,
        DataType.float16: np.float16,
        DataType.int64: np.int64,
        DataType.int32: np.int32,
        DataType.int16: np.int16,
        DataType.int8: np.int8,
        DataType.uint8: np.uint8,
    }
    _TORCH_DTYPE_MAP_INV = {y: x for x, y in _NUMPY_DTYPE_MAP.items()}


_TRITON_DTYPE_MAP: dict[DataType, "tl.dtype"] = {}
_TRITON_DTYPE_MAP_INV: dict["tl.dtype", DataType] = {}


def _set_triton_dtype_map():
    from triton import language as tl

    global _TRITON_DTYPE_MAP, _TRITON_DTYPE_MAP_INV

    _TRITON_DTYPE_MAP = {
        DataType.float64: tl.float64,
        DataType.float32: tl.float32,
        DataType.float16: tl.float16,
        DataType.bfloat16: tl.bfloat16,
        DataType.int64: tl.int64,
        DataType.int32: tl.int32,
        DataType.int16: tl.int16,
        DataType.int8: tl.int8,
        DataType.uint8: tl.uint8,
    }

    _TRITON_DTYPE_MAP_INV = {y: x for x, y in _TRITON_DTYPE_MAP.items() if y is not None}
