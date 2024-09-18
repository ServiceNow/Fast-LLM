import dataclasses
import enum

import torch
import torch.nn
from triton import language as tl


class TritonConfig:
    # A global switch for triton kernels.
    # TODO: Improve.
    TRITON_ENABLED = True
    TRITON_LINEAR = False
    # A block size of 1024 is usually optimal for large pointwise kernels
    POINTWISE_BLOCK_SIZE = 1024
    MAX_BLOCK_SIZE_BYTES = 65536

    DTYPE_MAP = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.int16: tl.int16,
        torch.int32: tl.int32,
        torch.int64: tl.int64,
    }


class MLPRecomputeLevel(str, enum.Enum):
    none = "none"
    activation = "activation"
    activation_and_input = "activation_and_input"
    full = "full"

    @property
    def recompute_layer_1(self):
        return self == MLPRecomputeLevel.full

    @property
    def recompute_activation(self):
        return self != MLPRecomputeLevel.none

    @property
    def recompute_sparse_input(self):
        return self in (MLPRecomputeLevel.full, MLPRecomputeLevel.activation_and_input)


class ActivationType(str, enum.Enum):
    """
    An enum for the available activation types for the MLP layer.
    """

    gelu = "gelu"
    silu = "silu"
    relu = "relu"
    squared_relu = "squared_relu"

    @property
    def activation_fn(self):
        return _ACTIVATION_FN_MAP[self]

    @property
    def hf_name(self):
        return _ACTIVATION_HF_NAMES[self]

    @classmethod
    def from_hf_name(cls, hf_name) -> "ActivationType":
        return _ACTIVATION_HF_NAMES_INV[hf_name]


_ACTIVATION_FN_MAP = {
    ActivationType.gelu: lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
    ActivationType.silu: torch.nn.functional.silu,
    ActivationType.relu: torch.nn.functional.relu,
    ActivationType.squared_relu: lambda x: torch.pow(torch.nn.functional.relu(x), 2),
}
_ACTIVATION_HF_NAMES = {
    ActivationType.gelu: "gelu_pytorch_tanh",
    ActivationType.silu: "silu",
    ActivationType.relu: "relu",
    ActivationType.squared_relu: "relu2",
}
_ACTIVATION_HF_NAMES_INV = {value: key for key, value in _ACTIVATION_HF_NAMES.items()}


@dataclasses.dataclass()
class SparseMap:
    sparse_rows: torch.Tensor
    expert_ends: torch.Tensor
    expert_pad_begins: torch.Tensor
    num_rows_dense: int
    num_rows: int
    num_rows_unpadded: int
    num_experts: int
    num_experts_per_token: int


MAX_DROPLESS_BLOCK_SIZE_ROW = 128
