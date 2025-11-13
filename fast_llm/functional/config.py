import enum
import typing

if typing.TYPE_CHECKING:
    import torch


class TritonConfig:
    # A global switch for triton kernels.
    # TODO: Improve.
    TRITON_ENABLED = True
    TRITON_LINEAR = False
    # A block size of 1024 is usually optimal for large pointwise kernels
    POINTWISE_BLOCK_SIZE = 1024
    MAX_BLOCK_SIZE_BYTES = 65536


class MLPRecomputeLevel(enum.StrEnum):
    none = "none"
    activation = "activation"
    activation_and_input = "activation_and_input"
    full = "full"

    @property
    def recompute_layer_1(self) -> bool:
        return self == MLPRecomputeLevel.full

    @property
    def recompute_activation(self) -> bool:
        return self != MLPRecomputeLevel.none

    @property
    def recompute_sparse_input(self) -> bool:
        return self in (MLPRecomputeLevel.full, MLPRecomputeLevel.activation_and_input)


class ActivationType(enum.StrEnum):
    """
    An enum for the available activation types for the MLP layer.
    """

    gelu = "gelu"
    silu = "silu"
    relu = "relu"
    squared_relu = "squared_relu"
    identity = "identity"
    gpt_oss_glu = "gpt_oss_glu"  # Custom GLU for GPT-OSS: (up + 1) * (gate * sigmoid(gate * 1.702))

    @property
    def activation_fn(self) -> typing.Callable[["torch.Tensor"], "torch.Tensor"]:
        if not _ACTIVATION_FN_MAP:
            _set_activation_fn_map()
        return _ACTIVATION_FN_MAP[self]

    @property
    def hf_name(self) -> str:
        return _ACTIVATION_HF_NAMES[self]

    @classmethod
    def from_hf_name(cls, hf_name) -> typing.Self:
        return _ACTIVATION_HF_NAMES_INV[hf_name]


def _set_activation_fn_map() -> None:
    import torch
    import torch.nn

    global _ACTIVATION_FN_MAP

    def gpt_oss_glu_activation(x: torch.Tensor) -> torch.Tensor:
        # Custom GPT-OSS GLU: (up + 1) * (gate * sigmoid(gate * 1.702))
        # Input x has shape [..., 2*dim] where first half is gate, second half is up
        # Includes clamping: gate max 7.0, up in [-7.0, 7.0]
        gate, up = x.chunk(2, dim=-1)
        alpha = 1.702
        limit = 7.0
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        glu = gate * torch.sigmoid(gate * alpha)
        return (up + 1.0) * glu

    _ACTIVATION_FN_MAP = {
        ActivationType.gelu: lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
        ActivationType.silu: torch.nn.functional.silu,
        ActivationType.relu: torch.nn.functional.relu,
        ActivationType.squared_relu: lambda x: torch.pow(torch.nn.functional.relu(x), 2),
        ActivationType.identity: lambda x: x,
        ActivationType.gpt_oss_glu: gpt_oss_glu_activation,
    }


_ACTIVATION_FN_MAP: dict[ActivationType, typing.Callable[["torch.Tensor"], "torch.Tensor"]] = {}

_ACTIVATION_HF_NAMES = {
    ActivationType.gelu: "gelu_pytorch_tanh",
    ActivationType.silu: "silu",
    ActivationType.relu: "relu",
    ActivationType.squared_relu: "relu2",
    ActivationType.identity: "identity",
    ActivationType.gpt_oss_glu: "gpt_oss_glu",  # Custom activation for GPT-OSS
}
_ACTIVATION_HF_NAMES_INV = {value: key for key, value in _ACTIVATION_HF_NAMES.items()}

MAX_DROPLESS_BLOCK_SIZE_ROW = 128


class CrossEntropyImpl(str, enum.Enum):
    auto = "auto"
    torch = "torch"
    fused = "fused"
    triton = "triton"


class DistillationLossImpl(str, enum.Enum):
    reverse_kl = "reverse_kl"
    cross_entropy = "cross_entropy"


class TargetFormat(enum.StrEnum):
    labels = "labels"
    logits = "logits"
    probabilities = "probabilities"
