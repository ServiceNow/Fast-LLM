import torch

from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.functional.config import ActivationType
from fast_llm.tensor import ParameterMeta, TensorMeta

try:
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn  # noqa

    _causal_conv1d_available = True
except (ImportError, RuntimeError):
    _causal_conv1d_available = False


class CausalConv1d(torch.nn.Module):
    """
    TODO: Generalize to other convolutions?
    """

    def __init__(
        self,
        weight: ParameterMeta,
        bias: ParameterMeta | None,
        *,
        activation: ActivationType = ActivationType.identity,
    ):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self._activation = activation
        self.forward = (
            self._forward_causal_conv1d
            if _causal_conv1d_available and self._activation in (ActivationType.identity, ActivationType.silu)
            else self._forward_torch
        )

    def _forward_torch(self, input_: torch.Tensor) -> torch.Tensor:
        return self._activation.activation_fn(
            torch.nn.functional.conv1d(
                input_,
                self.weight,
                bias=self.bias,
                groups=self.weight.size(0),
                padding=self.weight.size(2) - 1,
            )[..., : input_.size(1)]
        )

    def _forward_causal_conv1d(self, input_: torch.Tensor) -> torch.Tensor:
        return _causal_conv1d_fn(
            input_,
            self.weight.squeeze(1),
            self.bias,
            activation=(None if self._activation == ActivationType.identity else self._activation.value),
        )

    def get_compute_usage(self, input_: TensorMeta, config: ResourceUsageConfig) -> int:
        raise NotImplementedError()
