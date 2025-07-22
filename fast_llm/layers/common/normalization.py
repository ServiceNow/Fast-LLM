import torch

from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.engine.config_utils.tensor_space import TensorDim
from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.normalization import triton_normalization_autograd
from fast_llm.layers.common.config import NormalizationImplementation
from fast_llm.tensor import ParameterMeta, accumulate_gradient, init_ones_, init_zeros_
from fast_llm.utils import Assert

try:
    import fused_layer_norm_cuda  # noqa

    _fused_normalization_available = True
except ImportError:
    _fused_normalization_available = False

try:
    import fast_layer_norm  # noqa

    _fast_normalization_available = True
except ImportError:
    _fast_normalization_available = False


_PERSIST_LN_SIZES = (
    1024,
    1536,
    2048,
    2304,
    3072,
    3840,
    4096,
    5120,
    6144,
    8192,
    10240,
    12288,
    12800,
    15360,
    16384,
    18432,
    20480,
    24576,
    25600,
    30720,
    32768,
    40960,
    49152,
    65536,
)


class FastLayerNorm(torch.autograd.Function):
    """
    The fast layer normalization implementation from `apex.contrib`.
    Faster than `FusedLayerNorm`, but doesn't support all layer widths.
    TODO: Move to functional.
    """

    @staticmethod
    def forward(
        ctx, input_: torch.Tensor, normalized_shape: torch.Size, weight: torch.Tensor, bias: torch.Tensor, eps: float
    ) -> torch.Tensor:  # noqa
        assert _fast_normalization_available
        Assert.incl(normalized_shape.numel(), _PERSIST_LN_SIZES)
        output, _, inv_var = fast_layer_norm.ln_fwd(input_, weight, bias, eps)
        ctx.save_for_backward(output, weight, bias, inv_var)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:  # noqa
        output, weight, bias, inv_var = ctx.saved_tensors
        # TODO: Gradients may be computed unnecessarily.
        grad_input, grad_weight, grad_bias, _, _ = fast_layer_norm.ln_bwd(
            grad_output, output, None, inv_var, weight, bias, True
        )
        if weight.requires_grad:
            accumulate_gradient(weight, grad_weight)
        if bias.requires_grad:
            accumulate_gradient(bias, grad_bias)
        return grad_input, None, None, None, None


class FusedLayerNorm(torch.autograd.Function):
    """
    The fused layer normalization implementation from `apex`.
    Faster than the stock pytorch implementation, supports all layer widths.
    TODO: Move to functional.
    """

    @staticmethod
    def forward(
        ctx, input_: torch.Tensor, normalized_shape: torch.Size, weight: torch.Tensor, bias: torch.Tensor, eps: float
    ) -> torch.Tensor:  # noqa
        assert _fused_normalization_available
        ctx.eps = eps
        ctx.normalized_shape = normalized_shape
        output, _, inv_var = fused_layer_norm_cuda.forward_affine(input_, normalized_shape, weight, bias, eps)
        ctx.save_for_backward(output, weight, bias, inv_var)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:  # noqa
        output, weight, bias, inv_var = ctx.saved_tensors
        # TODO: Gradients may be computed unnecessarily.
        grad_input, grad_weight, grad_bias = fused_layer_norm_cuda.backward_affine(
            grad_output, None, inv_var, output, ctx.normalized_shape, weight, bias, ctx.eps, True
        )
        if weight.requires_grad:
            accumulate_gradient(weight, grad_weight)
        if bias.requires_grad:
            accumulate_gradient(bias, grad_bias)
        return grad_input, None, None, None, None


class FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input_: torch.Tensor, normalized_shape: torch.Size, weight: torch.Tensor, eps: float
    ) -> torch.Tensor:  # noqa
        assert _fused_normalization_available
        ctx.eps = eps
        ctx.normalized_shape = normalized_shape
        output, inv_var = fused_layer_norm_cuda.rms_forward_affine(input_, normalized_shape, weight, eps)
        ctx.save_for_backward(output, weight, inv_var)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:  # noqa
        output, weight, inv_var = ctx.saved_tensors
        # TODO: Gradients may be computed unnecessarily.
        grad_input, grad_weight = fused_layer_norm_cuda.rms_backward_affine(
            grad_output.contiguous(), inv_var, output, ctx.normalized_shape, weight, ctx.eps, True
        )
        if weight.requires_grad:
            accumulate_gradient(weight, grad_weight)
        return grad_input, None, None, None


class LayerNorm(torch.nn.Module):
    """
    A layer normalization layer, supporting multiple implementations.
    Note: Converting input automatically to training dtype to match Apex behaviour,
     needed for full precision residual.
    TODO: Review this?
    """

    def __init__(
        self,
        hidden_dim: TensorDim,
        *,
        eps=1e-5,
        implementation: NormalizationImplementation = NormalizationImplementation.auto,
        weight_init_method=None,
        bias_init_method=init_zeros_,
        zero_centered: bool = False,
        lr_scale: float | None = None,
    ):
        super().__init__()
        assert not hidden_dim.is_parallel
        self._eps = eps
        self._zero_centered = zero_centered
        if implementation == NormalizationImplementation.auto:
            if _fast_normalization_available and hidden_dim.size in _PERSIST_LN_SIZES and not self._zero_centered:
                implementation = NormalizationImplementation.fast
            elif TritonConfig.TRITON_ENABLED or self._zero_centered:
                log_main_rank("Fast layer norm unavailable, using backup triton implementation.")
                implementation = NormalizationImplementation.triton
            elif _fused_normalization_available:
                log_main_rank("Fast layer norm unavailable, using backup fused implementation.")
                implementation = NormalizationImplementation.fused
            else:
                log_main_rank("Fast and fused layer norm unavailable, using backup pytorch implementation.")
                implementation = NormalizationImplementation.torch
        if self._zero_centered:
            assert implementation == NormalizationImplementation.triton
        if implementation == NormalizationImplementation.triton:
            self._forward = self._forward_triton
        elif implementation == NormalizationImplementation.fast:
            self._forward = self._forward_fast
        elif implementation == NormalizationImplementation.fused:
            self._forward = self._forward_fused
        elif implementation == NormalizationImplementation.torch:
            self._forward = self._forward_torch
        else:
            raise NotImplementedError(implementation)

        if weight_init_method is None:
            weight_init_method = init_zeros_ if self._zero_centered else init_ones_

        self.weight = ParameterMeta.from_dims(
            (hidden_dim,),
            init_method=weight_init_method,
            weight_decay=False,
            auto_grad_accumulation=implementation == NormalizationImplementation.torch,
            lr_scale=lr_scale,
        )
        self.bias = ParameterMeta.from_dims(
            (hidden_dim,),
            init_method=bias_init_method,
            weight_decay=False,
            auto_grad_accumulation=implementation == NormalizationImplementation.torch,
            lr_scale=lr_scale,
        )
        self.normalized_shape = self.weight.shape

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self._forward(input_.view(-1, *self.normalized_shape)).view_as(input_)

    def _forward_triton(self, input_: torch.Tensor) -> torch.Tensor:
        return triton_normalization_autograd(
            input_, self.weight, self.bias, self._eps, self.training, self._zero_centered
        )

    def _forward_fast(self, input_: torch.Tensor) -> torch.Tensor:
        return FastLayerNorm.apply(input_, self.normalized_shape, self.weight, self.bias, self._eps)

    def _forward_fused(self, input_: torch.Tensor) -> torch.Tensor:
        return FusedLayerNorm.apply(input_, self.normalized_shape, self.weight, self.bias, self._eps)

    def _forward_torch(self, input_: torch.Tensor) -> torch.Tensor:
        return torch.layer_norm(input_.to(self.weight.dtype), self.normalized_shape, self.weight, self.bias, self._eps)


class RMSNorm(torch.nn.Module):
    """
    A RMS normalization layer.
    Note: Converting input automatically to training dtype to match Apex behaviour,
     needed for full precision residual.
    TODO: Review this?
    """

    def __init__(
        self,
        hidden_dim: TensorDim,
        *,
        eps=1e-5,
        implementation: NormalizationImplementation = NormalizationImplementation.auto,
        weight_init_method=None,
        zero_centered: bool = False,
        lr_scale: float | None = None,
    ):
        super().__init__()
        assert not hidden_dim.is_parallel
        self._eps = eps
        self._zero_centered = zero_centered
        if implementation == NormalizationImplementation.auto:
            if TritonConfig.TRITON_ENABLED or self._zero_centered:
                implementation = NormalizationImplementation.triton
            elif _fused_normalization_available:
                log_main_rank("Triton RMS norm unavailable, using fused implementation.")
                implementation = NormalizationImplementation.fused
            else:
                log_main_rank("Fused RMS norm unavailable, using backup implementation.")
                implementation = NormalizationImplementation.torch
        if self._zero_centered:
            assert implementation == NormalizationImplementation.triton
        if implementation == NormalizationImplementation.triton:
            self._forward = self._forward_triton
        elif implementation == NormalizationImplementation.torch:
            self._forward = self._forward_torch
        elif implementation == NormalizationImplementation.fused:
            self._forward = self._forward_fused
        else:
            raise NotImplementedError(implementation)

        if weight_init_method is None:
            weight_init_method = init_zeros_ if self._zero_centered else init_ones_

        self.weight = ParameterMeta.from_dims(
            (hidden_dim,),
            init_method=weight_init_method,
            weight_decay=False,
            auto_grad_accumulation=True,
            lr_scale=lr_scale,
        )
        self.normalized_shape = self.weight.shape

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self._forward(input_.view(-1, *self.normalized_shape)).view_as(input_)

    def _forward_triton(self, input_: torch.Tensor) -> torch.Tensor:
        return triton_normalization_autograd(input_, self.weight, None, self._eps, self.training, self._zero_centered)

    def _forward_fused(self, input_: torch.Tensor) -> torch.Tensor:
        return FusedRMSNorm.apply(input_, self.normalized_shape, self.weight, self._eps)

    def _forward_torch(self, input_: torch.Tensor) -> torch.Tensor:
        return torch.rms_norm(input_.to(self.weight.dtype), self.normalized_shape, self.weight, self._eps)
