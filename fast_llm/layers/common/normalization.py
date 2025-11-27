import torch
from torch.distributed import ProcessGroup, ReduceOp, all_reduce  # noqa

from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.engine.config_utils.tensor_space import TensorDim
from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.normalization import _layer_norm_bwd, _layer_norm_fwd, triton_normalization_autograd
from fast_llm.layers.common.config import NormalizationImplementation, TPRMSNormImplementation
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


class MambaRMSNormGated(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: TensorDim,
        *,
        eps=1e-5,
        implementation: NormalizationImplementation = TPRMSNormImplementation.autograd_redstats,
        weight_init_method=None,
        zero_centered: bool = False,
        lr_scale: float | None = None,
        group_size: int | None = None,
        norm_before_gate: bool = True,
    ):
        """
        RMS Norm per group of size group_size
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate

        self._eps = eps
        self._zero_centered = zero_centered

        if weight_init_method is None:
            weight_init_method = init_zeros_ if self._zero_centered else init_ones_
        # self._forward = self._forward_distributed

        self.weight = ParameterMeta.from_dims(  # local weights
            (hidden_dim,),
            init_method=weight_init_method,
            weight_decay=False,
            auto_grad_accumulation=True,
            lr_scale=lr_scale,
        )
        self.normalized_shape = self.weight.shape

    # def forward(self, input_: torch.Tensor) -> torch.Tensor:
    #     return self._forward(input_)

    def forward(self, input_: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        return LayerNormFn.apply(
            input_, self.weight, None, gate, self._eps, self.group_size, self.norm_before_gate, True
        )


class LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, is_rms_norm=False):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""

        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if z is not None:
            assert z.shape == x_shape_og
            z = z.reshape(-1, z.shape[-1])
            if z.stride(-1) != 1:
                z = z.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y, mean, rstd = _layer_norm_fwd(
            x,
            weight,
            bias,
            eps,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd, z)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.group_size = group_size
        ctx.norm_before_gate = norm_before_gate
        ctx.is_rms_norm = is_rms_norm
        return y.reshape(x_shape_og)

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, mean, rstd, z = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        assert dy.shape == x.shape
        dx, dw, db, dz = _layer_norm_bwd(
            dy, x, weight, bias, ctx.eps, mean, rstd, z, ctx.group_size, ctx.norm_before_gate, ctx.is_rms_norm
        )
        return (
            dx.reshape(ctx.x_shape_og),
            dw,
            db,
            dz.reshape(ctx.x_shape_og) if dz is not None else None,
            None,
            None,
            None,
            None,
        )
