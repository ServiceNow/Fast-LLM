"""LayerNorm and RMSNorm. The Triton kernel writes parameter gradients to a
`grad_buffer` attribute (Fast-LLM convention) instead of autograd's `.grad`."""

import dataclasses

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.normalization import triton_normalization_autograd
from fast_llm.layers.common.normalization.normalization import FastLayerNorm, FusedLayerNorm, FusedRMSNorm
from tools.benchmark.triton_kernels.runner import DtypedCase, Inputs, Variant
from tools.benchmark.triton_kernels.utils import (
    dtype_short,
    make_grad_reset,
    standard_pytorch_variants,
)

try:
    import fused_layer_norm_cuda  # noqa: F401

    _fused_normalization_available = torch.cuda.is_available()
except ImportError:
    _fused_normalization_available = False

try:
    import fast_layer_norm  # noqa: F401

    _fast_normalization_available = torch.cuda.is_available()
except ImportError:
    _fast_normalization_available = False

# (batch*seq, hidden). Numel fixed at 32M to mimic a constant training memory
# budget across model widths; hidden swept from 1K to 16K.
_SHAPES = [
    (32768, 1024),
    (16384, 2048),
    (8192, 4096),
    (4096, 8192),
    (2048, 16384),
]
_EPS = 1e-5


def _setup_param(tensor: torch.Tensor) -> torch.Tensor:
    tensor.grad_buffer = torch.zeros_like(tensor)
    tensor.param_grad_is_zero = True
    return tensor


@dataclasses.dataclass
class _NormalizationCase(DtypedCase):
    rows: int
    cols: int
    dtype: torch.dtype

    @property
    def name(self) -> str:
        return f"({self.rows}, {self.cols}) {dtype_short(self.dtype)}"


@dataclasses.dataclass
class LayerNormCase(_NormalizationCase):
    @property
    def expected_bytes(self) -> int:
        # 4× activations (fwd+bwd in/out) + weight & bias × (read + grad write).
        return 4 * self.rows * self.cols * self.dtype.itemsize + 6 * self.cols * self.dtype.itemsize

    @property
    def expected_flops(self) -> int:
        # fwd ≈ 7 FLOPs/elem (mean, variance, normalize, scale+shift); bwd ≈ 2× fwd.
        return 21 * self.rows * self.cols

    def make_inputs(self, device: torch.device) -> Inputs:
        return {
            "input": torch.randn(self.rows, self.cols, dtype=self.dtype, device=device, requires_grad=True),
            "weight": _setup_param(torch.randn(self.cols, dtype=self.dtype, device=device, requires_grad=True)),
            "bias": _setup_param(torch.zeros(self.cols, dtype=self.dtype, device=device, requires_grad=True)),
            "grad_output": torch.randn(self.rows, self.cols, dtype=self.dtype, device=device),
        }


@dataclasses.dataclass
class RmsNormCase(_NormalizationCase):
    @property
    def expected_bytes(self) -> int:
        # No bias compared to LayerNorm.
        return 4 * self.rows * self.cols * self.dtype.itemsize + 3 * self.cols * self.dtype.itemsize

    @property
    def expected_flops(self) -> int:
        # No mean subtraction or bias compared to LayerNorm.
        return 15 * self.rows * self.cols

    def make_inputs(self, device: torch.device) -> Inputs:
        return {
            "input": torch.randn(self.rows, self.cols, dtype=self.dtype, device=device, requires_grad=True),
            "weight": _setup_param(torch.randn(self.cols, dtype=self.dtype, device=device, requires_grad=True)),
            "grad_output": torch.randn(self.rows, self.cols, dtype=self.dtype, device=device),
        }


def _layer_norm_eager(input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.layer_norm(input_, weight.shape, weight, bias, _EPS)


def _rms_norm_eager(input_: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.rms_norm(input_, weight.shape, weight, _EPS)


def _param_grad(param: torch.Tensor) -> torch.Tensor:
    return param.grad if param.grad is not None else param.grad_buffer


def _layer_norm_triton(inputs: Inputs) -> torch.Tensor:
    return triton_normalization_autograd(
        inputs["input"], inputs["weight"], inputs["bias"], eps=_EPS, training=True, zero_centered=False
    )


def _rms_norm_triton(inputs: Inputs) -> torch.Tensor:
    return triton_normalization_autograd(
        inputs["input"], inputs["weight"], None, eps=_EPS, training=True, zero_centered=False
    )


def _layer_norm_triton_fwd(inputs: Inputs) -> dict:
    return {"output": _layer_norm_triton(inputs)}


def _layer_norm_triton_fwd_bwd(inputs: Inputs) -> dict:
    output = _layer_norm_triton(inputs)
    output.backward(inputs["grad_output"])
    return {
        "output": output.detach(),
        "grad_input": inputs["input"].grad,
        "grad_weight": _param_grad(inputs["weight"]),
        "grad_bias": _param_grad(inputs["bias"]),
    }


def _rms_norm_triton_fwd(inputs: Inputs) -> dict:
    return {"output": _rms_norm_triton(inputs)}


def _rms_norm_triton_fwd_bwd(inputs: Inputs) -> dict:
    output = _rms_norm_triton(inputs)
    output.backward(inputs["grad_output"])
    return {
        "output": output.detach(),
        "grad_input": inputs["input"].grad,
        "grad_weight": _param_grad(inputs["weight"]),
    }


def _layer_norm_apex_fused(input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return FusedLayerNorm.apply(input_, weight.shape, weight, bias, _EPS)


def _layer_norm_apex_fast(input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return FastLayerNorm.apply(input_, weight.shape, weight, bias, _EPS)


def _rms_norm_apex_fused(input_: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return FusedRMSNorm.apply(input_, weight.shape, weight, _EPS)


_LAYER_NORM_EXTRAS: dict = {}
if _fused_normalization_available:
    _LAYER_NORM_EXTRAS["apex_fused"] = _layer_norm_apex_fused
if _fast_normalization_available:
    _LAYER_NORM_EXTRAS["apex_fast"] = _layer_norm_apex_fast

_RMS_NORM_EXTRAS: dict = {}
if _fused_normalization_available:
    _RMS_NORM_EXTRAS["apex_fused"] = _rms_norm_apex_fused


def benchmarks(
    dtypes: tuple[torch.dtype, ...],
    shapes: list[tuple[int, int]] | None = None,
) -> list[tuple[str, list, list]]:
    shapes = shapes if shapes is not None else _SHAPES
    layer_norm_variants = standard_pytorch_variants(
        _layer_norm_eager,
        input_keys=("input", "weight", "bias"),
        grad_input_keys=("input", "weight", "bias"),
        grad_output_key="grad_output",
        extra_functions=_LAYER_NORM_EXTRAS,
    )
    if TritonConfig.enabled():
        layer_norm_variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_layer_norm_triton_fwd,
                fwd_bwd=_layer_norm_triton_fwd_bwd,
                reset_inputs=make_grad_reset(("input", "weight", "bias")),
            )
        )
    rms_norm_variants = standard_pytorch_variants(
        _rms_norm_eager,
        input_keys=("input", "weight"),
        grad_input_keys=("input", "weight"),
        grad_output_key="grad_output",
        extra_functions=_RMS_NORM_EXTRAS,
    )
    if TritonConfig.enabled():
        rms_norm_variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_rms_norm_triton_fwd,
                fwd_bwd=_rms_norm_triton_fwd_bwd,
                reset_inputs=make_grad_reset(("input", "weight")),
            )
        )
    return [
        (
            "normalization: layer_norm",
            [LayerNormCase(rows=r, cols=c, dtype=d) for d in dtypes for (r, c) in shapes],
            layer_norm_variants,
        ),
        (
            "normalization: rms_norm",
            [RmsNormCase(rows=r, cols=c, dtype=d) for d in dtypes for (r, c) in shapes],
            rms_norm_variants,
        ),
    ]
