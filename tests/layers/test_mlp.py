import dataclasses
import types

import pytest
import torch

from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.decoder.mlp.config import HybridMoEMLPConfig
from fast_llm.layers.decoder.mlp.mixture_of_experts import HybridMoEMLP, MixtureOfExpertMLP
from fast_llm.utils import Assert
from tests.utils.utils import get_stage, no_tf32

_NUM_TOKENS = 128
_HIDDEN_SIZE = 128
_INTERMEDIATE_SIZE = 128
_EXPERTS = 4


def _norm() -> dict:
    # Fresh dict per call: `from_dict` consumes the `type` key during parsing, so a shared
    # module-level dict would silently become `{}` (i.e. default `layer_norm`) on second use.
    return {"type": "rms_norm"}


class _RouterBridge(torch.autograd.Function):
    """Catches the router's real output and incoming gradient for cross-side comparison and
    substitutes deterministic mock data in both directions, so the assembly under test does
    not depend on routing.

    Forward: catches `real_logits`, returns `mock_logits` so the downstream `torch.topk` +
    softmax + expert dispatch is deterministic regardless of ~1e-7 FP perturbations that
    would otherwise flip top-k for near-boundary tokens.
    Backward: catches the gradient computed against `mock_logits`, returns `mock_grad` to
    `real_logits` so the router's parameters still receive a gradient through their normal
    backward path.

    `captured` is a mutable dict the caller passes in to receive both catches.
    """

    @staticmethod
    def forward(ctx, real_logits, mock_logits, mock_grad, captured):
        captured["real_logits"] = real_logits.detach()
        ctx.save_for_backward(mock_grad)
        ctx.captured = captured
        return mock_logits.detach()

    @staticmethod
    def backward(ctx, grad_output):
        (mock_grad,) = ctx.saved_tensors
        ctx.captured["mock_logits_grad"] = grad_output.detach()
        return mock_grad, None, None, None


def _wrap_router(routed: MixtureOfExpertMLP, mock_logits, mock_grad, captured) -> None:
    if not hasattr(routed, "_orig_topk_routing"):
        routed._orig_topk_routing = routed._topk_routing
    real_topk = routed._orig_topk_routing

    def _patched(self, logits, grad_scale=None, losses=None):
        bridged = _RouterBridge.apply(logits, mock_logits, mock_grad, captured)
        return real_topk(bridged, grad_scale, losses)

    routed._topk_routing = types.MethodType(_patched, routed)


@dataclasses.dataclass
class HybridMoEMLPTestConfig:
    name: str
    gated: bool = False
    experts_per_token: int = 1
    wrapper_pre_norm: bool = False
    wrapper_post_norm: bool = False
    dense_pre_norm: bool = False
    dense_post_norm: bool = False
    routed_pre_norm: bool = False
    routed_post_norm: bool = False

    def get_mlp_config(self) -> HybridMoEMLPConfig:
        dense: dict = {
            "intermediate_size": _INTERMEDIATE_SIZE,
            "gated": self.gated,
            "add_linear_biases": False,
        }
        routed: dict = {
            "intermediate_size": _INTERMEDIATE_SIZE,
            "gated": self.gated,
            "add_linear_biases": False,
            "experts": _EXPERTS,
            "experts_per_token": self.experts_per_token,
        }
        if self.dense_pre_norm:
            dense["pre_norm"] = _norm()
        if self.dense_post_norm:
            dense["post_norm"] = _norm()
        if self.routed_pre_norm:
            routed["pre_norm"] = _norm()
        if self.routed_post_norm:
            routed["post_norm"] = _norm()
        wrapper: dict = {"dense": dense, "routed": routed}
        if self.wrapper_pre_norm:
            wrapper["pre_norm"] = _norm()
        if self.wrapper_post_norm:
            wrapper["post_norm"] = _norm()
        return HybridMoEMLPConfig.from_dict(wrapper)

    def expected_output(self, hybrid: HybridMoEMLP, input_: torch.Tensor, kwargs: dict) -> torch.Tensor:
        # Hybrid-assembly test. The dense and routed branches are treated as black boxes (covered
        # by `MLP` and `MixtureOfExpertMLP` tests); pre/post norms are computed via
        # `torch.rms_norm` so the wrapper's norms do not appear in their own reference. Runs
        # under autograd so the caller can backward through this reference.
        def _rms_norm(x: torch.Tensor, norm_module) -> torch.Tensor:
            return torch.rms_norm(x, x.shape[-1:], norm_module.weight, 1e-5)

        shared = _rms_norm(input_, hybrid.pre_norm) if hybrid.pre_norm is not None else input_
        dense_out, _ = hybrid.dense(shared, kwargs)
        routed_out, _ = hybrid.routed(shared, kwargs)
        out = dense_out + routed_out
        if hybrid.post_norm is not None:
            out = _rms_norm(out, hybrid.post_norm)
        return out


_test_configs = [
    HybridMoEMLPTestConfig(name="basic"),
    HybridMoEMLPTestConfig(name="gated", gated=True),
    HybridMoEMLPTestConfig(name="topk2", experts_per_token=2),
    HybridMoEMLPTestConfig(name="gated_topk2", gated=True, experts_per_token=2),
    HybridMoEMLPTestConfig(name="branch_pre_norms", dense_pre_norm=True, routed_pre_norm=True),
    HybridMoEMLPTestConfig(name="branch_post_norms", dense_post_norm=True, routed_post_norm=True),
    HybridMoEMLPTestConfig(name="wrapper_norms", wrapper_pre_norm=True, wrapper_post_norm=True),
    HybridMoEMLPTestConfig(
        name="all_norms",
        wrapper_pre_norm=True,
        wrapper_post_norm=True,
        dense_pre_norm=True,
        dense_post_norm=True,
        routed_pre_norm=True,
        routed_post_norm=True,
    ),
    HybridMoEMLPTestConfig(name="asymmetric_norms", dense_pre_norm=True, routed_post_norm=True),
]


@pytest.mark.parametrize("config", [pytest.param(c, id=c.name) for c in _test_configs])
def test_hybrid_moe_mlp(config: HybridMoEMLPTestConfig) -> None:
    distributed_config = DistributedConfig(use_cuda=torch.cuda.is_available())
    distributed = Distributed(distributed_config)
    device = distributed.device
    hidden_dim = TensorDim("hidden", _HIDDEN_SIZE)

    hybrid: HybridMoEMLP = config.get_mlp_config().get_layer(
        distributed_config, hidden_dim, lr_scale=None, peft=None, return_bias=False
    )
    stage = get_stage([hybrid], distributed)
    # Train mode so the codebase's custom-autograd kernels retain backward context. With
    # dropout=0 and jitter_eps=0 (defaults), train mode is functionally identical to eval
    # mode here — the only difference is context retention.
    hybrid.train()

    # Predetermined mock router output + incoming gradient, shared between actual and reference.
    n_router_experts = hybrid.routed._config.unshared_experts
    g = torch.Generator(device=device).manual_seed(0xB007)
    mock_logits = torch.randn(_NUM_TOKENS, n_router_experts, device=device, generator=g)
    mock_grad = torch.randn(_NUM_TOKENS, n_router_experts, device=device, generator=g)

    input_ = torch.randn(_NUM_TOKENS, _HIDDEN_SIZE, device=device)
    token_dim = TensorDim("tokens", _NUM_TOKENS)
    kwargs = {BlockKwargs.hidden_token_dim: token_dim}

    captures_actual: dict = {}
    _wrap_router(hybrid.routed, mock_logits, mock_grad, captures_actual)
    stage.reset_gradients()
    with no_tf32():
        input_actual = input_.clone().requires_grad_(True)
        output = hybrid(input_actual, kwargs)
        output.backward(torch.ones_like(output))
        grad_actual = input_actual.grad.clone()

    captures_ref: dict = {}
    _wrap_router(hybrid.routed, mock_logits, mock_grad, captures_ref)
    stage.reset_gradients()
    with no_tf32():
        input_ref = input_.clone().requires_grad_(True)
        expected = config.expected_output(hybrid, input_ref, kwargs)
        expected.backward(torch.ones_like(expected))
        grad_ref = input_ref.grad.clone()

    # 1e-4 absorbs FP32 noise from the wrapper pre-norm + post-norm Triton-vs-`torch.rms_norm`
    # divergence propagated through matmuls (up to ~5e-5 observed for `wrapper_norms` /
    # `all_norms`). All other configs are bit-exact or in the 1e-7 range, well below threshold.
    Assert.rms_close_relative(output, expected, 1e-4, 1e-7)
    Assert.rms_close_relative(grad_actual, grad_ref, 1e-4, 1e-7)
    Assert.rms_close_relative(captures_actual["real_logits"], captures_ref["real_logits"], 1e-4, 1e-7)
    Assert.rms_close_relative(captures_actual["mock_logits_grad"], captures_ref["mock_logits_grad"], 1e-4, 1e-7)
