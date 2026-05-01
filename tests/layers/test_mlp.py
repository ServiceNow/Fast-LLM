import dataclasses

import pytest
import torch

from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.decoder.mlp.config import HybridMoEMLPConfig
from fast_llm.layers.decoder.mlp.mixture_of_experts import HybridMoEMLP
from fast_llm.utils import Assert
from tests.utils.utils import get_stage

_NUM_TOKENS = 128
_HIDDEN_SIZE = 128
_INTERMEDIATE_SIZE = 128
_EXPERTS = 4

_NORM = {"type": "rms_norm"}


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
            dense["pre_norm"] = _NORM
        if self.dense_post_norm:
            dense["post_norm"] = _NORM
        if self.routed_pre_norm:
            routed["pre_norm"] = _NORM
        if self.routed_post_norm:
            routed["post_norm"] = _NORM
        wrapper: dict = {"dense": dense, "routed": routed}
        if self.wrapper_pre_norm:
            wrapper["pre_norm"] = _NORM
        if self.wrapper_post_norm:
            wrapper["post_norm"] = _NORM
        return HybridMoEMLPConfig.from_dict(wrapper)

    def expected_output(self, hybrid: HybridMoEMLP, input_: torch.Tensor, kwargs: dict) -> torch.Tensor:
        with torch.no_grad():
            shared = hybrid.pre_norm(input_) if hybrid.pre_norm is not None else input_
            dense_out, _ = hybrid.dense(shared, kwargs)
            routed_out, _ = hybrid.routed(shared, kwargs)
            out = dense_out + routed_out
            if hybrid.post_norm is not None:
                out = hybrid.post_norm(out)
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
    get_stage([hybrid], distributed)
    hybrid.eval()

    input_ = torch.randn(_NUM_TOKENS, _HIDDEN_SIZE, device=device)
    token_dim = TensorDim("tokens", _NUM_TOKENS)
    kwargs = {BlockKwargs.hidden_token_dim: token_dim}

    with torch.no_grad():
        output = hybrid(input_, kwargs)

    expected = config.expected_output(hybrid, input_, kwargs)
    Assert.rms_close_relative(output, expected, 1e-5, 1e-7)
