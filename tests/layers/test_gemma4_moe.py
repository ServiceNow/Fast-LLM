import pytest
import torch

from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.decoder.mlp.config import Gemma4MoEMLPConfig
from tests.utils.utils import get_stage


def _make_gemma4_moe(device: torch.device, experts_per_token: int = 2):
    config = Gemma4MoEMLPConfig._from_dict(
        {
            "type": "gemma4_moe",
            "intermediate_size": 12,
            "moe_intermediate_size": 4,
            "experts": 3,
            "experts_per_token": experts_per_token,
            "gated": True,
            "activation": "gelu",
            "add_linear_biases": False,
            "implementation": "looped",
            "layer_1": {},
            "layer_2": {},
            "expert_layer_1": {},
            "expert_layer_2": {},
            "router": {},
            "router_scale": {},
            "per_expert_scale": {},
            "post_feedforward_norm_1": {"type": "rms_norm"},
            "pre_feedforward_norm_2": {"type": "rms_norm"},
            "post_feedforward_norm_2": {"type": "rms_norm"},
        }
    )
    distributed_config = DistributedConfig(compute_dtype="float32", use_cuda=torch.cuda.is_available())
    layer = config.get_layer(distributed_config, TensorDim("hidden_size", 8), lr_scale=None, peft=None, return_bias=True)
    get_stage([layer], Distributed(distributed_config))
    return layer.to(device)


@pytest.mark.slow
@pytest.mark.parametrize("experts_per_token", [1, 2])
def test_gemma4_moe_forward_scale_gradients(experts_per_token):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = _make_gemma4_moe(device, experts_per_token)

    input_ = torch.randn(5, 8, dtype=torch.float32, device=device, requires_grad=True)
    kwargs = {
        BlockKwargs.pre_mlp_residual: input_,
        BlockKwargs.hidden_token_dim: TensorDim("tokens", 5),
    }
    output, bias = layer(layer.pre_feedforward_norm_2(input_), kwargs)
    assert bias is None
    assert output.shape == input_.shape
    assert layer.expert_layer_2.transposed_weight

    output.sum().backward()

    assert input_.grad is not None
    assert (layer.router_scale.grad is not None) or (getattr(layer.router_scale, "grad_buffer", None) is not None)
    assert (layer.per_expert_scale.grad is not None) or (
        getattr(layer.per_expert_scale, "grad_buffer", None) is not None
    )


def test_gemma4_routing_softmaxes_before_topk():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = _make_gemma4_moe(device)
    with torch.no_grad():
        layer.per_expert_scale.copy_(torch.tensor([1.0, 2.0, 3.0], device=device))
    logits = torch.tensor([[5.0, 4.0, -10.0], [0.0, 1.0, 2.0]], device=device)

    scores, top_experts = layer._topk_routing(logits)

    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    expected_scores, expected_top_experts = torch.topk(probs, k=2, dim=-1)
    expected_scores = expected_scores / expected_scores.sum(dim=-1, keepdim=True)
    expected_scores = expected_scores * layer.per_expert_scale[expected_top_experts]

    top_logits, _ = torch.topk(logits, k=2, dim=-1)
    standard_moe_scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32)

    torch.testing.assert_close(top_experts, expected_top_experts)
    torch.testing.assert_close(scores, expected_scores.type_as(scores))
    assert not torch.allclose(scores, standard_moe_scores.type_as(scores))
