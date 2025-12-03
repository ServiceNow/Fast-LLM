import pytest
import torch

from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.functional.config import ActivationType
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.ssm.config import GatedDeltaNetConfig

try:
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig, Qwen3NextGatedDeltaNet
except ImportError:
    Qwen3NextConfig, Qwen3NextGatedDeltaNet = None, None


def _materialize_mixer_tensors(module: torch.nn.Module, distributed: Distributed, device: torch.device) -> None:
    """
    Instantiate meta-allocated parameters on the requested device so the layer can run standalone.
    """
    for name, param in module.named_parameters():
        if param.device.type != "meta":
            continue
        param_data = torch.empty_like(param, device=device)
        param.init_parameter(param_data, distributed)
        module_path, param_name = name.rsplit(".", 1) if "." in name else (None, name)
        target = module
        if module_path is not None:
            for part in module_path.split("."):
                target = getattr(target, part)
        new_param = torch.nn.Parameter(param_data, requires_grad=param.requires_grad)
        new_param.grad = None
        new_param.grad_buffer = torch.zeros_like(param_data)
        new_param.param_grad_is_zero = True
        target._parameters[param_name] = new_param


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Varlen test needs CUDA")
@pytest.mark.skipif(Qwen3NextConfig is None, reason="transformers with Qwen3-Next not installed")
def test_fast_llm_gdn_matches_qwen3_next_forward():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    hidden_size = 16
    seq_len = 6
    num_k_heads = 2
    num_v_heads = 4
    head_k_dim = 4
    head_v_dim = 4
    kernel_size = 4

    hf_config = Qwen3NextConfig(
        hidden_size=hidden_size,
        linear_num_key_heads=num_k_heads,
        linear_num_value_heads=num_v_heads,
        linear_key_head_dim=head_k_dim,
        linear_value_head_dim=head_v_dim,
        linear_conv_kernel_dim=kernel_size,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        dtype=dtype,
    )
    hf_layer = Qwen3NextGatedDeltaNet(hf_config, layer_idx=0).to(device=device, dtype=dtype).eval()

    fast_config = GatedDeltaNetConfig(
        value_heads=num_v_heads,
        key_heads=num_k_heads,
        value_head_dim=head_v_dim,
        key_head_dim=head_k_dim,
        activation=ActivationType.silu,
        normalization={"epsilon": hf_config.rms_norm_eps},
        convolution_layer={"kernel_size": kernel_size, "activation": ActivationType.silu},
    )
    distributed_config = DistributedConfig(
        tensor_parallel=1,
        pipeline_parallel=1,
        sequence_data_parallel=1,
        local_world_size=1,
        world_size=1,
    )
    hidden_dim = TensorDim("hidden", hidden_size)
    fast_layer = fast_config.get_layer(distributed_config, hidden_dim, lr_scale=None, peft=None, return_bias=False)
    distributed = Distributed(config=distributed_config)
    fast_layer.setup(distributed)
    _materialize_mixer_tensors(fast_layer, distributed, device)
    fast_layer.to(device=device, dtype=dtype).eval()

    with torch.no_grad():
        fast_layer.in_proj_qkvz.weight.copy_(hf_layer.in_proj_qkvz.weight)
        fast_layer.in_proj_ba.weight.copy_(hf_layer.in_proj_ba.weight)
        fast_layer.convolution.weight.copy_(hf_layer.conv1d.weight)
        if fast_layer.convolution.bias is not None and hf_layer.conv1d.bias is not None:
            fast_layer.convolution.bias.copy_(hf_layer.conv1d.bias)
        fast_layer.out_proj.weight.copy_(hf_layer.out_proj.weight)
        fast_layer.A_log.copy_(hf_layer.A_log)
        fast_layer.dt_bias.copy_(hf_layer.dt_bias)
        fast_layer.norm.weight.copy_(hf_layer.norm.weight)

    hidden_states = torch.randn(1, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=False)

    param_map = {
        "in_proj_qkvz.weight": "in_proj_qkvz.weight",
        "in_proj_ba.weight": "in_proj_ba.weight",
        "convolution.weight": "conv1d.weight",
        "convolution.bias": "conv1d.bias",
        "out_proj.weight": "out_proj.weight",
        "A_log": "A_log",
        "dt_bias": "dt_bias",
        "norm.weight": "norm.weight",
    }
    for k, p in fast_layer.state_dict().items():
        torch.testing.assert_close(p, hf_layer.state_dict()[param_map[k]], atol=1e-6, rtol=1e-6)

    # need to monkey patch the hf implementation with our fix_query_key_value_ordering due to the layout differences
    hf_layer.fix_query_key_value_ordering = fast_layer.fix_query_key_value_ordering
    hf_layer._local_key_heads = fast_layer._local_key_heads
    hf_layer._local_value_heads = fast_layer._local_value_heads
    hf_layer._config = fast_layer._config

    hf_out = hf_layer(hidden_states)

    fast_kwargs = {
        BlockKwargs.sequence_first: False,
        BlockKwargs.hidden_dims: (hidden_dim,),
    }
    fast_out = fast_layer(hidden_states, fast_kwargs)

    torch.testing.assert_close(fast_out, hf_out, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
