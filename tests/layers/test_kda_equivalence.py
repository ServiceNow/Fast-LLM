import pytest
import torch

import fast_llm.layers.ssm.kda as kda_module
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.ssm.config import KimiDeltaAttentionConfig

try:
    from fast_llm_external_models.apriel_hybrid_ssm.configuration_apriel_hybrid_ssm import AprielHybridSSMConfig
    from fast_llm_external_models.apriel_hybrid_ssm.modeling_apriel_hybrid_ssm import KimiDeltaAttention
except ImportError:
    AprielHybridSSMConfig, KimiDeltaAttention = None, None


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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="KDA equivalence test needs CUDA")
@pytest.mark.skipif(KimiDeltaAttention is None or AprielHybridSSMConfig is None, reason="Apriel KDA deps missing")
@pytest.mark.skipif(kda_module.chunk_kda is None, reason="KDA fused kernels not available")
def test_fast_llm_kda_matches_apriel_forward():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    hidden_size = 16
    seq_len = 65
    num_heads = 4
    head_dim = 4
    kernel_size = 4

    hf_config = AprielHybridSSMConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_hidden_layers=1,
        rms_norm_eps=1e-6,
    )
    # Populate fields expected by the HF implementation.
    hf_config.short_conv_kernel_size = kernel_size
    hf_config.head_dim = head_dim
    hf_config.num_heads = num_heads
    hf_layer = KimiDeltaAttention(hf_config, layer_idx=0).to(device=device, dtype=dtype).eval()

    fast_config = KimiDeltaAttentionConfig(
        heads=num_heads,
        head_dim=head_dim,
        convolution_layer={"kernel_size": kernel_size, "activation": "silu"},
        normalization={"epsilon": hf_config.rms_norm_eps, "activation": "sigmoid"},
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
        fast_layer.q_proj.weight.copy_(hf_layer.q_proj.weight)
        fast_layer.k_proj.weight.copy_(hf_layer.k_proj.weight)
        fast_layer.v_proj.weight.copy_(hf_layer.v_proj.weight)
        fast_layer.q_conv.weight.copy_(hf_layer.q_conv1d.weight)
        fast_layer.k_conv.weight.copy_(hf_layer.k_conv1d.weight)
        fast_layer.v_conv.weight.copy_(hf_layer.v_conv1d.weight)
        if fast_layer.q_conv.bias is not None and hf_layer.q_conv1d.bias is not None:
            fast_layer.q_conv.bias.copy_(hf_layer.q_conv1d.bias)
        if fast_layer.k_conv.bias is not None and hf_layer.k_conv1d.bias is not None:
            fast_layer.k_conv.bias.copy_(hf_layer.k_conv1d.bias)
        if fast_layer.v_conv.bias is not None and hf_layer.v_conv1d.bias is not None:
            fast_layer.v_conv.bias.copy_(hf_layer.v_conv1d.bias)
        fast_layer.f_a_proj.weight.copy_(hf_layer.f_a_proj.weight)
        fast_layer.f_b_proj.weight.copy_(hf_layer.f_b_proj.weight)
        fast_layer.g_a_proj.weight.copy_(hf_layer.g_a_proj.weight)
        fast_layer.g_b_proj.weight.copy_(hf_layer.g_b_proj.weight)
        fast_layer.beta_proj.weight.copy_(hf_layer.b_proj.weight)
        fast_layer.o_proj.weight.copy_(hf_layer.o_proj.weight)
        fast_layer.A_log.copy_(hf_layer.A_log.reshape_as(fast_layer.A_log))
        fast_layer.dt_bias.copy_(hf_layer.dt_bias.reshape_as(fast_layer.dt_bias))
        fast_layer.norm.weight.copy_(hf_layer.o_norm.weight)

    param_map = {
        "q_proj.weight": "q_proj.weight",
        "k_proj.weight": "k_proj.weight",
        "v_proj.weight": "v_proj.weight",
        "q_conv.weight": "q_conv1d.weight",
        "k_conv.weight": "k_conv1d.weight",
        "v_conv.weight": "v_conv1d.weight",
        "f_a_proj.weight": "f_a_proj.weight",
        "f_b_proj.weight": "f_b_proj.weight",
        "g_a_proj.weight": "g_a_proj.weight",
        "g_b_proj.weight": "g_b_proj.weight",
        "beta_proj.weight": "b_proj.weight",
        "o_proj.weight": "o_proj.weight",
        "A_log": "A_log",
        "dt_bias": "dt_bias",
        "norm.weight": "o_norm.weight",
    }
    for fast_name, hf_name in param_map.items():
        fast_param = fast_layer.state_dict()[fast_name]
        hf_param = hf_layer.state_dict()[hf_name]
        if fast_param.shape != hf_param.shape:
            hf_param = hf_param.reshape_as(fast_param)
        print(f"Comparing parameter {fast_name} with shape {fast_param.shape}")
        torch.testing.assert_close(fast_param, hf_param, atol=1e-6, rtol=1e-6)

    hidden_states = torch.randn(2, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=False)
    hf_layer.training = True
    hf_out = hf_layer(hidden_states)

    sequence_lengths = [[seq_len] for _ in range(hidden_states.size(0))]
    fast_kwargs = {
        BlockKwargs.device: device,
        BlockKwargs.sequence_first: False,
        BlockKwargs.sequence_lengths: sequence_lengths,
        BlockKwargs.hidden_dims: (hidden_dim,),
    }
    fast_layer.preprocess(fast_kwargs)
    fast_out = fast_layer(hidden_states, fast_kwargs)

    torch.testing.assert_close(fast_out, hf_out, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
