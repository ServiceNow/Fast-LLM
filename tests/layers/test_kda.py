import pytest
import torch

import fast_llm.layers.ssm.kda as kda_module
from fast_llm.config import UpdateType
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.utils import Assert
from tests.utils.utils import get_base_model, get_stage, requires_cuda

try:
    from fast_llm_external_models.apriel_hybrid_ssm.configuration_apriel_hybrid_ssm import AprielHybridSSMConfig
    from fast_llm_external_models.apriel_hybrid_ssm.modeling_apriel_hybrid_ssm import KimiDeltaAttention
except ImportError:
    AprielHybridSSMConfig, KimiDeltaAttention = None, None

VOCAB_SIZE = 500
HIDDEN_SIZE = 16
SEQ_LEN = 65
NUM_HEADS = 4
HEAD_DIM = 4
KERNEL_SIZE = 4


@pytest.mark.slow
@requires_cuda
@pytest.mark.skipif(KimiDeltaAttention is None or AprielHybridSSMConfig is None, reason="Apriel KDA deps missing")
@pytest.mark.skipif(kda_module.chunk_kda is None, reason="KDA fused kernels not available")
def test_fast_llm_kda_matches_apriel_forward():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    hf_config = AprielHybridSSMConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=1,
        rms_norm_eps=1e-6,
    )
    hf_config.short_conv_kernel_size = KERNEL_SIZE
    hf_config.head_dim = HEAD_DIM
    hf_config.num_heads = NUM_HEADS
    hf_layer = KimiDeltaAttention(hf_config, layer_idx=0).to(device=device, dtype=dtype).eval()

    config = GPTBaseModelConfig.from_dict(
        {
            "decoder": {
                "num_blocks": 1,
                "block": {
                    "mixer": {
                        "type": "kda",
                        "heads": NUM_HEADS,
                        "head_dim": HEAD_DIM,
                        "convolution_layer": {"kernel_size": KERNEL_SIZE, "activation": "silu"},
                        "normalization": {"epsilon": hf_config.rms_norm_eps, "activation": "sigmoid"},
                    }
                },
            },
            "embeddings": {"vocab_size": VOCAB_SIZE},
            "hidden_size": HIDDEN_SIZE,
        },
        update_type=UpdateType.update,
    )

    model, distributed = get_base_model(
        GPTModelConfig.from_dict(
            {
                "base_model": config,
                "distributed": {},
            },
        )
    )
    fast_layer = model.decoder[0].mixer
    get_stage([fast_layer], distributed, [], {})
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
    hf_params = hf_layer.state_dict()
    for fast_name, fast_param in fast_layer.state_dict().items():
        hf_param = hf_params[param_map[fast_name]]
        if fast_param.shape != hf_param.shape:
            Assert.eq(fast_param.numel(), hf_param.numel(), msg=fast_name)
            hf_param = hf_param.reshape_as(fast_param)
        Assert.rms_close_relative(fast_param, hf_param, 1e-5, 1e-5, msg=fast_name)

    hidden_states = torch.randn(2, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=dtype, requires_grad=False)
    hf_layer.training = True
    hf_out = hf_layer(hidden_states)

    sequence_lengths = [[SEQ_LEN] for _ in range(hidden_states.size(0))]
    fast_kwargs = {
        BlockKwargs.device: device,
        BlockKwargs.sequence_first: False,
        BlockKwargs.sequence_lengths: sequence_lengths,
        BlockKwargs.hidden_dims: (HIDDEN_SIZE,),
    }
    fast_layer.preprocess(fast_kwargs)
    fast_out, _ = fast_layer(hidden_states, fast_kwargs)

    Assert.rms_close_relative(fast_out, hf_out, 1e-5, 1e-5)
