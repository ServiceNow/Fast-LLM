"""Test numerical equivalence between Fast-LLM KDA and Apriel2 KimiDeltaAttention."""

import pytest
import torch

import fast_llm.layers.ssm.kda as kda_module
from fast_llm.config import UpdateType
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.utils import Assert
from tests.utils.utils import get_base_model, get_stage, requires_cuda

try:
    from fast_llm_external_models.apriel2.modeling_apriel2 import KimiDeltaAttention
except ImportError:
    KimiDeltaAttention = None

# Test constants
VOCAB_SIZE = 500
HIDDEN_SIZE = 64
SEQ_LEN = 65
BATCH_SIZE = 2
NUM_HEADS = 4
HEAD_DIM = 16
KERNEL_SIZE = 4


def _copy_weights(fast_layer, hf_layer):
    """Copy weights from Apriel2 KDA to Fast-LLM KDA."""
    with torch.no_grad():
        # Main projections
        fast_layer.q_proj.weight.copy_(hf_layer.q_proj.weight)
        fast_layer.k_proj.weight.copy_(hf_layer.k_proj.weight)
        fast_layer.v_proj.weight.copy_(hf_layer.v_proj.weight)
        fast_layer.o_proj.weight.copy_(hf_layer.o_proj.weight)

        # Convolutions
        fast_layer.q_conv.weight.copy_(hf_layer.q_conv.weight)
        fast_layer.k_conv.weight.copy_(hf_layer.k_conv.weight)
        fast_layer.v_conv.weight.copy_(hf_layer.v_conv.weight)
        if fast_layer.q_conv.bias is not None and hf_layer.q_conv.bias is not None:
            fast_layer.q_conv.bias.copy_(hf_layer.q_conv.bias)
        if fast_layer.k_conv.bias is not None and hf_layer.k_conv.bias is not None:
            fast_layer.k_conv.bias.copy_(hf_layer.k_conv.bias)
        if fast_layer.v_conv.bias is not None and hf_layer.v_conv.bias is not None:
            fast_layer.v_conv.bias.copy_(hf_layer.v_conv.bias)

        # Gate projections (low-rank)
        fast_layer.f_a_proj.weight.copy_(hf_layer.f_a_proj.weight)
        fast_layer.f_b_proj.weight.copy_(hf_layer.f_b_proj.weight)
        fast_layer.g_a_proj.weight.copy_(hf_layer.g_a_proj.weight)
        fast_layer.g_b_proj.weight.copy_(hf_layer.g_b_proj.weight)

        # Beta and learnable params
        fast_layer.beta_proj.weight.copy_(hf_layer.beta_proj.weight)
        fast_layer.A_log.copy_(hf_layer.A_log.reshape_as(fast_layer.A_log))
        fast_layer.dt_bias.copy_(hf_layer.dt_bias.reshape_as(fast_layer.dt_bias))

        # Normalization
        fast_layer.norm.weight.copy_(hf_layer.norm.weight)


@pytest.mark.slow
@requires_cuda
@pytest.mark.skipif(KimiDeltaAttention is None, reason="Apriel2 KDA not available")
@pytest.mark.skipif(kda_module.chunk_kda is None, reason="KDA fused kernels not available")
def test_fast_llm_kda_matches_apriel2_forward():
    """Verify Fast-LLM KDA output matches Apriel2 KimiDeltaAttention."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Create Apriel2 KDA layer
    kda_config = {
        "heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "convolution_layer": {"kernel_size": KERNEL_SIZE},
        "normalization": {"epsilon": 1e-6},
    }
    hf_layer = KimiDeltaAttention(HIDDEN_SIZE, kda_config, layer_idx=0).to(device=device, dtype=dtype)
    hf_layer.eval()

    # Create Fast-LLM KDA layer
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
                        "normalization": {"epsilon": 1e-6, "activation": "sigmoid"},
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
    fast_layer.to(device=device, dtype=dtype)
    fast_layer.eval()

    # Copy weights
    _copy_weights(fast_layer, hf_layer)

    # Verify all parameters match
    hf_state = hf_layer.state_dict()
    for name, fast_param in fast_layer.state_dict().items():
        assert name in hf_state, f"Parameter {name} missing in HF layer"
        hf_param = hf_state[name]
        if fast_param.shape != hf_param.shape:
            hf_param = hf_param.reshape_as(fast_param)
        Assert.all_equal(fast_param, hf_param)

    # Forward passes
    hidden_states = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=dtype, requires_grad=False)

    hf_out = hf_layer(hidden_states)[0]

    fast_kwargs = {
        BlockKwargs.device: device,
        BlockKwargs.sequence_first: False,
        BlockKwargs.sequence_lengths: [[SEQ_LEN] for _ in range(BATCH_SIZE)],
        BlockKwargs.hidden_dims: (HIDDEN_SIZE,),
    }
    fast_layer.preprocess(fast_kwargs)
    fast_out, _ = fast_layer(hidden_states, fast_kwargs)

    # Compare outputs
    Assert.rms_close(fast_out, hf_out, 1e-5)
