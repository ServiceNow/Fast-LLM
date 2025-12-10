"""Test numerical equivalence between Fast-LLM GDN and Apriel2 GatedDeltaNet."""

import pytest
import torch

from fast_llm.config import UpdateType
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.utils import Assert
from tests.utils.utils import get_base_model, get_stage, requires_cuda

try:
    from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2GatedDeltaNet
except ImportError:
    Apriel2GatedDeltaNet = None

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    _gdn_kernel_available = True
except ImportError:
    _gdn_kernel_available = False

# Test constants
VOCAB_SIZE = 500
HIDDEN_SIZE = 64
SEQ_LEN = 65
BATCH_SIZE = 2
NUM_V_HEADS = 4
NUM_K_HEADS = 2
HEAD_DIM = 16
KERNEL_SIZE = 4


@pytest.mark.slow
@requires_cuda
@pytest.mark.skipif(Apriel2GatedDeltaNet is None, reason="Apriel2 GDN not available")
@pytest.mark.skipif(not _gdn_kernel_available, reason="GDN CUDA kernels not available")
def test_fast_llm_gdn_matches_apriel2_forward():
    """Verify Fast-LLM GDN output matches Apriel2 GatedDeltaNet."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Create Apriel2 GDN layer
    gdn_config = {
        "value_heads": NUM_V_HEADS,
        "key_heads": NUM_K_HEADS,
        "key_head_dim": HEAD_DIM,
        "value_head_dim": HEAD_DIM,
        "convolution_layer": {"kernel_size": KERNEL_SIZE, "activation": "silu"},
        "norm_eps": 1e-5,
    }
    hf_layer = Apriel2GatedDeltaNet(HIDDEN_SIZE, gdn_config, layer_idx=0, dtype=dtype).to(device=device, dtype=dtype)
    hf_layer.eval()

    # Create Fast-LLM GDN layer
    config = GPTBaseModelConfig.from_dict(
        {
            "decoder": {
                "num_blocks": 1,
                "block": {
                    "mixer": {
                        "type": "gdn",
                        "value_heads": NUM_V_HEADS,
                        "key_heads": NUM_K_HEADS,
                        "key_head_dim": HEAD_DIM,
                        "value_head_dim": HEAD_DIM,
                        "convolution_layer": {"kernel_size": KERNEL_SIZE, "activation": "silu"},
                        "normalization": {"epsilon": 1e-5},
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

    # Copy weights: parameter names match exactly, so use load_state_dict
    hf_layer.load_state_dict(fast_layer.state_dict())

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
        BlockKwargs.hidden_dims: (HIDDEN_SIZE,),
        BlockKwargs.sequence_length: SEQ_LEN,
        BlockKwargs.sequence_lengths: [[SEQ_LEN] for _ in range(BATCH_SIZE)],
    }
    fast_layer.preprocess(fast_kwargs)
    fast_out, _ = fast_layer(hidden_states, fast_kwargs)

    # Compare outputs
    Assert.rms_close(fast_out, hf_out, 1e-5)
