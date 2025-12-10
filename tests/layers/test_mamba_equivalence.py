"""Test numerical equivalence between Fast-LLM Mamba2 and Apriel2 Mamba.

Note: Fast-LLM's "mamba_2" type is actually a Mamba 1 variant (not the true Mamba 2
architecture). It corresponds to the HuggingFace/Apriel Mamba implementation.
"""

import pytest
import torch

from fast_llm.config import UpdateType
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.ssm.config import Mamba2Config  # Ensures mamba_2 type is registered
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.utils import Assert
from tests.utils.utils import get_base_model, get_stage, requires_cuda

# Ensure Mamba2Config is registered for dynamic type lookup
_ = Mamba2Config

try:
    from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2Mamba
except ImportError:
    Apriel2Mamba = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    _mamba_kernel_available = True
except (ImportError, RuntimeError):
    _mamba_kernel_available = False

# Test constants
VOCAB_SIZE = 500
HIDDEN_SIZE = 64
SEQ_LEN = 65
BATCH_SIZE = 2
D_INNER = 128
D_XB = 64
D_STATE = 16
D_CONV = 4
DT_RANK = 4


def _copy_weights(fast_layer, hf_layer):
    """Copy weights from Apriel2 Mamba to Fast-LLM Mamba2."""
    with torch.no_grad():
        # Main projections
        fast_layer.in_proj.weight.copy_(hf_layer.in_proj.weight)
        if fast_layer.in_proj.bias is not None and hf_layer.in_proj.bias is not None:
            fast_layer.in_proj.bias.copy_(hf_layer.in_proj.bias)

        # DT projections
        fast_layer.dt_in_proj.weight.copy_(hf_layer.dt_in_proj.weight)
        if fast_layer.dt_in_proj.bias is not None and hf_layer.dt_in_proj.bias is not None:
            fast_layer.dt_in_proj.bias.copy_(hf_layer.dt_in_proj.bias)

        fast_layer.dt_proj.weight.copy_(hf_layer.dt_proj.weight)
        if fast_layer.dt_proj.bias is not None and hf_layer.dt_proj.bias is not None:
            fast_layer.dt_proj.bias.copy_(hf_layer.dt_proj.bias)

        # Convolution (Fast-LLM uses "convolution", Apriel2 uses "conv1d")
        fast_layer.convolution.weight.copy_(hf_layer.conv1d.weight)
        if fast_layer.convolution.bias is not None and hf_layer.conv1d.bias is not None:
            fast_layer.convolution.bias.copy_(hf_layer.conv1d.bias)

        # SSM parameters
        fast_layer.A_log.copy_(hf_layer.A_log)
        fast_layer.D.copy_(hf_layer.D)

        # Output projection
        fast_layer.out_proj.weight.copy_(hf_layer.out_proj.weight)
        if fast_layer.out_proj.bias is not None and hf_layer.out_proj.bias is not None:
            fast_layer.out_proj.bias.copy_(hf_layer.out_proj.bias)


@pytest.mark.slow
@requires_cuda
@pytest.mark.skipif(Apriel2Mamba is None, reason="Apriel2 Mamba not available")
@pytest.mark.skipif(not _mamba_kernel_available, reason="Mamba CUDA kernels not available")
@pytest.mark.parametrize("add_linear_biases", [True, False])
@pytest.mark.parametrize("repeat_kv_before_conv", [True, False])
def test_fast_llm_mamba2_matches_apriel2(add_linear_biases, repeat_kv_before_conv):
    """Verify Fast-LLM Mamba2 output matches Apriel2 Mamba.

    Args:
        add_linear_biases: Whether to add biases to linear layers.
        repeat_kv_before_conv: Whether to repeat KV before or after convolution.
    """
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Create Apriel2 Mamba layer
    # Note: Apriel2 has separate conv_bias and dt_proj_bias controls.
    # We align them with Fast-LLM's single add_linear_biases flag.
    mamba_config = {
        "d_inner": D_INNER,
        "d_xb": D_XB,
        "state_size": D_STATE,
        "d_conv": D_CONV,
        "dt_rank": DT_RANK,
        "conv_bias": add_linear_biases,
        "dt_proj_bias": add_linear_biases,
        "add_linear_biases": add_linear_biases,
        "repeat_kv_before_conv": repeat_kv_before_conv,
        "dt_min": 0.001,
        "dt_max": 0.1,
        "dt_init_floor": 1e-4,
    }
    hf_layer = Apriel2Mamba(HIDDEN_SIZE, mamba_config, layer_idx=0, dtype=dtype).to(device=device, dtype=dtype)
    hf_layer.eval()

    # Create Fast-LLM Mamba2 layer
    config = GPTBaseModelConfig.from_dict(
        {
            "decoder": {
                "num_blocks": 1,
                "block": {
                    "mixer": {
                        "type": "mamba_2",
                        "d_inner": D_INNER,
                        "d_xb": D_XB,
                        "state_size": D_STATE,
                        "convolution_layer": {"kernel_size": D_CONV},
                        "dt_rank": DT_RANK,
                        "add_linear_biases": add_linear_biases,
                        "repeat_kv_before_conv": repeat_kv_before_conv,
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

    # Verify key parameters match (not all names match between implementations)
    Assert.all_equal(fast_layer.in_proj.weight, hf_layer.in_proj.weight)
    Assert.all_equal(fast_layer.convolution.weight, hf_layer.conv1d.weight)
    Assert.all_equal(fast_layer.A_log, hf_layer.A_log)
    Assert.all_equal(fast_layer.D, hf_layer.D)
    Assert.all_equal(fast_layer.out_proj.weight, hf_layer.out_proj.weight)

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

    # Compare outputs (slightly looser tolerance for Mamba due to numerical differences)
    Assert.rms_close(fast_out, hf_out, 1e-4)
