import pytest
import torch
import transformers

from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.decoder.config import MixerConfig
from fast_llm.layers.ssm.config import GatedDeltaNetConfig, KimiDeltaAttentionConfig, MambaConfig
from fast_llm.layers.ssm.kda import _kda_available
from fast_llm.utils import Assert
from tests.utils.utils import get_stage, requires_cuda

try:
    from fast_llm_external_models.apriel2.modeling_apriel2 import (
        Apriel2GatedDeltaNet,
        Apriel2Mamba,
        KimiDeltaAttention,
    )
except ImportError:
    Apriel2GatedDeltaNet = None
    Apriel2Mamba = None

HIDDEN_SIZE = 16
SEQ_LEN = 65


def _compare_mixers(
    fast_llm_config: MixerConfig, hf_layer: torch.nn.Module, param_map: dict[str, str], threshold=1e-5
):
    distributed = Distributed(
        distributed_config := DistributedConfig(compute_dtype=DataType.bfloat16, use_cuda=torch.cuda.is_available())
    )
    fast_llm_layer = fast_llm_config.get_layer(
        distributed_config,
        TensorDim("", HIDDEN_SIZE),
        lr_scale=None,
        peft=None,
    ).eval()
    get_stage([fast_llm_layer], distributed, [], {})
    hf_layer = hf_layer.to(device=distributed.device, dtype=distributed_config.compute_dtype.torch)

    with torch.no_grad():
        hf_state_dict = hf_layer.state_dict()
        for name, param in fast_llm_layer.named_parameters():
            param.copy_(hf_state_dict[param_map.get(name, name)].view_as(param))

    hf_params = hf_layer.state_dict()
    for name, fast_param in fast_llm_layer.state_dict().items():
        hf_param = hf_params[param_map.get(name, name)]
        Assert.rms_close_relative(fast_param, hf_param.view_as(fast_param), threshold, 1e-5, msg=name)

    hidden_states = torch.randn(
        2,
        SEQ_LEN,
        HIDDEN_SIZE,
        device=distributed.device,
        dtype=distributed_config.compute_dtype.torch,
        requires_grad=False,
    )

    hf_layer.train()
    hf_out = hf_layer(hidden_states)
    if isinstance(hf_out, tuple):
        (hf_out,) = hf_out

    sequence_lengths = [[SEQ_LEN] for _ in range(hidden_states.size(0))]
    fast_kwargs = {
        BlockKwargs.device: distributed.device,
        BlockKwargs.sequence_first: False,
        BlockKwargs.sequence_lengths: sequence_lengths,
        BlockKwargs.hidden_dims: (HIDDEN_SIZE,),
        BlockKwargs.sequence_q_dim: TensorDim("", SEQ_LEN),
        BlockKwargs.sequence_k_dim: TensorDim("", SEQ_LEN),
    }
    fast_llm_layer.train()
    fast_llm_layer.preprocess(fast_kwargs)
    fast_out = fast_llm_layer(hidden_states, fast_kwargs)

    Assert.rms_close_relative(fast_out, hf_out, threshold, 1e-5)


@pytest.mark.slow
# Arguments ('seq_idx',) not implemented for torch implementation of 1d convolution.
@pytest.mark.skipif(not transformers.utils.import_utils.is_causal_conv1d_available(), reason="GDN deps missing")
def test_gdn():
    dtype = torch.bfloat16

    NUM_V_HEADS = 4
    NUM_K_HEADS = 2
    HEAD_DIM = 4
    KERNEL_SIZE = 4

    config_common = {
        "value_heads": NUM_V_HEADS,
        "key_heads": NUM_K_HEADS,
        "key_head_dim": HEAD_DIM,
        "value_head_dim": HEAD_DIM,
        "convolution_layer": {"kernel_size": KERNEL_SIZE, "activation": "silu"},
    }

    hf_layer = (
        Apriel2GatedDeltaNet(HIDDEN_SIZE, {**config_common, "norm_eps": 1e-5}, layer_idx=0, dtype=dtype)
        .to(device="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype)
        .eval()
    )
    fast_llm_config = GatedDeltaNetConfig.from_dict(config_common, {"normalization": {"epsilon": 1e-5}})
    _compare_mixers(fast_llm_config, hf_layer, {})


@pytest.mark.slow
@pytest.mark.skipif(not _kda_available, reason="KDA fused kernels not available")
def test_kda():
    NUM_HEADS = 4
    HEAD_DIM = 4
    KERNEL_SIZE = 4

    kda_config = {
        "heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "convolution_layer": {"kernel_size": KERNEL_SIZE, "activation": "silu"},
        "normalization": {"epsilon": 1e-5, "activation": "sigmoid"},
    }

    hf_layer = KimiDeltaAttention(HIDDEN_SIZE, kda_config, layer_idx=0)

    fast_llm_config = KimiDeltaAttentionConfig.from_dict(kda_config, {})

    _compare_mixers(fast_llm_config, hf_layer, {})


@pytest.mark.slow
@pytest.mark.parametrize("add_linear_biases", [True, False])
@pytest.mark.parametrize("repeat_kv_before_conv", [True, False])
@pytest.mark.skipif(not transformers.utils.import_utils.is_mamba_ssm_available(), reason="Mamba not available")
def test_mamba(add_linear_biases, repeat_kv_before_conv):
    D_INNER = 128
    D_XB = 64
    D_STATE = 16
    D_CONV = 4
    DT_RANK = 4

    config_common = {
        "d_inner": D_INNER,
        "d_xb": D_XB,
        "state_size": D_STATE,
        "dt_rank": DT_RANK,
        "repeat_kv_before_conv": repeat_kv_before_conv,
        "add_linear_biases": add_linear_biases,
    }

    mamba_config = {
        "conv_bias": add_linear_biases,
        "dt_proj_bias": add_linear_biases,
        **config_common,
    }
    hf_layer = Apriel2Mamba(HIDDEN_SIZE, mamba_config, layer_idx=0)

    # Create Fast-LLM Mamba layer
    fast_llm_config = MambaConfig(
        convolution_layer={"kernel_size": D_CONV},
        **config_common,
    )

    param_map = {
        "convolution.weight": "conv1d.weight",
        "convolution.bias": "conv1d.bias",
    }
    # TODO: This is a really high threshold.
    _compare_mixers(fast_llm_config, hf_layer, param_map, threshold=1e-2)
