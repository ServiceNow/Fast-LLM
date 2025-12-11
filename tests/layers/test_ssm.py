import pytest
import torch

from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.decoder.config import MixerConfig
from fast_llm.layers.ssm import kda as kda_module
from fast_llm.layers.ssm.config import GatedDeltaNetConfig, KimiDeltaAttentionConfig
from fast_llm.utils import Assert
from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2GatedDeltaNet
from fast_llm_external_models.apriel_hybrid_ssm.configuration_apriel_hybrid_ssm import AprielHybridSSMConfig
from fast_llm_external_models.apriel_hybrid_ssm.modeling_apriel_hybrid_ssm import KimiDeltaAttention
from tests.utils.utils import get_stage, requires_cuda

HIDDEN_SIZE = 16
SEQ_LEN = 65
NUM_HEADS = 4
NUM_V_HEADS = 4
NUM_K_HEADS = 2
HEAD_DIM = 4
KERNEL_SIZE = 4


def _compare_mixers(fast_llm_config: MixerConfig, hf_layer: torch.nn.Module, param_map: dict[str, str]):
    distributed = Distributed(distributed_config := DistributedConfig(compute_dtype=DataType.bfloat16))
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
        Assert.rms_close_relative(fast_param, hf_param.view_as(fast_param), 1e-5, 1e-5, msg=name)

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
    print("AAAA", fast_out.shape, [x.shape for x in hf_out])

    Assert.rms_close_relative(fast_out, hf_out, 1e-5, 1e-5)


@pytest.mark.slow
@requires_cuda
def test_gdn():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    config_common = {
        "value_heads": NUM_V_HEADS,
        "key_heads": NUM_K_HEADS,
        "key_head_dim": HEAD_DIM,
        "value_head_dim": HEAD_DIM,
        "convolution_layer": {"kernel_size": KERNEL_SIZE, "activation": "silu"},
    }

    hf_layer = (
        Apriel2GatedDeltaNet(HIDDEN_SIZE, {**config_common, "norm_eps": 1e-5}, layer_idx=0, dtype=dtype)
        .to(device=device, dtype=dtype)
        .eval()
    )
    fast_llm_config = GatedDeltaNetConfig.from_dict(config_common, {"normalization": {"epsilon": 1e-5}})
    _compare_mixers(fast_llm_config, hf_layer, {})


@pytest.mark.slow
@requires_cuda
@pytest.mark.skipif(KimiDeltaAttention is None or AprielHybridSSMConfig is None, reason="Apriel KDA deps missing")
@pytest.mark.skipif(kda_module.chunk_kda is None, reason="KDA fused kernels not available")
def test_kda():
    hf_config = AprielHybridSSMConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=1,
        rms_norm_eps=1e-6,
    )
    hf_config.short_conv_kernel_size = KERNEL_SIZE
    hf_config.head_dim = HEAD_DIM
    hf_config.num_heads = NUM_HEADS
    hf_layer = KimiDeltaAttention(hf_config, layer_idx=0)

    fast_llm_config = KimiDeltaAttentionConfig(
        heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        convolution_layer={"kernel_size": KERNEL_SIZE, "activation": "silu"},
        normalization={"epsilon": 1e-6, "activation": "sigmoid"},
    )

    param_map = {
        "q_conv.weight": "q_conv1d.weight",
        "k_conv.weight": "k_conv1d.weight",
        "v_conv.weight": "v_conv1d.weight",
        "beta_proj.weight": "b_proj.weight",
        "norm.weight": "o_norm.weight",
    }
    _compare_mixers(fast_llm_config, hf_layer, param_map)
