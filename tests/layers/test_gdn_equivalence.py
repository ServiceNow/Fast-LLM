import pytest
import torch

from fast_llm.config import UpdateType
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2GatedDeltaNet
from tests.utils.utils import get_base_model, get_stage, requires_cuda

VOCAB_SIZE = 500
HIDDEN_SIZE = 16
SEQ_LEN = 65
NUM_V_HEADS = 4
NUM_K_HEADS = 2
HEAD_DIM = 4
KERNEL_SIZE = 4


@pytest.mark.slow
@requires_cuda
def test_fast_llm_gdn_matches_qwen3_next_forward():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    config_dict_hf = {
        "num_value_heads": NUM_V_HEADS,
        "num_key_heads": NUM_K_HEADS,
        "key_head_dim": HEAD_DIM,
        "value_head_dim": HEAD_DIM,
        "conv_kernel_size": KERNEL_SIZE,
        "activation": "silu",
        "norm_eps": 1e-5,
    }

    hf_layer = (
        Apriel2GatedDeltaNet(HIDDEN_SIZE, config_dict_hf, layer_idx=0, dtype=dtype)
        .to(device=device, dtype=dtype)
        .eval()
    )

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
        fast_layer.in_proj_qkvz.weight.copy_(hf_layer.gdn.in_proj_qkvz.weight)
        fast_layer.in_proj_ba.weight.copy_(hf_layer.gdn.in_proj_ba.weight)
        fast_layer.convolution.weight.copy_(hf_layer.gdn.conv1d.weight)
        if fast_layer.convolution.bias is not None and hf_layer.gdn.conv1d.bias is not None:
            fast_layer.convolution.bias.copy_(hf_layer.gdn.conv1d.bias)
        fast_layer.out_proj.weight.copy_(hf_layer.gdn.out_proj.weight)
        fast_layer.A_log.copy_(hf_layer.gdn.A_log)
        fast_layer.dt_bias.copy_(hf_layer.gdn.dt_bias)
        fast_layer.norm.weight.copy_(hf_layer.gdn.norm.weight)

    hidden_states = torch.randn(1, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=dtype, requires_grad=False)

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
    hf_state_dict = hf_layer.gdn.state_dict()
    for k, p in fast_layer.state_dict().items():
        torch.testing.assert_close(p, hf_state_dict[param_map[k]], atol=1e-5, rtol=1e-5)

    # need to monkey patch the hf implementation with our fix_query_key_value_ordering due to the layout differences
    hf_layer.gdn.fix_query_key_value_ordering = fast_layer.fix_query_key_value_ordering
    hf_layer._local_key_heads = fast_layer._local_key_heads
    hf_layer._local_value_heads = fast_layer._local_value_heads
    hf_layer._config = fast_layer._config

    hf_out = hf_layer(hidden_states)[0]

    sequence_lengths = [[SEQ_LEN] for _ in range(hidden_states.size(0))]
    fast_kwargs = {
        BlockKwargs.device: device,
        BlockKwargs.sequence_first: False,
        BlockKwargs.hidden_dims: (HIDDEN_SIZE,),
        BlockKwargs.sequence_length: SEQ_LEN,
        BlockKwargs.sequence_lengths: sequence_lengths,
    }
    fast_layer.preprocess(fast_kwargs)
    fast_out, _ = fast_layer(hidden_states, fast_kwargs)

    torch.testing.assert_close(fast_out, hf_out, atol=1e-5, rtol=1e-5)
