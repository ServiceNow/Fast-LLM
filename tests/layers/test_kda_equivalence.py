import pytest
import torch

import fast_llm.layers.ssm.kda as kda_module
from fast_llm.config import UpdateType
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from tests.utils.utils import get_base_model, get_stage, requires_cuda

try:
    from fast_llm_external_models.apriel2.modeling_apriel2 import KimiDeltaAttention
except ImportError:
    KimiDeltaAttention = None

VOCAB_SIZE = 500
HIDDEN_SIZE = 16
SEQ_LEN = 65
NUM_HEADS = 4
HEAD_DIM = 4
KERNEL_SIZE = 4


@pytest.mark.slow
@requires_cuda
@pytest.mark.skipif(KimiDeltaAttention is None, reason="Apriel KDA deps missing")
@pytest.mark.skipif(kda_module.chunk_kda is None, reason="KDA fused kernels not available")
def test_fast_llm_kda_matches_apriel_forward():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    config_dict_hf = {
        "heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "convolution_layer": {"kernel_size": KERNEL_SIZE, "activation": "silu"},
        "normalization": {"epsilon": 1e-5, "activation": "sigmoid"},
    }

    hf_layer = KimiDeltaAttention(HIDDEN_SIZE, config_dict_hf, layer_idx=0).to(device=device, dtype=dtype).eval()

    config = GPTBaseModelConfig.from_dict(
        {
            "decoder": {
                "num_blocks": 1,
                "block": {"mixer": {"type": "kda", **config_dict_hf}},
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
    hf_layer.load_state_dict(fast_layer.state_dict())

    hf_state_dict = hf_layer.state_dict()
    for fast_name, p in fast_layer.state_dict().items():
        print(f"Comparing parameter {fast_name} with shape {p.shape}")
        torch.testing.assert_close(p, hf_state_dict[fast_name], atol=1e-5, rtol=1e-5)

    hidden_states = torch.randn(2, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=dtype, requires_grad=False)
    hf_layer.training = True
    hf_out = hf_layer(hidden_states)[0]

    sequence_lengths = [[SEQ_LEN] for _ in range(hidden_states.size(0))]
    fast_kwargs = {
        BlockKwargs.device: device,
        BlockKwargs.sequence_first: False,
        BlockKwargs.sequence_lengths: sequence_lengths,
        BlockKwargs.hidden_dims: (HIDDEN_SIZE,),
    }
    fast_layer.preprocess(fast_kwargs)
    fast_out, _ = fast_layer(hidden_states, fast_kwargs)

    torch.testing.assert_close(fast_out, hf_out, atol=1e-5, rtol=1e-5)
