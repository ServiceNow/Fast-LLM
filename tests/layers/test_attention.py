import pytest
import torch

from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.attention.attention import Attention
from fast_llm.layers.attention.config import AttentionConfig, AttentionKwargs
from fast_llm.utils import Assert
from tests.utils.utils import requires_cuda


@requires_cuda
@pytest.mark.parametrize("cross_document_attention", (True, False))
@pytest.mark.parametrize(("causal", "window_size"), ((True, None), (True, 50), (False, None)))
def test_attention_implementations(cross_document_attention: bool, causal: bool, window_size: int | None):
    """
    Check that the flash and backup attention implementation give the same result.
    """
    attention: Attention = AttentionConfig(
        head_size=32,
        heads=4,
        head_groups=2,
        window_size=window_size,
        cross_document_attention=cross_document_attention,
        causal=causal,
    ).get_layer(
        DistributedConfig(compute_dtype="bfloat16"),
        TensorDim("hidden_size", 256),
        lr_scale=None,
        peft=None,
    )
    query = torch.empty(4, 100, 4, 32, dtype=torch.bfloat16, device="cuda").normal_()
    key = torch.empty(4, 100, 2, 32, dtype=torch.bfloat16, device="cuda").normal_()
    value = torch.empty(4, 100, 2, 32, dtype=torch.bfloat16, device="cuda").normal_()
    kwargs = {
        AttentionKwargs.device: torch.device("cuda"),
        AttentionKwargs.sequence_length: 100,
        AttentionKwargs.sequence_lengths: [
            [20, 32, 10, 11, 9, 18],
            [100],
            [2, 8, 22, 7, 6, 5, 1, 10, 4, 11, 3, 8, 4, 9],
            [5 for _ in range(20)],
        ],
        AttentionKwargs.sequence_q_dim: TensorDim("sequence_q", 100),
        AttentionKwargs.sequence_k_dim: TensorDim("sequence_k", 100),
    }
    attention._preprocess_for_backup_attention(kwargs)
    attention._preprocess_for_flash_attention(kwargs)

    out_backup = attention._attn_backup(query, key, value, kwargs)
    out_flash = attention._attn_flash(query, key, value, kwargs)

    Assert.rms_close(out_backup, out_flash, 2e-3)
