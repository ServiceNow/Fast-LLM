import pytest
import torch

from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.attention.attention import Attention
from fast_llm.layers.attention.config import AttentionConfig, AttentionImplementation, AttentionKwargs
from fast_llm.layers.block.config import BlockDimNames
from fast_llm.utils import Assert
from tests.utils.utils import requires_cuda


# TODO: ====== micro-sequence ======
@pytest.mark.skip
def test_varlen_preprocessing():
    sequence_lengths = [[8, 13, 4, 11], [11, 16, 9]]
    # First micro-sequence:
    # [0...7,0...3] + [0...10,0] -> [0,8,12,23,24]
    # Second micro-sequence:
    # [4...12,0...2] + [1...12] -> [0,9,12,24]
    # Third micro-sequence:
    # [3,0...10] + [13...15, 0...8] -> [1,12,15,24]
    cumulative_sequences_q = [
        torch.tensor([0, 8, 12, 23, 24], dtype=torch.int32),
        torch.tensor([0, 0, 9, 12, 12, 24], dtype=torch.int32),
        torch.tensor([0, 0, 0, 1, 12, 12, 15, 24], dtype=torch.int32),
    ]
    cumulative_sequences_k = [
        torch.tensor([0, 8, 12, 23, 24], dtype=torch.int32),
        torch.tensor([0, 8, 21, 24, 35, 48], dtype=torch.int32),
        torch.tensor([0, 8, 21, 25, 36, 47, 63, 72], dtype=torch.int32),
    ]
    micro_sequence_length = 12
    sequence_length = 36
    attention = Attention(
        AttentionConfig(head_size=64, implementation=AttentionImplementation.flash, cross_document_attention=False),
        DistributedConfig(compute_dtype="bfloat16"),
        hidden_dim=TensorDim("", 1),
        lr_scale=None,
        peft=None,
    )
    for micro_seq_idx in range(int(sequence_length / micro_sequence_length)):
        kwargs = {
            AttentionKwargs.sequence_q_dim: TensorDim(BlockDimNames.sequence_k, micro_sequence_length),
            AttentionKwargs.sequence_k_dim: TensorDim(
                BlockDimNames.sequence_k, (micro_seq_idx + 1) * micro_sequence_length
            ),
            AttentionKwargs.sequence_length: sequence_length,
            AttentionKwargs.sequence_lengths: sequence_lengths,
            AttentionKwargs.device: torch.device("cpu"),
        }
        attention.preprocess(kwargs)
        Assert.all_equal(kwargs[AttentionKwargs.cu_seqlens_q], cumulative_sequences_q[micro_seq_idx])
        Assert.all_equal(kwargs[AttentionKwargs.cu_seqlens_k], cumulative_sequences_k[micro_seq_idx])


@requires_cuda
@pytest.mark.parametrize("cross_document_attention", (True, False))
@pytest.mark.parametrize(("causal", "window_size"), ((True, None), (True, 50), (False, None)))
@pytest.mark.parametrize("padding", (0, 10))
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
