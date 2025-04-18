import unittest.mock

import torch

from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.transformer.attention import Attention
from fast_llm.layers.transformer.config import TransformerConfig, TransformerDimNames, TransformerKwargs
from fast_llm.layers.transformer.preprocessing import FlashAttnVarlenPreprocessor
from fast_llm.utils import Assert


def test_decide_window_size():
    attention = unittest.mock.Mock(spec=Attention)
    attention._decide_window_size = Attention._decide_window_size.__get__(attention)  # Attach real method

    # Arrange - Case 1: window_size is returned (layer_index >= max_window_layers)
    attention._config = TransformerConfig(window_size=512, max_window_layers=2)
    attention._layer_index = 2
    assert attention._decide_window_size() == 512

    # Arrange - Case 2: window_size is None (layer_index < max_window_layers)
    attention._config = TransformerConfig(window_size=512, max_window_layers=2)
    attention._layer_index = 1
    assert attention._decide_window_size() is None

    # Arrange - Case 3: max_window_layers is None (always return window_size)
    attention._config = TransformerConfig(window_size=512, max_window_layers=None)
    assert attention._decide_window_size() == 512


def test_attention_constructor():
    transformer_conf = TransformerConfig(
        num_layers=2,
        num_attention_heads=2,
        hidden_size=16,
    )
    distributed_config = DistributedConfig()
    tensor_space = TensorSpace(distributed_config=distributed_config)
    transformer_conf.setup_tensor_space(tensor_space)

    Attention(transformer_conf, tensor_space, 1)


def test_varlen_preprocessor():
    sequence_lengths = [torch.tensor([8, 13, 4, 11], dtype=torch.int32), torch.tensor([11, 16, 9], dtype=torch.int32)]
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
    transformer_cfg = TransformerConfig(
        num_layers=2,
        num_attention_heads=2,
        hidden_size=16,
        use_flash_attention=True,
    )
    distributed_cfg = DistributedConfig(training_dtype="bfloat16")
    distributed = Distributed(distributed_cfg, use_cpu=True)
    tensor_space = TensorSpace(distributed_config=distributed_cfg)
    tensor_space.setup(distributed)
    transformer_cfg.setup_tensor_space(tensor_space)
    varlen_preprocessor = FlashAttnVarlenPreprocessor(transformer_cfg, tensor_space=tensor_space)
    for micro_seq_idx in range(int(sequence_length / micro_sequence_length)):
        kwargs = {
            TransformerKwargs.sequence_q_dim: TensorDim(TransformerDimNames.sequence_k, micro_sequence_length),
            TransformerKwargs.sequence_k_dim: TensorDim(
                TransformerDimNames.sequence_k, (micro_seq_idx + 1) * micro_sequence_length
            ),
            TransformerKwargs.sequence_length: sequence_length,
            TransformerKwargs.sequence_lengths: sequence_lengths,
        }
        varlen_preprocessor.preprocess(None, kwargs)
        Assert.all_equal(kwargs[TransformerKwargs.cu_seqlens_q], cumulative_sequences_q[micro_seq_idx])
        Assert.all_equal(kwargs[TransformerKwargs.cu_seqlens_k], cumulative_sequences_k[micro_seq_idx])
