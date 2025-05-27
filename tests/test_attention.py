from numpy.core.umath import dtype
import torch

from fast_llm.core.distributed import set_generator
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import StageConfig
from fast_llm.engine.multi_stage.stage import Stage
from fast_llm.functional.config import TritonConfig
from fast_llm.layers.transformer.attention import Attention
from fast_llm.layers.transformer.config import (
    ATTENTION_IMPLEMENTATION_SPECS,
    AttentionImplementation,
    AttentionMode,
    TransformerConfig,
    TransformerDimNames,
    TransformerKwargs,
)
from fast_llm.layers.transformer.preprocessing import FastAttentionPreprocessor, BackupAttentionPreprocessor
from fast_llm.utils import Assert


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
        use_fast_attention=True,
    )
    distributed_cfg = DistributedConfig(training_dtype="bfloat16")
    distributed = Distributed(distributed_cfg, use_cpu=True)
    tensor_space = TensorSpace(distributed_config=distributed_cfg)
    tensor_space.setup(distributed)
    transformer_cfg.setup_tensor_space(tensor_space)
    preprocessor = FastAttentionPreprocessor(transformer_cfg, tensor_space=tensor_space)
    for micro_seq_idx in range(int(sequence_length / micro_sequence_length)):
        kwargs = {
            TransformerKwargs.sequence_q_dim: TensorDim(TransformerDimNames.sequence_k, micro_sequence_length),
            TransformerKwargs.sequence_k_dim: TensorDim(
                TransformerDimNames.sequence_k, (micro_seq_idx + 1) * micro_sequence_length
            ),
            TransformerKwargs.sequence_length: sequence_length,
            TransformerKwargs.sequence_lengths: sequence_lengths,
        }
        preprocessor.preprocess(None, kwargs)
        Assert.all_equal(kwargs[TransformerKwargs.cu_seqlens_q], cumulative_sequences_q[micro_seq_idx])
        Assert.all_equal(kwargs[TransformerKwargs.cu_seqlens_k], cumulative_sequences_k[micro_seq_idx])


# @requires_cuda
# @pytest.mark.slow
def test_mask_mod() -> None:
    batch_size = 2
    sequence_length = 36
    transformer_config = TransformerConfig(
        num_layers=2,
        num_attention_heads=2,
        hidden_size=16,
        add_linear_biases=False,
        attention_mode=AttentionMode.causal,
    )
    distributed_config = DistributedConfig(training_dtype="bfloat16")
    distributed = Distributed(distributed_config, use_cpu=False)
    dtype = distributed_config.training_dtype.torch
    device = distributed.device
    tensor_space = TensorSpace(distributed_config=distributed_config)
    tensor_space.setup(distributed)
    tensor_space.add_tensor_dim(TensorDim(TransformerDimNames.batch, batch_size))
    transformer_config.setup_tensor_space(tensor_space)
    # preprocessor = FastAttentionPreprocessor(config=transformer_config, tensor_space=tensor_space)
    preprocessor = BackupAttentionPreprocessor(config=transformer_config, tensor_space=tensor_space)
    attention = Attention(config=transformer_config, tensor_space=tensor_space, layer_index=1)
    stage = Stage(
        config=StageConfig(),
        base_model=[attention],
        distributed_config=distributed_config,
        begin=0,
        end=1,
        index=0,
    )
    stage.setup(distributed=distributed)
    stage.initialize_weights()
    stage.restore_parameters()
    stage.reset_gradients()
    TritonConfig.TRITON_LINEAR = True

    with set_generator(tensor_space.distributed.tp_generator):
        input_ = torch.randn(
            batch_size,
            sequence_length,
            transformer_config.hidden_size,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
    kwargs = {
        TransformerKwargs.attention_implementation: AttentionImplementation.BACKUP,
        TransformerKwargs.sequence_length: sequence_length,
        TransformerKwargs.sequence_q_dim: TensorDim(TransformerDimNames.sequence_q, sequence_length),
        TransformerKwargs.sequence_k_dim: TensorDim(TransformerDimNames.sequence_k, sequence_length),
        TransformerKwargs.sequence_first: False,
    }
    preprocessor.preprocess(None, kwargs)
    if ATTENTION_IMPLEMENTATION_SPECS[kwargs[TransformerKwargs.attention_implementation]].flex:
        assert TransformerKwargs.block_mask in kwargs
    output_, _ = attention.forward(
        input_=input_,
        kwargs=kwargs,
    )
    print(f"output shape: {output_.shape}")
    print(f"output: {output_}")
