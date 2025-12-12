import pytest
import torch

from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.decoder.config import MixerConfig
from fast_llm.layers.ssm import gdn as gdn_module
from fast_llm.layers.ssm import kda as kda_module
from fast_llm.layers.ssm.config import GatedDeltaNetConfig, KimiDeltaAttentionConfig, Mamba2Config
from fast_llm.utils import Assert
from tests.utils.utils import get_stage, requires_cuda


# TODO: include mamba varlen
@pytest.mark.slow
@requires_cuda
@pytest.mark.parametrize(
    "config",
    [
        AttentionConfig(heads=4, head_groups=2, head_size=16, cross_document_attention=False),
        Mamba2Config(
            d_inner=128,
            d_xb=64,
            state_size=16,
            dt_rank=8,
            cross_document_attention=False,
            marks=pytest.mark.skip("Mamba varlen kernel not available"),
        ),
        pytest.param(
            GatedDeltaNetConfig(value_heads=4, key_heads=2, key_head_dim=16, value_head_dim=16),
            marks=pytest.mark.skipif(
                gdn_module.chunk_gated_delta_rule is None, reason="GDN fused kernels not available"
            ),
        ),
        pytest.param(
            KimiDeltaAttentionConfig(heads=4, head_dim=16),
            marks=pytest.mark.skipif(kda_module.chunk_kda is None, reason="KDA fused kernels not available"),
        ),
    ],
)
def test_mixer_varlen_stacking_equivalence(config: MixerConfig):
    """
    Check that Gated Delta Net forward/backward match with and without packing.
    """
    hidden_size = 32
    hidden_dim = TensorDim("hidden", hidden_size)
    distributed = Distributed(distributed_config := DistributedConfig(compute_dtype=DataType.float16))
    mixer = config.get_layer(distributed_config, hidden_dim, lr_scale=None, peft=None, return_bias=False)
    stage = get_stage([mixer], distributed)

    batch_size = 2  # cu_seqlens path requires flattened batch
    seq_len = 15

    sequence_lengths = [[6, 9], [4, 1, 10]]
    hidden_states = torch.randn(
        batch_size,
        seq_len,
        hidden_size,
        device=distributed.device,
        dtype=distributed_config.compute_dtype.torch,
        requires_grad=True,
    )

    kwargs = {
        BlockKwargs.device: distributed.device,
        BlockKwargs.sequence_first: False,
        BlockKwargs.hidden_dims: (hidden_dim,),
        BlockKwargs.sequence_q_dim: TensorDim("", seq_len),
        BlockKwargs.sequence_k_dim: TensorDim("", seq_len),
    }

    kwargs_packed = {**kwargs, BlockKwargs.sequence_lengths: sequence_lengths}
    mixer.preprocess(kwargs_packed)

    out_packed, context = stage.forward(hidden_states, kwargs_packed)
    stage.backward(torch.ones_like(out_packed), context)

    names, parameters = zip(*list(mixer.named_parameters()))
    grads_packed = [parameter.grad_buffer.clone() for parameter in parameters]

    stage.reset_gradients()
    # Run reference path separately per sequence without varlen packing, then concatenate.
    out_refs = []
    for i in range(batch_size):
        for seq in torch.split(hidden_states[i], sequence_lengths[i], dim=0):
            kwargs_seq = {**kwargs, BlockKwargs.sequence_lengths: [[len(seq)]]}
            mixer.preprocess(kwargs_seq)
            out, context = stage.forward(seq.unsqueeze(0), kwargs_seq)
            stage.backward(torch.ones_like(out), context)
            out_refs.append(out)
    out_ref = torch.cat(out_refs, dim=1).view_as(out_packed)

    Assert.rms_close_relative(out_packed, out_ref, 1e-3, 1e-4)

    for name, parameter, grad_packed in zip(names, parameters, grads_packed, strict=True):
        Assert.rms_close_relative(grad_packed, parameter.grad_buffer, 1e-3, 1e-4, msg=name)


if __name__ == "__main__":
    pytest.main([__file__])
