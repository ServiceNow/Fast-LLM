import pytest
import torch

from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.language_model import LanguageModelBatch
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.decoder.config import MixerConfig
from fast_llm.layers.ssm.config import GatedDeltaNetConfig, KimiDeltaAttentionConfig, MambaConfig
from fast_llm.layers.ssm.gdn import _causal_conv1d_available
from fast_llm.layers.ssm.kda import _kda_available
from fast_llm.utils import Assert
from tests.utils.utils import get_stage


# TODO: include mamba varlen
@pytest.mark.slow
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            MambaConfig(
                d_inner=128,
                d_xb=64,
                state_size=16,
                dt_rank=8,
            ),
            marks=pytest.mark.skip("Mamba varlen kernel not available"),
        ),
        pytest.param(
            GatedDeltaNetConfig(value_heads=4, key_heads=2, key_head_dim=16, value_head_dim=16),
            marks=pytest.mark.skipif(not _causal_conv1d_available, reason="GDN not available"),
        ),
        pytest.param(
            KimiDeltaAttentionConfig(heads=4, head_dim=16),
            marks=pytest.mark.skipif(not _kda_available, reason="KDA not available"),
        ),
    ],
)
@pytest.mark.parametrize("lengths", ([6, 9], [4, 1, 10]))
def test_mixer_varlen_stacking_equivalence(config: MixerConfig, lengths: list[int]):
    """
    Check that Gated Delta Net forward/backward match with and without packing.
    """
    hidden_dim = TensorDim("hidden", hidden_size := 32)
    distributed = Distributed(
        distributed_config := DistributedConfig(compute_dtype=DataType.float16, use_cuda=torch.cuda.is_available())
    )
    mixer = config.get_layer(distributed_config, hidden_dim, lr_scale=None, peft=None, return_bias=False)
    stage = get_stage([mixer], distributed)

    num_tokens = sum(lengths)

    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        device=distributed.device,
        dtype=distributed_config.compute_dtype.torch,
        requires_grad=True,
    )

    (model_input_packed,) = LanguageModelBatch(
        tokens=torch.empty(num_tokens, dtype=torch.int64, device=distributed.device), lengths=lengths
    ).get_model_inputs(
        LanguageModelBatchPreprocessingConfig(
            distributed=distributed_config,
            predicted_tokens=0,
            **mixer.get_preprocessing_config(),
        )
    )
    kwargs_packed = model_input_packed.to_kwargs()
    mixer.preprocess(kwargs_packed)

    out_packed, context = stage.forward(hidden_states, kwargs_packed)
    stage.backward(torch.ones_like(out_packed), context)

    names, parameters = zip(*list(mixer.named_parameters()))
    grads_packed = [parameter.grad_buffer.clone() for parameter in parameters]

    stage.reset_gradients()
    # Run reference path separately per sequence without varlen packing, then concatenate.
    out_refs = []
    for length, hidden_states_ in zip(lengths, torch.split(hidden_states, lengths, dim=0), strict=True):
        (model_input_unpacked,) = LanguageModelBatch(
            tokens=torch.empty(length, dtype=torch.int64, device=distributed.device), lengths=[length]
        ).get_model_inputs(
            LanguageModelBatchPreprocessingConfig(
                distributed=distributed_config,
                predicted_tokens=0,
                **mixer.get_preprocessing_config(),
            )
        )
        kwargs_unpacked = model_input_unpacked.to_kwargs()
        mixer.preprocess(kwargs_unpacked)
        out, context = stage.forward(hidden_states_, kwargs_unpacked)
        stage.backward(torch.ones_like(out), context)
        out_refs.append(out)
    out_ref = torch.cat(out_refs, dim=0)

    Assert.rms_close_relative(out_packed, out_ref, 1e-3, 1e-4)

    for name, parameter, grad_packed in zip(names, parameters, grads_packed, strict=True):
        Assert.rms_close_relative(grad_packed, parameter.grad_buffer, 1e-3, 1e-4, msg=name)
