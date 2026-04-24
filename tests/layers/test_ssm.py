import pytest
import torch
import transformers

from fast_llm.data.document.block import BlockModelInput, LengthModelInputPreprocessor
from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.decoder.config import MixerConfig
from fast_llm.layers.ssm.config import GatedDeltaNetConfig, KimiDeltaAttentionConfig, MambaConfig
from fast_llm.layers.ssm.gdn import _fast_gdn_available
from fast_llm.layers.ssm.kda import _kda_available
from fast_llm.utils import Assert
from tests.utils.utils import get_stage

try:
    from fast_llm_external_models.apriel2.modeling_apriel2 import (
        Apriel2GatedDeltaNet,
        Apriel2Mamba,
        KimiDeltaAttention,
        _gdn_fla_available,
        _kda_fla_available,
        is_fast_path_available,
    )
except ImportError:
    Apriel2GatedDeltaNet = None
    Apriel2Mamba = None
    KimiDeltaAttention = None
    _gdn_fla_available = False
    _kda_fla_available = False
    is_fast_path_available = False

HIDDEN_SIZE = 16
SEQUENCE_LENGTH = 65
BATCH_SIZE = 2


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
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        HIDDEN_SIZE,
        device=distributed.device,
        dtype=distributed_config.compute_dtype.torch,
        requires_grad=False,
    )

    model_input = BlockModelInput()
    LengthModelInputPreprocessor(
        lengths=[SEQUENCE_LENGTH for _ in range(hidden_states.size(0))],
        sequence_k_past=0,
        first_document_begin=0,
        last_document_end=BATCH_SIZE * SEQUENCE_LENGTH,
        device=distributed.device,
        unpadded_length=BATCH_SIZE * SEQUENCE_LENGTH,
        sequence_length=BATCH_SIZE * SEQUENCE_LENGTH,
    ).preprocess(
        model_input,
        LanguageModelBatchPreprocessingConfig(
            distributed=distributed_config,
            predicted_tokens=0,
            **fast_llm_layer.get_preprocessing_config(),
        ),
    )
    kwargs = model_input.to_kwargs()

    hf_layer.train()
    hf_out = hf_layer(hidden_states)
    if isinstance(hf_out, tuple):
        (hf_out,) = hf_out

    fast_llm_layer.train()
    fast_out = fast_llm_layer(hidden_states.flatten(0, 1), kwargs).view_as(hidden_states)

    Assert.rms_close_relative(fast_out, hf_out, threshold, 1e-5)


@pytest.mark.slow
# Arguments ('seq_idx',) not implemented for torch implementation of 1d convolution.
@pytest.mark.skipif(not is_fast_path_available, reason="GDN deps missing")
@pytest.mark.parametrize(
    "use_backup",
    [
        pytest.param(False, marks=pytest.mark.skipif(not _fast_gdn_available, reason="FLA not available")),
        pytest.param(True, marks=pytest.mark.skipif(not _gdn_fla_available, reason="GDN fla kernels not available")),
    ],
    ids=["fast", "backup"],
)
def test_gdn(testing_device, use_backup, monkeypatch):
    if use_backup:
        import fast_llm.layers.ssm.gdn as gdn_module

        monkeypatch.setattr(gdn_module, "_fast_gdn_available", False)

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
        .to(device=testing_device, dtype=dtype)
        .eval()
    )
    fast_llm_config = GatedDeltaNetConfig.from_dict(config_common, {"normalization": {"epsilon": 1e-5}})
    # The backup uses float32 arithmetic while the reference uses the FLA kernel, so
    # bfloat16-level numerical differences are expected; use a looser threshold.
    _compare_mixers(fast_llm_config, hf_layer, {}, threshold=1e-2 if use_backup else 1e-5)


@pytest.mark.slow
@pytest.mark.skipif(KimiDeltaAttention is None, reason="KDA external model not available")
@pytest.mark.parametrize(
    "use_backup",
    [
        pytest.param(False, marks=pytest.mark.skipif(not _kda_available, reason="KDA fused kernels not available")),
        pytest.param(True, marks=pytest.mark.skipif(not _kda_fla_available, reason="KDA fla kernels not available")),
    ],
    ids=["fast", "backup"],
)
def test_kda(testing_device, use_backup, monkeypatch):
    if use_backup:
        import fast_llm.layers.ssm.kda as kda_module

        monkeypatch.setattr(kda_module, "_kda_available", False)

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

    # The backup uses float32 arithmetic while the reference uses FLA kernels, so
    # bfloat16-level numerical differences are expected; use a looser threshold.
    _compare_mixers(fast_llm_config, hf_layer, {}, threshold=1e-2 if use_backup else 1e-5)


@pytest.mark.slow
@pytest.mark.skip("Mamba is broken")
@pytest.mark.parametrize("add_linear_biases", [True, False])
@pytest.mark.parametrize("repeat_kv_before_conv", [True, False])
@pytest.mark.skipif(not transformers.utils.import_utils.is_mamba_ssm_available(), reason="Mamba not available")
def test_mamba(add_linear_biases, repeat_kv_before_conv):

    config_common = {
        "d_inner": 128,
        "d_xb": 64,
        "state_size": 16,
        "dt_rank": 4,
        "repeat_kv_before_conv": repeat_kv_before_conv,
        "add_linear_biases": add_linear_biases,
    }

    mamba_config = {
        "conv_bias": add_linear_biases,
        "dt_proj_bias": add_linear_biases,
        "d_conv": 4**config_common,
    }
    hf_layer = Apriel2Mamba(HIDDEN_SIZE, mamba_config, layer_idx=0)

    # Create Fast-LLM Mamba layer
    fast_llm_config = MambaConfig(
        convolution_layer={"kernel_size": 4},
        **config_common,
    )

    param_map = {
        "convolution.weight": "conv1d.weight",
        "convolution.bias": "conv1d.bias",
    }
    # TODO: This is a really high threshold.
    _compare_mixers(fast_llm_config, hf_layer, param_map, threshold=1e-2)
