"""
Parity tests for Ultravox v0.5 audio-encoder integration.

Covers:
  - The Step-1 refactor: HF ``encoder.layer_norm`` lands in
    ``audio_encoder.final_norm.norm.{weight,bias}``, NOT in
    ``audio_encoder.adapter.norm_1`` (the legacy slot).
  - End-to-end numerical parity between HF ``UltravoxProjector``
    (verbatim from ``scripts/ultravox_model.py``) and Fast-LLM's
    ``AudioAdapter`` configured for Ultravox: same RMSNorm pre/mid, SwiGLU,
    biasless linears, no dropout, swap-on-import for the gated linear_1.
"""

import math

import pytest
import torch

from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.layers.audio_encoder.adapter import AudioAdapter
from fast_llm.layers.audio_encoder.config import AudioEncoderConfig, AudioKwargs
from fast_llm.models.multimodal.conversion.ultravox import (
    SwapGatedHalvesWeightConverter,
    UltravoxAudioAdapterConverter,
    UltravoxAudioEncoderConverter,
)
from tests.utils.utils import requires_cuda


# ----------------------------------------------------------------------
# Step 1: encoder.layer_norm → AudioEncoder.final_norm.norm
# ----------------------------------------------------------------------


def test_whisper_encoder_layer_norm_routes_to_final_norm():
    """
    The new converter must map ``encoder.layer_norm.*`` to
    ``audio_encoder.final_norm.norm.*`` and must NOT emit a mapping to
    ``audio_encoder.adapter.norm_1`` for it.
    """
    from fast_llm.models.multimodal.conversion.whisper import WhisperAudioEncoderConverter

    # Minimal Whisper audio config — only fields the encoder converter reads.
    audio_config = {
        "d_model": 16,
        "num_hidden_layers": 1,
        "encoder_layers": 1,
        "encoder_ffn_dim": 32,
        "encoder_attention_heads": 4,
        "num_mel_bins": 80,
        "max_source_positions": 4,
        "activation_function": "gelu",
        "layer_norm_eps": 1e-5,
    }
    cfg = AudioEncoderConfig.from_dict(WhisperAudioEncoderConverter.import_config(audio_config))
    converters = WhisperAudioEncoderConverter.get_converters(cfg)

    pairs = {(c.fast_llm_name[0], c.export_name[0]) for c in converters}
    assert ("audio_encoder.final_norm.norm.weight", "encoder.layer_norm.weight") in pairs
    assert ("audio_encoder.final_norm.norm.bias", "encoder.layer_norm.bias") in pairs
    # No converter should send encoder.layer_norm into the adapter slot.
    assert not any(
        fl.startswith("audio_encoder.adapter.norm_1") and hf == "encoder.layer_norm.weight"
        for fl, hf in pairs
    )


# ----------------------------------------------------------------------
# Step 2 + Step 3: Ultravox projector ↔ Fast-LLM AudioAdapter numerical parity
# ----------------------------------------------------------------------


class _HFRMSNorm(torch.nn.Module):
    """RMSNorm matching HF ``LlamaRMSNorm`` with the Ultravox init scalar."""

    def __init__(self, hidden: int, init: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((hidden,), float(init)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class _HFStackAudioFrames(torch.nn.Module):
    def __init__(self, stack_factor: int):
        super().__init__()
        self.stack_factor = stack_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        T_pad = (T + self.stack_factor - 1) // self.stack_factor * self.stack_factor
        x = torch.nn.functional.pad(x, (0, 0, 0, T_pad - T))
        B, T, C = x.shape
        return x.view(B, T // self.stack_factor, C * self.stack_factor)


class _HFUltravoxProjector(torch.nn.Module):
    """
    Minimal in-test reproduction of ``scripts/ultravox_model.py::UltravoxProjector``
    for v0.5 (``projector_ln_mid=True``, ``projector_act='swiglu'``, biasless linears).
    """

    def __init__(self, audio_hidden: int, stack_factor: int, hidden_dim: int, text_dim: int, norm_init: float):
        super().__init__()
        self._pad_and_stack = _HFStackAudioFrames(stack_factor)
        dim_in = audio_hidden * stack_factor
        self.ln_pre = _HFRMSNorm(dim_in, init=norm_init)
        self.linear_1 = torch.nn.Linear(dim_in, hidden_dim, bias=False)
        dim_mid = hidden_dim // 2  # SwiGLU halves
        self.ln_mid = _HFRMSNorm(dim_mid, init=norm_init)
        self.linear_2 = torch.nn.Linear(dim_mid, text_dim, bias=False)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        h = self._pad_and_stack(audio_features)
        h = self.ln_pre(h)
        h = self.linear_1(h)
        x, gate = h.chunk(2, dim=-1)
        h = torch.nn.functional.silu(gate) * x
        h = self.ln_mid(h)
        h = self.linear_2(h)
        return h


def _materialize_adapter_cuda(config: AudioEncoderConfig, audio_hidden: int, lm_hidden: int) -> AudioAdapter:
    """Construct an AudioAdapter on CUDA, materializing every ParameterMeta."""
    from fast_llm.engine.distributed.config import DistributedConfig
    from fast_llm.engine.distributed.distributed import Distributed

    dc = DistributedConfig()
    distributed = Distributed(dc)
    audio_hidden_dim = TensorDim("audio_hidden", audio_hidden)
    output_dim = TensorDim("lm_hidden", lm_hidden)
    adapter = AudioAdapter(config, audio_hidden_dim, output_dim, distributed_config=dc)
    adapter.setup(distributed)

    for param_name, param in list(adapter.named_parameters()):
        if param.device.type != "meta":
            continue
        param_data = param.new_empty(param.shape, device="cuda")
        if hasattr(param, "init_parameter"):
            param.init_parameter(param_data, distributed)
        else:
            torch.nn.init.normal_(param_data)
        module_path, leaf_name = (
            param_name.rsplit(".", 1) if "." in param_name else (None, param_name)
        )
        module = adapter
        if module_path:
            for part in module_path.split("."):
                module = getattr(module, part)
        module._parameters[leaf_name] = torch.nn.Parameter(param_data, requires_grad=param.requires_grad)
    return adapter


@requires_cuda
def test_ultravox_projector_matches_fastllm_adapter():
    """
    Configure a Fast-LLM ``AudioAdapter`` exactly like an Ultravox v0.5 projector,
    copy weights from a reference HF ``UltravoxProjector``, and verify both produce
    the same output for the same input (within fp32 tolerance).
    """
    # Test-only shapes (small but structurally identical to v0_5-llama-3_2-1b).
    audio_hidden = 32
    stack_factor = 4
    projector_hidden = 64  # ``config.hidden_size`` in HF terms (pre-SwiGLU)
    text_dim = 24
    norm_init = 0.4

    # Build HF reference projector.
    hf = _HFUltravoxProjector(audio_hidden, stack_factor, projector_hidden, text_dim, norm_init).cuda()
    hf.eval()

    # Build Fast-LLM adapter via the same import_config path the loader uses.
    audio_config = {
        "d_model": audio_hidden,
        "num_hidden_layers": 1,
        "encoder_layers": 1,
        "encoder_ffn_dim": audio_hidden * 4,
        "encoder_attention_heads": 4,
        "num_mel_bins": 80,
        "max_source_positions": 32,
        "activation_function": "gelu",
        "layer_norm_eps": 1e-5,
    }
    top_level = {
        "stack_factor": stack_factor,
        "hidden_size": projector_hidden,
        "projector_act": "swiglu",
        "projector_ln_mid": True,
        "norm_init": norm_init,
    }
    cfg_dict = UltravoxAudioEncoderConverter.import_config(audio_config, top_level)
    cfg = AudioEncoderConfig.from_dict(cfg_dict)
    adapter = _materialize_adapter_cuda(cfg, audio_hidden, text_dim).eval()

    # Copy HF weights into Fast-LLM, mirroring exactly what the checkpoint loader does
    # via UltravoxAudioAdapterConverter.
    with torch.no_grad():
        adapter.norm_1.weight.copy_(hf.ln_pre.weight)
        adapter.norm_2.weight.copy_(hf.ln_mid.weight)
        adapter.layer_1.weight.copy_(
            SwapGatedHalvesWeightConverter._swap(hf.linear_1.weight)
        )
        adapter.layer_2.weight.copy_(hf.linear_2.weight)

    # Forward an arbitrary input. Use a length that is already a multiple of the
    # stack factor so HF's right-pad branch matches Fast-LLM's trim-to-multiple branch.
    torch.manual_seed(0)
    N_clips, T = 2, stack_factor * 3
    audio_features = torch.randn(N_clips, T, audio_hidden, device="cuda", dtype=torch.float32)

    hf_out = hf(audio_features)  # (N_clips, T/k, text_dim)

    # Fast-LLM adapter expects flattened (N_clips * T, audio_hidden).
    fl_input = audio_features.reshape(N_clips * T, audio_hidden)
    fl_kwargs = {AudioKwargs.audio_num_clips: N_clips}
    fl_out_flat = adapter(fl_input, fl_kwargs)
    fl_out = fl_out_flat.view(N_clips, -1, text_dim)

    assert hf_out.shape == fl_out.shape, (hf_out.shape, fl_out.shape)
    # fp32 tolerance: RMSNorm + SwiGLU + 2 linears. Anything looser would mask real bugs.
    max_abs = (hf_out - fl_out).abs().max().item()
    rms = (hf_out - fl_out).pow(2).mean().sqrt().item()
    assert max_abs < 1e-4, f"projector parity broken: max_abs={max_abs}, rms={rms}"


@pytest.mark.parametrize(
    "projector_act,expected_gated,expected_activation",
    [
        ("swiglu", True, "silu"),
        ("gelu", False, "gelu"),
    ],
)
def test_ultravox_import_config_activation_dispatch(projector_act, expected_gated, expected_activation):
    """Activation / gating must follow ``projector_act`` from the HF config."""
    audio_config = {
        "d_model": 16,
        "num_hidden_layers": 1,
        "encoder_layers": 1,
        "encoder_ffn_dim": 32,
        "encoder_attention_heads": 4,
        "num_mel_bins": 80,
        "max_source_positions": 4,
        "activation_function": "gelu",
        "layer_norm_eps": 1e-5,
    }
    top_level = {
        "stack_factor": 2,
        "hidden_size": 8,
        "projector_act": projector_act,
        "projector_ln_mid": True,
        "norm_init": 0.4,
    }
    out = UltravoxAudioEncoderConverter.import_config(audio_config, top_level)
    assert out["adapter_gated"] is expected_gated
    assert out["adapter_activation_type"].value == expected_activation
    assert out["adapter_bias"] is False
    assert out["adapter_dropout"] == 0.0
    assert math.isclose(
        out["adapter_pre_normalization"]["weight"]["initialization"]["value"], 0.4
    )


def test_ultravox_import_config_rejects_v0_4_layout():
    """v0.4.x layout (``projector_ln_mid=False``) is not supported and must error out."""
    audio_config = {
        "d_model": 16,
        "num_hidden_layers": 1,
        "encoder_layers": 1,
        "encoder_ffn_dim": 32,
        "encoder_attention_heads": 4,
        "num_mel_bins": 80,
        "max_source_positions": 4,
        "activation_function": "gelu",
        "layer_norm_eps": 1e-5,
    }
    top_level = {
        "stack_factor": 2,
        "hidden_size": 8,
        "projector_act": "swiglu",
        "projector_ln_mid": False,  # v0.4.x layout
        "norm_init": 0.4,
    }
    with pytest.raises(NotImplementedError):
        UltravoxAudioEncoderConverter.import_config(audio_config, top_level)


def test_ultravox_adapter_converter_uses_swap_for_gated_linear_1():
    """``layer_1.weight`` mapping must use the swap converter when SwiGLU is configured."""
    audio_config = {
        "d_model": 16,
        "num_hidden_layers": 1,
        "encoder_layers": 1,
        "encoder_ffn_dim": 32,
        "encoder_attention_heads": 4,
        "num_mel_bins": 80,
        "max_source_positions": 4,
        "activation_function": "gelu",
        "layer_norm_eps": 1e-5,
    }
    top_level = {
        "stack_factor": 2,
        "hidden_size": 8,
        "projector_act": "swiglu",
        "projector_ln_mid": True,
        "norm_init": 0.4,
    }
    cfg = AudioEncoderConfig.from_dict(
        UltravoxAudioEncoderConverter.import_config(audio_config, top_level)
    )
    converters = UltravoxAudioAdapterConverter.get_converters(
        cfg,
        fast_llm_prefix="audio_encoder.adapter",
        hf_encoder_prefix="audio_tower",
        hf_projector_prefix="multi_modal_projector",
    )
    layer_1_converters = [c for c in converters if c.fast_llm_name[0].endswith("layer_1.weight")]
    assert len(layer_1_converters) == 1
    assert isinstance(layer_1_converters[0], SwapGatedHalvesWeightConverter)
