"""
Ultravox HF checkpoint converter (``fixie-ai/ultravox-v0_5-*``).

Ultravox is structurally an Ayra-shaped audio-language model:
  - ``audio_tower``           — Whisper-large-v3-turbo encoder (LayerNorm internals)
  - ``multi_modal_projector`` — 2-layer MLP projector with two RMSNorms and SwiGLU
  - ``language_model``        — Llama-family LLM

This subclasses :class:`AyraHuggingfaceCheckpointHandler` and only overrides:
  - HF state-dict prefixes (``audio_tower`` / ``multi_modal_projector`` / ``language_model``)
  - the projector / adapter weight mapping (different field names, biasless linears,
    RMSNorm in place of LayerNorm)
  - top-level projector knobs read from the Ultravox config
    (``stack_factor``, ``projector_act``, ``projector_ln_mid``, ``norm_init``).

The Ultravox checkpoint must be repackaged with an inline ``text_config`` block
(full Llama config, not just a ``text_model_id`` pointer). This mirrors the Ayra
layout and avoids needing HF-hub access at Fast-LLM startup.

Weight name mapping:
  audio_tower.conv1.weight                       → audio_encoder.conv.conv1_weight
  audio_tower.conv2.weight                       → audio_encoder.conv.conv2_weight
  audio_tower.conv{1,2}.bias                     → audio_encoder.conv.conv{1,2}_bias
  audio_tower.embed_positions.weight             → audio_encoder.conv.positional_embeddings
  audio_tower.layers.{i}.*                       → audio_encoder.encoder.{i}.* (reused Whisper block converter)
  audio_tower.layer_norm.{weight,bias}           → audio_encoder.final_norm.norm.{weight,bias}
  multi_modal_projector.ln_pre.weight            → audio_encoder.adapter.norm_1.weight  (RMSNorm; no bias)
  multi_modal_projector.linear_1.weight          → audio_encoder.adapter.layer_1.weight (no bias)
  multi_modal_projector.ln_mid.weight            → audio_encoder.adapter.norm_2.weight  (RMSNorm; no bias)
  multi_modal_projector.linear_2.weight          → audio_encoder.adapter.layer_2.weight (no bias)
  language_model.*                               → (dispatched to the Llama text handler)
"""

import typing

import torch

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import WeightConverter
from fast_llm.layers.audio_encoder.config import AudioEncoderConfig
from fast_llm.models.gpt.conversion.llama import get_weight_and_bias_converters
from fast_llm.models.multimodal.conversion.ayra import AyraHuggingfaceCheckpointHandler
from fast_llm.models.multimodal.conversion.config import UltravoxCheckpointFormat
from fast_llm.models.multimodal.conversion.whisper import WhisperAudioEncoderConverter
from fast_llm.tensor import SafeTensorSlice


class SwapGatedHalvesWeightConverter(WeightConverter):
    """
    Swap the two halves (along dim 0) of a gated linear's weight/bias.

    HF Ultravox SwiGLU: ``x, gate = h.chunk(2, dim=-1); silu(gate) * x`` —
    the first half of ``linear_1``'s output is the multiplier, the second
    half is the silu-activated path.

    Fast-LLM's ``torch_mlp_activation(gated=True, silu)`` does
    ``silu(x1) * x2`` — first half is activated, second half is multiplier.

    To make HF weights produce identical outputs in Fast-LLM, swap the halves.
    """

    @staticmethod
    def _swap(t: torch.Tensor | SafeTensorSlice) -> torch.Tensor:
        full = t[:]
        a, b = full.chunk(2, dim=0)
        return torch.cat([b, a], dim=0).contiguous()

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        return tuple(self._swap(w) for w in weight)

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        return tuple(self._swap(w) for w in weight)


class UltravoxAudioAdapterConverter:
    """
    Adapter / projector weight converter for Ultravox v0.5 checkpoints.

    Differs from :class:`WhisperAudioAdapterConverter` in three ways:
      1. Different HF field names (``ln_pre`` / ``linear_1`` / ``ln_mid`` / ``linear_2``).
      2. Linears are biasless.
      3. The two norms are RMSNorm (weight-only); no ``.bias`` entries.

    The encoder's terminal LayerNorm (``audio_tower.layer_norm``) is handled
    separately by :class:`WhisperFinalNormConverter` (it maps into
    ``AudioEncoder.final_norm`` rather than the adapter).
    """

    @classmethod
    def get_converters(
        cls,
        config: AudioEncoderConfig,
        fast_llm_prefix: str,
        hf_encoder_prefix: str,
        hf_projector_prefix: str,
    ) -> list[WeightConverter]:
        converters = [
            # norm_1 ← Ultravox projector ln_pre (RMSNorm, weight only).
            WeightConverter(f"{fast_llm_prefix}.norm_1.weight", f"{hf_projector_prefix}.ln_pre.weight"),
            # norm_2 ← Ultravox projector ln_mid (RMSNorm, weight only).
            WeightConverter(f"{fast_llm_prefix}.norm_2.weight", f"{hf_projector_prefix}.ln_mid.weight"),
        ]
        # layer_1: biasless. For SwiGLU we additionally swap the two halves
        # (HF chunks as (x, gate), Fast-LLM expects (gate, x)).
        layer_1_converter_class = SwapGatedHalvesWeightConverter if config.adapter_gated else WeightConverter
        converters += get_weight_and_bias_converters(
            f"{fast_llm_prefix}.layer_1",
            f"{hf_projector_prefix}.linear_1",
            config.adapter_bias,
            layer_1_converter_class,
        )
        # layer_2: biasless, straight passthrough.
        converters += get_weight_and_bias_converters(
            f"{fast_llm_prefix}.layer_2", f"{hf_projector_prefix}.linear_2", config.adapter_bias
        )
        return converters


class UltravoxAudioEncoderConverter(WhisperAudioEncoderConverter):
    """
    Encoder converter for Ultravox: inherits Whisper's conv + transformer + final-norm
    mappings unchanged, overrides only the adapter mapping and the config import to
    pick up the Ultravox projector knobs from the top-level config.
    """

    adapter_converter_class: typing.ClassVar = UltravoxAudioAdapterConverter

    @classmethod
    def import_config(cls, audio_config: dict, top_level_config: dict) -> dict:
        from fast_llm.functional.config import ActivationType

        base = super().import_config(audio_config)

        # Reject v0.4.x layout (ln_post after linear_2) — we map to ln_mid only.
        if not top_level_config.get("projector_ln_mid", True):
            raise NotImplementedError(
                "Ultravox v0.4.x layout (projector_ln_mid=False, ln_post after linear_2) "
                "is not supported. Use v0.5+ which puts the norm between linear_1 and linear_2."
            )

        projector_act = top_level_config.get("projector_act", "swiglu")
        if projector_act == "swiglu":
            # HF SwiGLU is silu(gate) * x with chunk(2, dim=-1); Fast-LLM
            # torch_mlp_activation matches this when activation=silu and gated=True.
            activation_type = ActivationType.silu
            gated = True
        else:
            activation_type = ActivationType.from_hf_name(projector_act)
            gated = False

        norm_init = top_level_config.get("norm_init", 1.0)
        rms_eps = top_level_config.get("rms_norm_eps", 1e-6)

        # Two separate dicts: ``Config.from_dict`` consumes its input, so reusing one
        # dict for two fields would leave the second one half-empty.
        def _rms_cfg() -> dict:
            return {
                "type": "rms_norm",
                "epsilon": rms_eps,
                "weight": {"initialization": {"type": "fill", "value": norm_init}},
            }

        # Projector intermediate size. For SwiGLU, linear_1 emits 2*adapter_size
        # so the activation halves back to adapter_size; the HF projector follows
        # the same convention (dim_mid = hidden_dim // 2 if swiglu else hidden_dim),
        # but stores ``hidden_size`` as the pre-activation dim.
        adapter_size = top_level_config["hidden_size"]
        if gated:
            # HF UltravoxProjector: linear_1 out_features = hidden_dim (pre-chunk),
            # then SwiGLU halves to hidden_dim/2 = adapter_size.
            adapter_size = adapter_size // 2

        base.update(
            {
                "adapter_size": adapter_size,
                "adapter_activation_type": activation_type,
                "adapter_gated": gated,
                "adapter_bias": False,
                "adapter_dropout": 0.0,
                "aud_downsampling_k": top_level_config["stack_factor"],
                "adapter_pre_normalization": _rms_cfg(),
                "adapter_mid_normalization": _rms_cfg(),
            }
        )
        return base


class UltravoxHuggingfaceCheckpointHandler(AyraHuggingfaceCheckpointHandler):
    """
    Loads ``fixie-ai/ultravox-v0_5-*`` HF checkpoints into Fast-LLM.

    Everything Ayra does applies; only the format identifier, the HF architecture
    string, the encoder converter (for the projector parity knobs), and the three
    HF state-dict prefixes change.
    """

    format: typing.ClassVar[type[CheckpointFormat]] = UltravoxCheckpointFormat
    architecture: typing.ClassVar[str] = "UltravoxModel"
    audio_encoder_converter_class: typing.ClassVar = UltravoxAudioEncoderConverter
    hf_audio_encoder_prefix: typing.ClassVar[str] = "audio_tower"
    hf_audio_projector_prefix: typing.ClassVar[str] = "multi_modal_projector"
    hf_llm_prefix: typing.ClassVar[str] = "language_model"

    @classmethod
    def _load_config(cls, directory) -> dict:
        # ``model_type: "ultravox"`` is not a HF-registered architecture and the
        # repackaged checkpoint intentionally drops the original ``auto_map``
        # (Fast-LLM doesn't run the HF model code). Bypass AutoConfig and read
        # the inline config dict directly — everything Fast-LLM needs is structural.
        import json
        import pathlib

        config = json.loads((pathlib.Path(directory) / "config.json").read_text())
        cls._last_loaded_hf_config = config
        return config
