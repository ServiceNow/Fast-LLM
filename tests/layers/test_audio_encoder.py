"""
Tests for dynamic audio padding changes to AudioPreprocessor and AudioConv.

AudioPreprocessor tests run on CPU (no ParameterMeta materialization needed).
AudioConv forward tests require CUDA (ParameterMeta uses meta device → CUDA materialization).

These tests validate the behaviour implemented in the merge worktree:
  - AudioPreprocessor computes audio_token_lens from actual raw audio lengths
  - AudioPreprocessor mel shape follows aud_padding_duration (max_length mode) or
    batch-longest audio (longest mode), not a hardcoded 30s
  - AudioConv positional embedding is sliced to match actual sequence length,
    fixing the broadcast error when audio is shorter than max_source_positions
"""

import numpy as np
import pytest
import torch

from fast_llm.layers.audio_encoder.config import (
    AudioEncoderConfig,
    AudioEncoderKwargs,
    AudioEncoderType,
)
from fast_llm.layers.audio_encoder.preprocessing import AudioPreprocessor
from tests.utils.utils import requires_cuda


def _materialize_audio_conv_cuda(config: AudioEncoderConfig):
    """
    Construct an AudioConv with all ParameterMeta tensors materialized on CUDA.

    AudioConv stores its weights as ParameterMeta on the meta device.  Simply
    calling .cuda() on such a module fails with "Cannot copy out of meta tensor".
    This helper follows the same initialization path used by the Fast-LLM
    training engine (stage_base.py) but trimmed down for unit-test use:

      1. Create a shared DistributedConfig instance.
      2. Create Distributed from that config (initialises CUDA device/generators).
      3. Pass the same config instance to AudioConv so that Distributed.check_config
         (identity check) passes when conv.setup(distributed) is called.
      4. Iterate named parameters; for each ParameterMeta on the meta device,
         allocate a real CUDA tensor, call init_parameter to fill it, and swap
         the module's parameter entry.
    """
    from fast_llm.engine.distributed.config import DistributedConfig
    from fast_llm.engine.distributed.distributed import Distributed
    from fast_llm.layers.audio_encoder.encoder import AudioConv

    dc = DistributedConfig()
    distributed = Distributed(dc)
    conv = AudioConv(config, distributed_config=dc)
    conv.setup(distributed)

    for param_name, param in list(conv.named_parameters()):
        if param.device.type != "meta":
            continue
        param_data = param.new_empty(param.shape, device="cuda")
        if hasattr(param, "init_parameter"):
            param.init_parameter(param_data, distributed)
        else:
            torch.nn.init.normal_(param_data)
        module_path, leaf_name = param_name.rsplit(".", 1) if "." in param_name else (None, param_name)
        module = conv
        if module_path:
            for part in module_path.split("."):
                module = getattr(module, part)
        module._parameters[leaf_name] = torch.nn.Parameter(param_data, requires_grad=param.requires_grad)

    return conv

# ------------------------------------------------------------------
# Constants mirroring the Whisper / FastLLM audio pipeline
# ------------------------------------------------------------------
SR = 16_000       # default sampling rate
HOP = 160         # Whisper mel hop length
CONV_STRIDE = 2   # AudioConv conv2 stride
K = 5             # default aud_downsampling_k


def _token_count(raw_samples: int, k: int = K) -> int:
    """Expected LLM token count for a given number of raw audio samples."""
    return raw_samples // HOP // CONV_STRIDE // k


# ------------------------------------------------------------------
# Helper: build a minimal AudioPreprocessor on CPU
# ------------------------------------------------------------------

def _make_preprocessor(
    *,
    aud_padding_duration: int = 30,
    audio_padding: str = "max_length",
    k: int = K,
) -> AudioPreprocessor:
    """
    Construct an AudioPreprocessor configured for CPU-only testing.

    AudioPreprocessor.__init__ accepts aud_padding_duration and audio_padding
    as keyword arguments so that the preprocessor does not need to reach into
    BatchConfig at runtime.
    """
    config = AudioEncoderConfig(
        encoder_type=AudioEncoderType.whisper,
        aud_downsampling_k=k,
        aud_sampling_rate=SR,
    )
    return AudioPreprocessor(
        config,
        device=None,
        aud_padding_duration=aud_padding_duration,
        audio_padding=audio_padding,
    )


# ==================================================================
# AudioPreprocessor — audio_token_lens correctness
# ==================================================================


def test_audio_preprocessor_token_lens_reflect_actual_durations():
    """
    audio_token_lens stored in kwargs must equal each audio's actual raw length
    divided by (hop * conv_stride * k), regardless of padding mode.

    This is the core correctness requirement: the LLM should receive as many
    placeholder tokens as the encoder will actually produce for that audio,
    not the padded maximum.
    """
    preprocessor = _make_preprocessor(aud_padding_duration=30, audio_padding="max_length")
    audio_5s = np.zeros(5 * SR, dtype=np.float32)
    audio_10s = np.zeros(10 * SR, dtype=np.float32)
    # Two clips, flat list (AudioBatch.to_kwargs already flattens across documents)
    kwargs = {AudioEncoderKwargs.audio: [audio_5s, audio_10s]}
    preprocessor.preprocess(None, kwargs)

    assert AudioEncoderKwargs.audio_token_lens in kwargs, (
        "audio_token_lens must be added to kwargs by AudioPreprocessor.preprocess()"
    )
    lens = kwargs[AudioEncoderKwargs.audio_token_lens]
    assert lens[0].item() == _token_count(5 * SR), f"5s audio: got {lens[0].item()}"
    assert lens[1].item() == _token_count(10 * SR), f"10s audio: got {lens[1].item()}"


def test_audio_preprocessor_token_lens_longest_mode():
    """audio_token_lens reflects actual lengths even in 'longest' padding mode."""
    preprocessor = _make_preprocessor(aud_padding_duration=30, audio_padding="longest")
    audio_3s = np.zeros(3 * SR, dtype=np.float32)
    audio_15s = np.zeros(15 * SR, dtype=np.float32)
    kwargs = {AudioEncoderKwargs.audio: [audio_3s, audio_15s]}
    preprocessor.preprocess(None, kwargs)

    lens = kwargs[AudioEncoderKwargs.audio_token_lens]
    assert lens[0].item() == _token_count(3 * SR)
    assert lens[1].item() == _token_count(15 * SR)


def test_audio_preprocessor_token_lens_multiple_audios_per_sample():
    """audio_token_lens is flattened across batch; two audios in one sample are both recorded."""
    preprocessor = _make_preprocessor(aud_padding_duration=30, audio_padding="max_length")
    audio_5s = np.zeros(5 * SR, dtype=np.float32)
    audio_8s = np.zeros(8 * SR, dtype=np.float32)
    # Two clips from one sample, flat list
    kwargs = {AudioEncoderKwargs.audio: [audio_5s, audio_8s]}
    preprocessor.preprocess(None, kwargs)

    lens = kwargs[AudioEncoderKwargs.audio_token_lens]
    assert len(lens) == 2
    assert lens[0].item() == _token_count(5 * SR)
    assert lens[1].item() == _token_count(8 * SR)


def test_audio_preprocessor_no_audio_no_token_lens():
    """When no audio is in kwargs, audio_token_lens must not appear in kwargs."""
    preprocessor = _make_preprocessor()
    kwargs = {}
    preprocessor.preprocess(None, kwargs)
    assert AudioEncoderKwargs.audio_token_lens not in kwargs


# ==================================================================
# AudioPreprocessor — mel shape correctness
# ==================================================================


def test_audio_preprocessor_mel_shape_respects_config_padding_duration():
    """
    In max_length mode the mel time axis must equal
    aud_padding_duration * sr / hop — NOT the hardcoded 3000 (30s).

    This verifies the fix: max_length passed to WhisperFeatureExtractor is
    derived from config, not hardcoded to 30 * sr.
    """
    aud_padding_duration = 10  # 10s, NOT 30s
    preprocessor = _make_preprocessor(aud_padding_duration=aud_padding_duration, audio_padding="max_length")
    audio = np.zeros(5 * SR, dtype=np.float32)  # 5s < 10s padding
    kwargs = {AudioEncoderKwargs.audio: [audio]}
    preprocessor.preprocess(None, kwargs)

    mel = kwargs[AudioEncoderKwargs.audio_mel]
    expected_frames = aud_padding_duration * SR // HOP  # 1000 for 10s
    assert mel.shape[-1] == expected_frames, (
        f"mel frames should be {expected_frames} for {aud_padding_duration}s padding, got {mel.shape[-1]}. "
        "Likely cause: max_length is still hardcoded to 30s in preprocess()."
    )


def test_audio_preprocessor_mel_shape_longest_pads_to_batch_max():
    """
    In 'longest' mode the mel time axis must equal
    max_audio_len_in_batch / hop — NOT aud_padding_duration / hop.

    Here the batch-longest audio is 10s, so expected frames = 10 * sr / hop = 1000.
    """
    preprocessor = _make_preprocessor(aud_padding_duration=30, audio_padding="longest")
    audio_5s = np.zeros(5 * SR, dtype=np.float32)
    audio_10s = np.zeros(10 * SR, dtype=np.float32)
    kwargs = {AudioEncoderKwargs.audio: [audio_5s, audio_10s]}
    preprocessor.preprocess(None, kwargs)

    mel = kwargs[AudioEncoderKwargs.audio_mel]
    expected_frames = 10 * SR // HOP  # 1000
    assert mel.shape[-1] == expected_frames, (
        f"longest-mode mel frames should be {expected_frames}, got {mel.shape[-1]}"
    )


def test_audio_preprocessor_no_audio_produces_single_dummy_mel():
    """
    When no audio is present the mel tensor must still have batch dim == 1
    so downstream layers receive a constant-shape tensor.
    """
    preprocessor = _make_preprocessor(aud_padding_duration=30, audio_padding="max_length")
    kwargs = {}
    preprocessor.preprocess(None, kwargs)

    mel = kwargs[AudioEncoderKwargs.audio_mel]
    assert mel.shape[0] == 1, (
        f"Expected dummy mel with batch=1, got shape {tuple(mel.shape)}. "
        "Likely cause: max_pad=1 logic removed or broken."
    )


def test_audio_preprocessor_mel_batch_dim_equals_num_audio_clips():
    """mel batch dimension must equal the total number of audio clips across all samples."""
    preprocessor = _make_preprocessor(aud_padding_duration=30, audio_padding="max_length")
    audio_a = np.zeros(3 * SR, dtype=np.float32)
    audio_b = np.zeros(7 * SR, dtype=np.float32)
    audio_c = np.zeros(5 * SR, dtype=np.float32)
    # Three clips flat (two from sample 0, one from sample 1)
    kwargs = {AudioEncoderKwargs.audio: [audio_a, audio_b, audio_c]}
    preprocessor.preprocess(None, kwargs)

    mel = kwargs[AudioEncoderKwargs.audio_mel]
    assert mel.shape[0] == 3, f"Expected 3 audio clips in mel, got {mel.shape[0]}"


# ==================================================================
# AudioConv — constructor (CPU, no materialization needed)
# ==================================================================


def test_audio_conv_constructor():
    """AudioConv constructs without error on CPU."""
    from fast_llm.layers.audio_encoder.encoder import AudioConv

    config = AudioEncoderConfig(
        encoder_type=AudioEncoderType.whisper,
        aud_downsampling_k=K,
        aud_sampling_rate=SR,
    )
    conv = AudioConv(config)
    assert conv is not None


# ==================================================================
# AudioConv — positional embedding slice fix (CUDA required)
# ==================================================================


@requires_cuda
def test_audio_conv_forward_full_length():
    """AudioConv forward pass works at max_source_positions (1500 frames, 30s audio)."""
    from fast_llm.layers.audio_encoder.config import AudioKwargs

    config = AudioEncoderConfig(
        encoder_type=AudioEncoderType.whisper,
        aud_downsampling_k=K,
        aud_sampling_rate=SR,
    )
    conv = _materialize_audio_conv_cuda(config)

    # 30s → 3000 mel frames → 1500 after conv2 (stride 2): exactly max_source_positions
    mel_frames = 30 * SR // HOP  # 3000
    mel = torch.randn(1, config.num_mel_bins, mel_frames, device="cuda", dtype=torch.float32)
    kwargs = {AudioKwargs.audio_mel: mel}
    dummy_input = torch.zeros(1, device="cuda")

    output = conv(dummy_input, kwargs)
    # Output is (N_clips * T, hidden_size); for N_clips=1, shape[0] is T
    assert output.shape[0] == mel_frames // CONV_STRIDE  # 1500


@requires_cuda
def test_audio_conv_forward_short_audio_no_error():
    """
    AudioConv forward pass with audio shorter than max_source_positions must NOT raise.

    The positional embedding is sliced to [:T] to match actual sequence length,
    fixing the broadcast error when audio is shorter than max_source_positions.
    """
    from fast_llm.layers.audio_encoder.config import AudioKwargs

    config = AudioEncoderConfig(
        encoder_type=AudioEncoderType.whisper,
        aud_downsampling_k=K,
        aud_sampling_rate=SR,
    )
    conv = _materialize_audio_conv_cuda(config)

    # 5s → 500 mel frames → 250 after conv2 (stride 2): well below 1500
    mel_frames = 5 * SR // HOP  # 500
    mel = torch.randn(1, config.num_mel_bins, mel_frames, device="cuda", dtype=torch.float32)
    kwargs = {AudioKwargs.audio_mel: mel}
    dummy_input = torch.zeros(1, device="cuda")

    output = conv(dummy_input, kwargs)
    # Output is (N_clips * T, hidden_size); for N_clips=1, shape[0] is T
    assert output.shape[0] == mel_frames // CONV_STRIDE  # 250, not 1500


@requires_cuda
@pytest.mark.parametrize("duration_s", [5, 10, 15, 20, 25, 30])
def test_audio_conv_output_length_matches_input_duration(duration_s: int):
    """AudioConv output time dimension scales linearly with audio duration."""
    from fast_llm.layers.audio_encoder.config import AudioKwargs

    config = AudioEncoderConfig(
        encoder_type=AudioEncoderType.whisper,
        aud_downsampling_k=K,
        aud_sampling_rate=SR,
    )
    conv = _materialize_audio_conv_cuda(config)

    mel_frames = duration_s * SR // HOP
    mel = torch.randn(1, config.num_mel_bins, mel_frames, device="cuda", dtype=torch.float32)
    kwargs = {AudioKwargs.audio_mel: mel}
    dummy_input = torch.zeros(1, device="cuda")

    output = conv(dummy_input, kwargs)
    expected_time = mel_frames // CONV_STRIDE
    # Output is (N_clips * T, hidden_size); for N_clips=1, shape[0] is T
    assert output.shape[0] == expected_time, (
        f"{duration_s}s audio: expected conv output T={expected_time}, got {output.shape[0]}"
    )


# ==================================================================
# AudioAdapter — trimming behaviour (CUDA required)
# ==================================================================


def _materialize_audio_adapter_cuda(
    config: AudioEncoderConfig,
    audio_hidden_size: int,
    lm_hidden_size: int,
):
    """
    Construct an AudioAdapter with all ParameterMeta weights materialized on CUDA.

    Mirrors _materialize_audio_conv_cuda but targets AudioAdapter.
    audio_hidden_size must match config.hidden_size so that norm_1 gets the
    correct dimension.
    """
    from fast_llm.engine.config_utils.tensor_dim import TensorDim
    from fast_llm.engine.distributed.config import DistributedConfig
    from fast_llm.engine.distributed.distributed import Distributed
    from fast_llm.layers.audio_encoder.adapter import AudioAdapter

    dc = DistributedConfig()
    distributed = Distributed(dc)

    audio_hidden_dim = TensorDim("audio_hidden", audio_hidden_size)
    output_dim = TensorDim("lm_hidden", lm_hidden_size)

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
        module._parameters[leaf_name] = torch.nn.Parameter(
            param_data, requires_grad=param.requires_grad
        )

    return adapter


# Small dimensions keep the tests fast while covering all code paths.
_HIDDEN = 64   # audio encoder hidden size (must be divisible by layer-norm internals)
_LM_HIDDEN = 32  # LM embedding size (adapter output)
_ADAPTER_SIZE = 128  # adapter intermediate size


def _make_adapter_config(k: int) -> AudioEncoderConfig:
    return AudioEncoderConfig(
        encoder_type=AudioEncoderType.whisper,
        aud_downsampling_k=k,
        hidden_size=_HIDDEN,
        adapter_size=_ADAPTER_SIZE,
        aud_sampling_rate=SR,
    )


@requires_cuda
def test_audio_adapter_output_shape_no_trim_needed():
    """
    When seq_len is already a multiple of k, output shape is
    (N_clips * T/k, lm_hidden) via the flat reshape path.
    """
    from fast_llm.layers.audio_encoder.config import AudioKwargs

    k = 4
    N_clips, T = 2, 8  # T=8 is a multiple of k=4
    config = _make_adapter_config(k)
    adapter = _materialize_audio_adapter_cuda(config, _HIDDEN, _LM_HIDDEN)

    input_ = torch.randn(N_clips * T, _HIDDEN, device="cuda")
    kwargs = {AudioKwargs.audio_num_clips: N_clips}
    output = adapter(input_, kwargs)

    expected_tokens = N_clips * (T // k)  # 2 * 2 = 4
    assert output.shape == (expected_tokens, _LM_HIDDEN), (
        f"Expected ({expected_tokens}, {_LM_HIDDEN}), got {tuple(output.shape)}"
    )


@requires_cuda
def test_audio_adapter_trims_seq_len_to_multiple_of_k():
    """
    When seq_len % k != 0 the adapter discards the trailing frames so that
    the remaining length is a multiple of k.

    T=9 with k=4 → trim to 8 → new_seq_len=2 → output (N_clips*2, lm_hidden).
    """
    from fast_llm.layers.audio_encoder.config import AudioKwargs

    k = 4
    N_clips, T = 1, 9  # 9 is NOT a multiple of 4; must trim to 8
    config = _make_adapter_config(k)
    adapter = _materialize_audio_adapter_cuda(config, _HIDDEN, _LM_HIDDEN)

    input_ = torch.randn(N_clips * T, _HIDDEN, device="cuda")
    kwargs = {AudioKwargs.audio_num_clips: N_clips}
    output = adapter(input_, kwargs)

    # trimmed T=8, new_seq_len=2, total tokens = 1*2 = 2
    assert output.shape == (2, _LM_HIDDEN), (
        f"Expected (2, {_LM_HIDDEN}) after trimming T=9→8 with k=4, got {tuple(output.shape)}"
    )


@requires_cuda
@pytest.mark.parametrize(
    "T,k,expected_new_seq_len",
    [
        (7, 3, 2),   # 7 → trim to 6, 6//3 = 2
        (11, 4, 2),  # 11 → trim to 8, 8//4 = 2
        (5, 2, 2),   # 5 → trim to 4, 4//2 = 2
        (6, 3, 2),   # 6 is already a multiple of 3, 6//3 = 2 (no actual trim)
    ],
)
def test_audio_adapter_trim_parametrized(T: int, k: int, expected_new_seq_len: int):
    """Adapter output length equals floor(T/k) for various (T, k) combinations."""
    from fast_llm.layers.audio_encoder.config import AudioKwargs

    N_clips = 1
    config = _make_adapter_config(k)
    adapter = _materialize_audio_adapter_cuda(config, _HIDDEN, _LM_HIDDEN)

    input_ = torch.randn(N_clips * T, _HIDDEN, device="cuda")
    kwargs = {AudioKwargs.audio_num_clips: N_clips}
    output = adapter(input_, kwargs)

    assert output.shape == (N_clips * expected_new_seq_len, _LM_HIDDEN), (
        f"T={T}, k={k}: expected ({N_clips * expected_new_seq_len}, {_LM_HIDDEN}), "
        f"got {tuple(output.shape)}"
    )


@requires_cuda
def test_audio_adapter_per_clip_trim_when_audio_token_lens_shorter():
    """
    When audio_token_lens total < N_clips * new_seq_len, the adapter trims
    each clip to its actual token count and concatenates the results.

    N_clips=3, T=10 (k=2 → new_seq_len=5).
    audio_token_lens=[3, 4, 2], total=9 < 3*5=15 → per-clip trim path.
    Expected output shape: (9, lm_hidden).
    """
    from fast_llm.layers.audio_encoder.config import AudioKwargs

    k = 2
    N_clips, T = 3, 10  # new_seq_len = 5
    config = _make_adapter_config(k)
    adapter = _materialize_audio_adapter_cuda(config, _HIDDEN, _LM_HIDDEN)

    audio_token_lens = torch.tensor([3, 4, 2], dtype=torch.int32, device="cuda")
    input_ = torch.randn(N_clips * T, _HIDDEN, device="cuda")
    kwargs = {
        AudioKwargs.audio_num_clips: N_clips,
        AudioKwargs.audio_token_lens: audio_token_lens,
    }
    output = adapter(input_, kwargs)

    total_tokens = audio_token_lens.sum().item()  # 9
    assert output.shape == (total_tokens, _LM_HIDDEN), (
        f"Expected ({total_tokens}, {_LM_HIDDEN}), got {tuple(output.shape)}"
    )


@requires_cuda
def test_audio_adapter_no_per_clip_trim_when_all_clips_full():
    """
    When audio_token_lens total == N_clips * new_seq_len (all clips exactly fill
    their padded slot), the adapter uses the flat reshape path.

    N_clips=2, T=6, k=2 → new_seq_len=3.  audio_token_lens=[3, 3], total=6=2*3.
    Expected output shape: (6, lm_hidden).
    """
    from fast_llm.layers.audio_encoder.config import AudioKwargs

    k = 2
    N_clips, T = 2, 6  # new_seq_len = 3
    config = _make_adapter_config(k)
    adapter = _materialize_audio_adapter_cuda(config, _HIDDEN, _LM_HIDDEN)

    audio_token_lens = torch.tensor([3, 3], dtype=torch.int32, device="cuda")
    input_ = torch.randn(N_clips * T, _HIDDEN, device="cuda")
    kwargs = {
        AudioKwargs.audio_num_clips: N_clips,
        AudioKwargs.audio_token_lens: audio_token_lens,
    }
    output = adapter(input_, kwargs)

    # sum(lens) == N_clips * new_seq_len → no per-clip trim
    assert output.shape == (N_clips * (T // k), _LM_HIDDEN), (
        f"Expected ({N_clips * (T // k)}, {_LM_HIDDEN}), got {tuple(output.shape)}"
    )


@requires_cuda
def test_audio_adapter_no_audio_token_lens_uses_flat_reshape():
    """
    When audio_token_lens is absent from kwargs the adapter always uses the
    flat reshape path regardless of clip lengths.
    """
    from fast_llm.layers.audio_encoder.config import AudioKwargs

    k = 2
    N_clips, T = 2, 4  # new_seq_len = 2
    config = _make_adapter_config(k)
    adapter = _materialize_audio_adapter_cuda(config, _HIDDEN, _LM_HIDDEN)

    input_ = torch.randn(N_clips * T, _HIDDEN, device="cuda")
    kwargs = {AudioKwargs.audio_num_clips: N_clips}  # no audio_token_lens key
    output = adapter(input_, kwargs)

    assert output.shape == (N_clips * (T // k), _LM_HIDDEN), (
        f"Expected ({N_clips * (T // k)}, {_LM_HIDDEN}), got {tuple(output.shape)}"
    )
