"""
Tests for audio-aware sampling: get_num_audio_tokens() with the new audio_padding parameter.

Validates the core design of the dynamic audio padding plan:
- "max_length" mode (backward-compatible default): all token counts equal the padded max.
- "longest"   mode: each audio's token count is computed from its actual duration.
- Long-audio filtering works in both modes.
- Special start/end tokens correctly add +1 each to the count.
"""

import numpy as np
import pytest

from fast_llm.layers.audio_encoder.preprocessing import get_num_audio_tokens

# ------------------------------------------------------------------
# Constants matching the Whisper/FastLLM audio pipeline:
#   raw_samples → /160 (hop) → /2 (conv2 stride) → /k (adapter)
# ------------------------------------------------------------------
HOP_LENGTH = 160  # Whisper mel hop
CONV_STRIDE = 2  # AudioConv conv2 stride
K = 5  # default aud_downsampling_k
SR = 16_000  # default sampling rate
MAX_DUR = 30  # seconds
MAX_RAW = MAX_DUR * SR  # 480_000 raw samples


def _token_count(raw_samples: int, k: int = K) -> int:
    """Expected LLM token count for a given number of raw audio samples."""
    return raw_samples // HOP_LENGTH // CONV_STRIDE // k


# ------------------------------------------------------------------
# max_length mode
# ------------------------------------------------------------------


def test_max_length_all_equal_padded_max():
    """In max_length mode every audio gets the same token count regardless of actual length."""
    sizes = np.array([5 * SR, 10 * SR, 15 * SR])
    counts, to_filter = get_num_audio_tokens(
        sizes,
        aud_padding_duration=MAX_DUR,
        aud_sampling_rate=SR,
        aud_downsampling_k=K,
        audio_start_token=None,
        audio_end_token=None,
        audio_padding="max_length",
    )
    expected = _token_count(MAX_RAW)  # 300 for default 30s
    np.testing.assert_array_equal(counts, [expected, expected, expected])
    assert not to_filter


# ------------------------------------------------------------------
# longest mode
# ------------------------------------------------------------------


def test_longest_uses_actual_duration():
    """In longest mode each audio's count is computed from its actual sample count."""
    sizes = np.array([5 * SR, 10 * SR, 15 * SR])
    counts, to_filter = get_num_audio_tokens(
        sizes,
        aud_padding_duration=MAX_DUR,
        aud_sampling_rate=SR,
        aud_downsampling_k=K,
        audio_start_token=None,
        audio_end_token=None,
        audio_padding="longest",
    )
    np.testing.assert_array_equal(
        counts,
        [_token_count(5 * SR), _token_count(10 * SR), _token_count(15 * SR)],
    )
    assert not to_filter


def test_longest_max_length_produce_same_counts_when_all_audio_is_max():
    """When every audio is exactly aud_padding_duration long, both modes agree."""
    sizes = np.array([MAX_RAW, MAX_RAW])
    counts_max, _ = get_num_audio_tokens(sizes, MAX_DUR, SR, K, None, None, audio_padding="max_length")
    counts_long, _ = get_num_audio_tokens(sizes, MAX_DUR, SR, K, None, None, audio_padding="longest")
    np.testing.assert_array_equal(counts_max, counts_long)


# ------------------------------------------------------------------
# Filtering
# ------------------------------------------------------------------


@pytest.mark.parametrize("audio_padding", ["max_length", "longest"])
def test_filters_audio_longer_than_max_duration(audio_padding: str):
    """Audio exceeding aud_padding_duration triggers to_filter in both modes."""
    sizes = np.array([5 * SR, MAX_RAW + 1])  # second clip is 1 sample over 30 s
    _, to_filter = get_num_audio_tokens(sizes, MAX_DUR, SR, K, None, None, audio_padding=audio_padding)
    assert to_filter, f"Expected to_filter=True for audio_padding={audio_padding!r}"


@pytest.mark.parametrize("audio_padding", ["max_length", "longest"])
def test_no_filter_when_audio_within_max(audio_padding: str):
    """Audio at exactly the max duration should NOT trigger to_filter."""
    sizes = np.array([MAX_RAW])
    _, to_filter = get_num_audio_tokens(sizes, MAX_DUR, SR, K, None, None, audio_padding=audio_padding)
    assert not to_filter


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


def test_empty_sizes_returns_empty_no_filter():
    """Empty input produces an empty token array and to_filter=False."""
    sizes = np.array([], dtype=np.int64)
    counts, to_filter = get_num_audio_tokens(sizes, MAX_DUR, SR, K, None, None, audio_padding="longest")
    assert len(counts) == 0
    assert not to_filter


# ------------------------------------------------------------------
# Special tokens
# ------------------------------------------------------------------


def test_special_tokens_add_one_each():
    """audio_start_token and audio_end_token each contribute +1 token to the count."""
    sizes = np.array([10 * SR])
    base, _ = get_num_audio_tokens(sizes, MAX_DUR, SR, K, None, None, "longest")
    with_start, _ = get_num_audio_tokens(sizes, MAX_DUR, SR, K, audio_start_token=20, audio_end_token=None, audio_padding="longest")
    with_both, _ = get_num_audio_tokens(sizes, MAX_DUR, SR, K, audio_start_token=20, audio_end_token=22, audio_padding="longest")

    assert with_start[0] == base[0] + 1
    assert with_both[0] == base[0] + 2


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------


def test_default_audio_padding_is_max_length():
    """Calling without audio_padding should give the same result as audio_padding='max_length'."""
    sizes = np.array([5 * SR, 15 * SR])
    counts_default, _ = get_num_audio_tokens(sizes, MAX_DUR, SR, K, None, None)
    counts_explicit, _ = get_num_audio_tokens(sizes, MAX_DUR, SR, K, None, None, audio_padding="max_length")
    np.testing.assert_array_equal(counts_default, counts_explicit)
