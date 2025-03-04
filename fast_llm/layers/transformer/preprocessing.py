import logging
import math
import typing

import torch

from fast_llm.engine.config_utils.tensor_space import DefaultDimNames, TensorDim, TensorSpace
from fast_llm.functional.rotary import convert_rotary_complex_to_real
from fast_llm.layers.transformer.config import (
    RotaryConfig,
    RotaryEmbeddingType,
    TransformerConfig,
    TransformerDimNames,
    TransformerKwargs,
)
from fast_llm.tensor import TensorMeta

logger = logging.getLogger(__name__)


def apply_llama3_scaling(config: RotaryConfig, frequencies: torch.Tensor) -> torch.Tensor:
    """
    Llama3 scaling: https://github.com/meta-llama/llama-models/blob/baf7b01b6e62bc7126c7b558d2b67d4533142680/models/llama3/reference_impl/model.py#L45-L67
    """
    low_frequency_wavelength = config.original_context_length / config.low_frequency_factor
    high_frequency_wavelength = config.original_context_length / config.high_frequency_factor
    new_frequencies = []
    for frequency in frequencies:
        wavelength = 2 * math.pi / frequency
        if wavelength < high_frequency_wavelength:
            new_frequencies.append(frequency)
        elif wavelength > low_frequency_wavelength:
            new_frequencies.append(frequency / config.scale_factor)
        else:
            assert low_frequency_wavelength != high_frequency_wavelength
            smooth = (config.original_context_length / wavelength - config.low_frequency_factor) / (
                config.high_frequency_factor - config.low_frequency_factor
            )
            new_frequencies.append((1 - smooth) * frequency / config.scale_factor + smooth * frequency)
    return torch.tensor(new_frequencies, dtype=frequencies.dtype, device=frequencies.device), 1.0


def apply_yarn_scaling(config: RotaryConfig, frequencies: torch.Tensor, kv_channels, sequence_length) -> torch.Tensor:
    """
    Yarn scaling:
    https://github.com/huggingface/transformers/blob/006d9249ec0270ff6c4d3840979d23fe94bdc763/src/transformers/modeling_rope_utils.py#L163
    [original paper](https://arxiv.org/abs/2309.00071)
    """
    base = config.theta
    partial_rotary_factor = 1.0
    dim = int(kv_channels * partial_rotary_factor)
    max_position_embeddings = sequence_length
    factor = config.scale_factor

    attention_factor = config.attention_factor
    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
        """Find dimension range bounds based on rotations"""
        low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func


    # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
    # to expand the possible context length. In other words, interpolation = apply scaling factor.
    # pos_freqs = base ** (torch.arange(0, dim, 2).float().to(frequencies.device) / dim)
    # inv_freq_extrapolation = 1.0 / pos_freqs
    # inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    inv_freq_extrapolation = frequencies
    inv_freq_interpolation = frequencies / factor

    # TODO: max_position_embeddings or original_context_length?
    # see https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L304
    low, high = find_correction_range(config.beta_fast, config.beta_slow, dim, base, config.original_context_length)

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).float().to(frequencies.device)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )

    return inv_freq, attention_factor



def get_rotary_frequencies(
    config: RotaryConfig,
    sequence_length,
    kv_channels,
    *,
    device="cuda",
) -> torch.Tensor:
    # Calculate the complex frequencies (https://blog.eleuther.ai/rotary-embeddings/)
    # `exp(i * n * a) = cos(n * a) + i sin(n * a)`,
    # `a = theta ** - (2 * (channel // 2) / kv_channels)`,
    # where n is the position in the sequence.
    # We preform the calculation in high precision because it matters for rotary embeddings.
    positions = torch.arange(sequence_length, device=device, dtype=torch.float64)
    frequencies = config.theta ** -torch.arange(0, 1, 2 / kv_channels, device=device, dtype=torch.float64)
    # Apply scaling
    if config.type == RotaryEmbeddingType.llama3:
        frequencies, attention_scaling = apply_llama3_scaling(config, frequencies)
    elif config.type == RotaryEmbeddingType.yarn:
        frequencies, attention_scaling = apply_yarn_scaling(config, frequencies, kv_channels, sequence_length)
    else:
        attention_scaling = 1.0
    angles = torch.outer(positions, frequencies)
    frequencies = torch.polar(torch.ones_like(angles), angles)[None, :, None, :].to(torch.complex64)
    if not config.complex_format:
        frequencies = convert_rotary_complex_to_real(
            torch.view_as_real(frequencies).flatten(-2), kv_channels, 3
        ).contiguous()
    # Advanced Rope types like yarn apply a post-processing scaling factor, equivalent to scaling attention.
    frequencies = frequencies * attention_scaling
    return frequencies


class RotaryEmbeddingPreprocessor:
    _scalar_dim: TensorDim
    _kv_channels_dim: TensorDim
    _rotary_embedding_frequencies: torch.Tensor
    _mask: torch.Tensor
    _mask_value: torch.Tensor
    _tensor_cache_max_sequence_length: int = -1

    def __init__(
        self,
        config: RotaryConfig,
        tensor_space: TensorSpace,
    ):
        self._config = config
        assert self._config.enabled
        self._tensor_space = tensor_space
        self._distributed_config = self._tensor_space.distributed_config
        self._scalar_dim = self._tensor_space.get_tensor_dim(DefaultDimNames.scalar)
        self._kv_channels_dim = self._tensor_space.get_tensor_dim(TransformerDimNames.kv_channels)

    def create_tensors(self, sequence_length: int) -> None:
        if sequence_length <= self._tensor_cache_max_sequence_length:
            return
        self._tensor_cache_max_sequence_length = sequence_length

        self._rotary_embedding_frequencies = get_rotary_frequencies(
            self._config,
            sequence_length,
            self._kv_channels_dim.global_size,
            device=self._tensor_space.distributed.device,
        )

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        sequence_k = kwargs[TransformerKwargs.sequence_k_dim].size
        kwargs[TransformerKwargs.rotary_freq_q] = self._rotary_embedding_frequencies[
            :, sequence_k - kwargs[TransformerKwargs.sequence_q_dim].size : sequence_k
        ]
        kwargs[TransformerKwargs.rotary_freq_k] = self._rotary_embedding_frequencies[:, :sequence_k]

    def preprocess_meta(self, kwargs: dict[str, typing.Any]) -> None:
        kwargs[TransformerKwargs.rotary_freq_q] = TensorMeta.from_dims(
            (
                self._scalar_dim,
                kwargs[TransformerKwargs.sequence_q_dim],
                self._scalar_dim,
                self._kv_channels_dim,
            ),
            tensor_name=TransformerKwargs.rotary_freq_q,
        )
        kwargs[TransformerKwargs.rotary_freq_k] = TensorMeta.from_dims(
            (
                self._scalar_dim,
                kwargs[TransformerKwargs.sequence_q_dim],
                self._scalar_dim,
                self._kv_channels_dim,
            ),
            tensor_name=TransformerKwargs.rotary_freq_k,
        )


class BackupAttentionPreprocessor:
    _scalar_dim: TensorDim
    _kv_channels_dim: TensorDim
    _rotary_embedding_frequencies: torch.Tensor
    _mask: torch.Tensor
    _mask_value: torch.Tensor
    _tensor_cache_max_sequence_length: int = -1

    def __init__(
        self,
        config: TransformerConfig,
        tensor_space: TensorSpace,
    ):
        self._config = config
        self._tensor_space = tensor_space
        self._distributed_config = self._tensor_space.distributed_config
        assert not self._config.do_use_flash_attention(self._distributed_config)
        self._scalar_dim = self._tensor_space.get_tensor_dim(DefaultDimNames.scalar)

    def create_tensors(self, sequence_length: int) -> None:
        if sequence_length <= self._tensor_cache_max_sequence_length:
            return
        self._tensor_cache_max_sequence_length = sequence_length

        self._mask = torch.ones(
            (sequence_length, sequence_length),
            dtype=torch.bool,
            device=self._tensor_space.distributed.device,
        ).tril_()
        if self._config.window_size is not None:
            self._mask.triu_(-self._config.window_size + 1)
        self._mask_value = torch.full(
            [],
            torch.finfo(self._distributed_config.training_dtype.torch).min,
            dtype=self._distributed_config.training_dtype.torch,
            device=self._tensor_space.distributed.device,
        )

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        sequence_k = kwargs[TransformerKwargs.sequence_k_dim].size
        kwargs[TransformerKwargs.attention_mask] = self._mask[
            None, None, sequence_k - kwargs[TransformerKwargs.sequence_q_dim].size : sequence_k, None, :sequence_k
        ]
        kwargs[TransformerKwargs.attention_mask_value] = self._mask_value

    def preprocess_meta(self, kwargs: dict[str, typing.Any]) -> None:
        kwargs[TransformerKwargs.attention_mask] = TensorMeta.from_dims(
            (
                self._scalar_dim,
                self._scalar_dim,
                kwargs[TransformerKwargs.sequence_q_dim],
                self._scalar_dim,
                kwargs[TransformerKwargs.sequence_k_dim],
            ),
            tensor_name=TransformerKwargs.attention_mask,
            dtype=torch.bool,
        )
        kwargs[TransformerKwargs.attention_mask_value] = TensorMeta.from_dims(
            (self._scalar_dim,),
            tensor_name=TransformerKwargs.attention_mask_value,
            dtype=self._tensor_space.distributed_config.training_dtype.torch,
        )
