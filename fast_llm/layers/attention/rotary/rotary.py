import abc
import math
import typing

import torch

from fast_llm.config import Configurable
from fast_llm.engine.base_model.config import Preprocessor
from fast_llm.engine.config_utils.tensor_dim import TensorDim, scalar_dim
from fast_llm.functional.triton.rotary import triton_rotary_autograd_
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.attention.rotary.config import (
    DefaultRotaryConfig,
    Llama3RotaryConfig,
    NoRotaryConfig,
    RotaryConfig,
    YarnRotaryConfig,
)
from fast_llm.tensor import TensorMeta
from fast_llm.utils import div


def convert_rotary_complex_to_real(tensor: torch.Tensor, head_size: int, dim: int) -> torch.Tensor:
    return tensor.unflatten(dim, (-1, div(head_size, 2), 2)).movedim(dim + 1, dim + 2).flatten(dim, dim + 2)


def convert_rotary_real_to_complex(tensor: torch.Tensor, head_size: int, dim: int) -> torch.Tensor:
    return tensor.unflatten(dim, (-1, 2, div(head_size, 2))).movedim(dim + 1, dim + 2).flatten(dim, dim + 2)


def apply_rotary_embeddings(tensor: torch.Tensor, rope_frequencies: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to a tensor:
    * Convert it to a complex, full-precision tensor
    * Multiply by the frequencies
    * Convert back tho the input format.
    # TODO: Full precision only needed for bfloat16? (Doesn't support complex numbers)
    # TODO: This could use torch compile, but it doesn't support complex tensors at the moment.
    """
    complex_tensor = torch.view_as_complex(tensor.to(torch.float32).view(*tensor.shape[:-1], -1, 2))
    return torch.view_as_real(complex_tensor * rope_frequencies).view_as(tensor).type_as(tensor)


class Rotary[ConfigType: RotaryConfig](Configurable[ConfigType], torch.nn.Module, Preprocessor):
    def __init__(
        self,
        config: ConfigType,
        head_size_dim: TensorDim,
    ):
        super().__init__(config)
        self._head_size_dim = head_size_dim

    @abc.abstractmethod
    def forward(
        self, query: torch.Tensor, key: torch.Tensor, kwargs: dict[str, typing.Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class NoRotary[ConfigType: NoRotaryConfig](Rotary[ConfigType]):
    def forward(
        self, query: torch.Tensor, key: torch.Tensor, kwargs: dict[str, typing.Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return query, key

    def preprocess(self, batch, kwargs: dict[str, typing.Any]) -> None:
        pass

    def preprocess_meta(self, kwargs: dict[str, typing.Any]) -> None:
        pass


class DefaultRotary[ConfigType: DefaultRotaryConfig](Rotary[ConfigType]):
    _rotary_embedding_frequencies: torch.Tensor
    _tensor_cache_max_sequence_length: int = -1

    def preprocess(self, batch: torch.Tensor, kwargs: dict[str, typing.Any]) -> None:
        self._create_tensors(kwargs[AttentionKwargs.sequence_length], batch.device)
        sequence_k = kwargs[AttentionKwargs.sequence_k_dim].size
        kwargs[AttentionKwargs.rotary_freq_q] = self._rotary_embedding_frequencies[
            :, sequence_k - kwargs[AttentionKwargs.sequence_q_dim].size : sequence_k
        ]
        kwargs[AttentionKwargs.rotary_freq_k] = self._rotary_embedding_frequencies[:, :sequence_k]

    def preprocess_meta(self, kwargs: dict[str, typing.Any]) -> None:
        kwargs[AttentionKwargs.rotary_freq_q] = TensorMeta.from_dims(
            (
                scalar_dim,
                kwargs[AttentionKwargs.sequence_q_dim],
                scalar_dim,
                self._head_size_dim,
            ),
            tensor_name=AttentionKwargs.rotary_freq_q,
        )
        kwargs[AttentionKwargs.rotary_freq_k] = TensorMeta.from_dims(
            (
                scalar_dim,
                kwargs[AttentionKwargs.sequence_q_dim],
                scalar_dim,
                self._head_size_dim,
            ),
            tensor_name=AttentionKwargs.rotary_freq_k,
        )

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, kwargs: dict[str, typing.Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rotary_fn = triton_rotary_autograd_ if self._config.triton else apply_rotary_embeddings
        query = rotary_fn(query, kwargs[AttentionKwargs.rotary_freq_q])
        key = rotary_fn(key, kwargs[AttentionKwargs.rotary_freq_k])
        return query, key

    def _create_tensors(self, sequence_length: int, device: torch.device) -> None:
        if sequence_length <= self._tensor_cache_max_sequence_length:
            return
        self._tensor_cache_max_sequence_length = sequence_length

        self._rotary_embedding_frequencies = self._get_frequencies(
            sequence_length,
            self._head_size_dim.global_size,
            device=device,
        )

    def _get_frequencies(self, sequence_length: int, head_size: int, device: torch.device) -> torch.Tensor:
        # Calculate the complex frequencies (https://blog.eleuther.ai/rotary-embeddings/)
        # `exp(i * n * a) = cos(n * a) + i sin(n * a)`,
        # `a = theta ** - (2 * (channel // 2) / head_size)`,
        # where n is the position in the sequence.
        # We preform the calculation in high precision because it matters for rotary embeddings.
        positions = torch.arange(sequence_length, device=device, dtype=torch.float64)
        angles = torch.outer(positions, self._get_angle_scales(head_size, device))
        frequencies = torch.polar(torch.ones_like(angles), angles)[None, :, None, :].to(torch.complex64)
        if not self._config.complex_format:
            frequencies = convert_rotary_complex_to_real(
                torch.view_as_real(frequencies).flatten(-2), head_size, 3
            ).contiguous()
        return frequencies

    def _get_angle_scales(self, head_size: int, device: torch.device) -> torch.Tensor:
        return self._config.theta ** -torch.arange(0, 1, 2 / head_size, device=device, dtype=torch.float64)


class Llama3Rotary[ConfigType: Llama3RotaryConfig](DefaultRotary[ConfigType]):
    def _get_angle_scales(self, head_size: int, device: torch.device) -> torch.Tensor:
        scales = super()._get_angle_scales(head_size, device)
        low_frequency_wavelength = self._config.original_context_length / self._config.low_frequency_factor
        high_frequency_wavelength = self._config.original_context_length / self._config.high_frequency_factor
        new_scales = []
        for scale in scales:
            wavelength = 2 * math.pi / scale
            if wavelength < high_frequency_wavelength:
                new_scales.append(scale)
            elif wavelength > low_frequency_wavelength:
                new_scales.append(scale / self._config.scale_factor)
            else:
                smooth = (self._config.original_context_length / wavelength - self._config.low_frequency_factor) / (
                    self._config.high_frequency_factor - self._config.low_frequency_factor
                )
                new_scales.append((1 - smooth) * scale / self._config.scale_factor + smooth * scale)
        return torch.stack(new_scales)


class YarnRotary[ConfigType: YarnRotaryConfig](DefaultRotary[ConfigType]):
    """
    Yarn scaling:
    https://github.com/huggingface/transformers/blob/006d9249ec0270ff6c4d3840979d23fe94bdc763/src/transformers/modeling_rope_utils.py#L163
    [original paper](https://arxiv.org/abs/2309.00071)
    """

    def _get_frequencies(self, sequence_length: int, head_size: int, device: torch.device) -> torch.Tensor:
        return super()._get_frequencies(sequence_length, head_size, device) * self._config.attention_factor

    def _get_angle_scales(self, head_size: int, device: torch.device) -> torch.Tensor:
        scales = super()._get_angle_scales(head_size, device)
        # TODO: max_position_embeddings or original_context_length?
        # see https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L304
        low = max(math.floor(self._get_correction(self._config.beta_fast, head_size)), 0)
        high = min(math.ceil(self._get_correction(self._config.beta_slow, head_size)), head_size - 1)
        if low == high:
            high += 0.001  # Prevent singularity

        # Get n-dimensional rotational scaling corrected for extrapolation
        extrapolation_factor = torch.clamp(
            (torch.arange(head_size // 2, dtype=torch.float32, device=scales.device) - low) / (high - low), 0, 1
        )
        return scales / self._config.scale_factor * extrapolation_factor + scales * (1 - extrapolation_factor)

    def _linear_ramp_factor(self, min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    def _get_correction(self, beta: float, dim: int) -> float:
        return (
            dim
            * math.log(self._config.original_context_length / (beta * 2 * math.pi))
            / (2 * math.log(self._config.theta))
        )
