import logging
import math
import typing

import torch
import torch.nn.attention.flex_attention

from fast_llm.engine.base_model.config import Preprocessor
from fast_llm.engine.config_utils.tensor_space import DefaultDimNames, TensorDim, TensorSpace
from fast_llm.functional.rotary import convert_rotary_complex_to_real
from fast_llm.layers.transformer.config import (
    ATTENTION_IMPLEMENTATION_SPECS,
    AttentionMode,
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


class RotaryEmbeddingPreprocessor(Preprocessor):
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

    def _create_tensors(self, sequence_length: int) -> None:
        if sequence_length <= self._tensor_cache_max_sequence_length:
            return
        self._tensor_cache_max_sequence_length = sequence_length

        self._rotary_embedding_frequencies = get_rotary_frequencies(
            self._config,
            sequence_length,
            self._kv_channels_dim.global_size,
            device=self._tensor_space.distributed.device,
        )

    def preprocess(self, batch, kwargs: dict[str, typing.Any]) -> None:
        self._create_tensors(kwargs[TransformerKwargs.sequence_length])
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


class MaskMod(typing.Protocol):
    """Callable type for masking functions.
    Should take (batch, head, q_idx, kv_idx) and return a boolean mask tensor.
    """

    def __call__(
        self,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor: ...


def _build_seq_idx(cu_seqlens: torch.Tensor, total_length: torch.types.Number) -> torch.Tensor:
    """Maps each token position to its sequence index based on provided offsets.
    Used to relate flat token indices to their sequence in a packed batch.
    """
    range_tensor = torch.arange(total_length, device=cu_seqlens.device, dtype=torch.int32)
    seq_idx = torch.searchsorted(cu_seqlens, range_tensor, right=True) - 1
    return seq_idx


def _varlen_mask_mod_adapter(
    orig_mask_mod: MaskMod,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    allow_inter_sequence_attention: bool = False,
) -> MaskMod:
    """Lift a fixed-length predicate to packed (variable-length) sequences.

    Translates per-sequence indices to relative indices inside each document and
    blocks attention across different documents if cumulative sequence lengths
    are provided. If not, the original mask is returned.
    """
    if cu_seqlens_q is None:
        return orig_mask_mod
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    q_seq_idx = _build_seq_idx(cu_seqlens_q, cu_seqlens_q[-1])
    if cu_seqlens_q is cu_seqlens_k:
        kv_seq_idx = q_seq_idx
    else:
        kv_seq_idx = _build_seq_idx(cu_seqlens_k, cu_seqlens_k[-1])

    def _varlen_mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        b_varlen = q_seq_idx[q_idx]
        q_varlen = q_idx - cu_seqlens_q[q_seq_idx[q_idx]]
        kv_varlen = kv_idx - cu_seqlens_k[kv_seq_idx[kv_idx]]

        mask = orig_mask_mod(b_varlen, h, q_varlen, kv_varlen)

        # don't allow inter-sequence attention if requested
        if not allow_inter_sequence_attention:
            is_same_sequence = q_seq_idx[q_idx] == kv_seq_idx[kv_idx]
            return mask & is_same_sequence

        return mask

    return _varlen_mask_mod


def _window_mask_mod_adapter(
    orig_mask_mod: MaskMod,
    window_size: tuple[int, int] = (-1, -1),
) -> MaskMod:
    """Lift a fixed-length predicate to a windowed predicate.

    Translates per-sequence indices to relative indices inside each document and
    blocks attention across different documents. If no window is specified, the original
    mask is returned.
    """
    win_left, win_right = window_size
    if win_left == -1 and win_right == -1:
        return orig_mask_mod
    if win_left == -1 and win_right != -1:

        def _window_mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            return orig_mask_mod(b, h, q_idx, kv_idx) & (kv_idx <= q_idx + win_right)

    elif win_left != -1 and win_right == -1:

        def _window_mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            return orig_mask_mod(b, h, q_idx, kv_idx) & (kv_idx >= q_idx - win_left)

    else:

        def _window_mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            return orig_mask_mod(b, h, q_idx, kv_idx) & ((kv_idx >= q_idx - win_left) & (kv_idx <= q_idx + win_right))

    return _window_mask_mod


def _causal_mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
    """Standard left-to-right causal mask."""
    return kv_idx <= q_idx


def _bidirectional_mask_mod(
    b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
) -> torch.Tensor:
    """Fully bidirectional mask (all tokens can attend to all others)."""
    return True  # type: ignore


def _varlen_to_abs_idx(cu_seq_lens: torch.Tensor | None, b: torch.Tensor, rel_idx: torch.Tensor):
    """Converts relative sequence indices to absolute batch indices."""
    return rel_idx if cu_seq_lens is None else cu_seq_lens[b] + rel_idx


def _causal_or_block_bidirectional_mask_mod(
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    block_ids_q: torch.Tensor | None,
    block_ids_k: torch.Tensor | None,
) -> MaskMod:
    """Creates a mask that allows full (bidirectional) attention within blocks,
    but defaults to causal attention otherwise. Block id -1 disables bidirectional attention.
    """
    if block_ids_q is None:
        return _causal_mask_mod

    if block_ids_k is None:
        block_ids_k = block_ids_q

    def _mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        abs_q_idx = _varlen_to_abs_idx(cu_seqlens_q, b, q_idx)
        abs_kv_idx = _varlen_to_abs_idx(cu_seqlens_k, b, kv_idx)

        bid_q = block_ids_q[abs_q_idx]
        bid_k = block_ids_k[abs_kv_idx]

        # Allow bidirectional attention only when both tokens share the *same*
        # non-negative block id.  Any -1 disables the bidirectional attention and
        # defaults to causal attention.
        bidirectional_ok = (bid_q != -1) & (bid_k != -1) & (bid_q == bid_k)

        causal_ok = _causal_mask_mod(b, h, q_idx, kv_idx)

        return bidirectional_ok | causal_ok

    return _mask_mod


def _get_mask_mod(
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    block_ids_q: torch.Tensor | None = None,
    block_ids_k: torch.Tensor | None = None,
    window_size: tuple[int, int] = (-1, -1),
    attention_mode: AttentionMode = AttentionMode.causal,
) -> MaskMod:
    """
    Constructs a composite mask function based on attention mode,
    sequence lengths, block ids, and windowing.
    """
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q

    match attention_mode:
        case AttentionMode.causal:
            _mask_mod = _causal_mask_mod
        case AttentionMode.bidirectional:
            _mask_mod = _bidirectional_mask_mod
        case AttentionMode.causal_or_block_bidirectional:
            _mask_mod = _causal_or_block_bidirectional_mask_mod(cu_seqlens_q, cu_seqlens_k, block_ids_q, block_ids_k)
        case _:
            raise ValueError(f"Unsupported attention mode: {attention_mode}")

    _mask_mod = _varlen_mask_mod_adapter(_mask_mod, cu_seqlens_q, cu_seqlens_k)
    _mask_mod = _window_mask_mod_adapter(_mask_mod, window_size)

    return _mask_mod


class FastAttentionPreprocessor(Preprocessor):
    def __init__(
        self,
        config: TransformerConfig,
        tensor_space: TensorSpace,
    ) -> None:
        self._config = config
        self._tensor_space = tensor_space
        self._distributed_config = self._tensor_space.distributed_config

        try:
            import flash_attn.flash_attn_interface

            self._flash_available = True
        except ImportError:
            self._flash_available = False

    def _preprocess_varlen(self, batch, kwargs: dict[str, typing.Any]) -> None:
        """
        Prepares cu_seqlens_q and cu_seqlens_k for variable length attention.
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py#L1375
        cu_seqlens_q and cu_seqlens_k are cumulative sequence lengths for the query and key/value tensors, respectively.
        Assumes a flattened batch of documents. In absence of sequence_data_parallelism, cu_seqlens_q = cu_seqlens_k.
        If sequence_data_parallelism > 1, query tensors contain tokens only from current micro-sequence, whereas key/value tensors additionally
        also contain previous tokens from the first document in micro-sequence.
        We use individual sequence lengths of each document to (optionally) find the micro-sequences in the batch and compute the cumulative lengths.
        """
        if (sequence_lengths := kwargs.get(TransformerKwargs.sequence_lengths)) is None:
            return
        sequence_lengths = kwargs[TransformerKwargs.sequence_lengths]
        sequence_k = kwargs[TransformerKwargs.sequence_k_dim].size
        sequence_q = kwargs[TransformerKwargs.sequence_q_dim].size
        if sequence_q < kwargs[TransformerKwargs.sequence_length]:
            cumsums = [torch.cumsum(x, dim=0) for x in sequence_lengths]
            # The first and last documents in a microsequence need to be handled separately. Include all tokens from other documents
            # in the microsequence. We need to consider all keys computed so far from the first sample. We also store the offsets
            # of the first documents so that we can index into their kv pairs
            start_seq_idx = [
                torch.argmax((cu_seqlens >= sequence_k - sequence_q).to(torch.uint8), dim=0) for cu_seqlens in cumsums
            ]
            end_seq_idx = [torch.argmax((cu_seqlens >= sequence_k).to(torch.uint8), dim=0) for cu_seqlens in cumsums]
            seqlens_q = []
            seqlens_k = []
            for idx, sample_seqlens in enumerate(sequence_lengths):
                start_idx = start_seq_idx[idx]
                end_idx = end_seq_idx[idx]
                seqlens_q.extend([0] * start_idx)
                n_attention_tokens = sample_seqlens[end_idx] - (cumsums[idx][end_idx] - sequence_k)
                if start_idx == end_idx:
                    seqlens_q.append(sequence_q)
                else:
                    start_q_tokens = cumsums[idx][start_idx] - (sequence_k - sequence_q)
                    seqlens_q.extend(
                        [
                            start_q_tokens,
                            *(sample_seqlens[idx] for idx in range(start_idx + 1, end_idx)),
                            n_attention_tokens,
                        ]
                    )
                seqlens_k.extend(sample_seqlens[: end_idx + 1])
                seqlens_k[-1] = n_attention_tokens
            seqlens_q = torch.tensor(seqlens_q, dtype=torch.int32)
            seqlens_k = torch.tensor(seqlens_k, dtype=torch.int32)
        else:
            seqlens_q = torch.cat(sequence_lengths)
            seqlens_k = torch.cat(sequence_lengths)
        kwargs[TransformerKwargs.cu_seqlens_q] = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=self._tensor_space.distributed.device),
                torch.cumsum(seqlens_q, dim=0, dtype=torch.int32).to(self._tensor_space.distributed.device),
            )
        )
        kwargs[TransformerKwargs.cu_seqlens_k] = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=self._tensor_space.distributed.device),
                torch.cumsum(seqlens_k, dim=0, dtype=torch.int32).to(self._tensor_space.distributed.device),
            )
        )
        kwargs[TransformerKwargs.max_seqlen_q] = seqlens_q.max()
        kwargs[TransformerKwargs.max_seqlen_k] = seqlens_k.max()

    def _preprocess_block_mask(self, batch, kwargs: dict[str, typing.Any]) -> None:
        cu_seqlens_q = kwargs.get(TransformerKwargs.cu_seqlens_q)
        cu_seqlens_k = kwargs.get(TransformerKwargs.cu_seqlens_k)
        block_ids_q = kwargs.get(TransformerKwargs.block_ids_q)
        block_ids_k = kwargs.get(TransformerKwargs.block_ids_k)
        window_size = (-1, -1) if self._config.window_size is None else (self._config.window_size - 1, 0)

        mask_mod = _get_mask_mod(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_ids_q=block_ids_q,
            block_ids_k=block_ids_k,
            window_size=window_size,
            attention_mode=kwargs.get(TransformerKwargs.attention_mode, AttentionMode.causal),
        )

        if cu_seqlens_q is not None:
            batch_size = cu_seqlens_q.numel() - 1
            if cu_seqlens_k is not None:
                assert cu_seqlens_k.numel() - 1 == batch_size, "cu_seqlens_k must have same shape as cu_seqlens_q"
            seqlen_q = cu_seqlens_q[-1]
            if cu_seqlens_k is not None:
                seqlen_k = cu_seqlens_k[-1]
            else:
                seqlen_k = seqlen_q
        else:
            batch_size = self._tensor_space.get_tensor_dim(TransformerDimNames.batch).size
            seqlen_q = kwargs[TransformerKwargs.sequence_q_dim].size
            seqlen_k = kwargs[TransformerKwargs.sequence_k_dim].size

        head_groups = self._tensor_space.get_tensor_dim(TransformerDimNames.head_groups).size
        heads_per_group = self._tensor_space.get_tensor_dim(TransformerDimNames.group_heads).size
        query_heads = head_groups * heads_per_group

        block_mask = torch.nn.attention.flex_attention.create_block_mask(
            mask_mod=mask_mod,
            B=batch_size,
            H=query_heads,
            Q_LEN=seqlen_q,
            KV_LEN=seqlen_k,
            device=self._tensor_space.distributed.device,
            BLOCK_SIZE=torch.nn.attention.flex_attention._DEFAULT_SPARSE_BLOCK_SIZE,
            _compile=False,  # TODO: switch on compilation once we have a working version
        )

        kwargs[TransformerKwargs.block_mask] = block_mask

    def preprocess(self, batch, kwargs: dict[str, typing.Any]) -> None:
        attention_implementation = kwargs.get(
            TransformerKwargs.attention_implementation,
            self._config.select_attention_implementation(
                require_variable_length=TransformerKwargs.sequence_lengths in kwargs,
                training_dtype=self._tensor_space.distributed_config.training_dtype,
                flash_available=self._flash_available,
            ),
        )
        attention_implementation_spec = ATTENTION_IMPLEMENTATION_SPECS[attention_implementation]
        if attention_implementation_spec.variable_length:
            self._preprocess_varlen(batch, kwargs)
        if attention_implementation_spec.flex:
            self._preprocess_block_mask(batch, kwargs)


class BackupAttentionPreprocessor(Preprocessor):
    _scalar_dim: TensorDim
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
        self._scalar_dim = self._tensor_space.get_tensor_dim(DefaultDimNames.scalar)

    def _create_tensors(self, sequence_length: int) -> None:
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

    def preprocess(self, batch, kwargs: dict[str, typing.Any]) -> None:
        self._create_tensors(kwargs[TransformerKwargs.sequence_length])
        sequence_k = kwargs[TransformerKwargs.sequence_k_dim].size
        sequence_q = kwargs[TransformerKwargs.sequence_q_dim].size
        kwargs[TransformerKwargs.attention_mask] = self._mask[
            None, None, sequence_k - sequence_q : sequence_k, None, :sequence_k
        ]
        if (sequence_lengths := kwargs.get(TransformerKwargs.sequence_lengths, None)) is not None:
            seq_ids = torch.stack(
                [
                    torch.cat([torch.full((x,), i) for i, x in enumerate(sample_lens)])
                    for sample_lens in sequence_lengths
                ]
            )
            document_mask = (seq_ids[:, None, :] == seq_ids[:, :, None]).to(self._tensor_space.distributed.device)
            kwargs[TransformerKwargs.attention_mask] = (
                kwargs[TransformerKwargs.attention_mask]
                & document_mask[:, None, sequence_k - sequence_q : sequence_k, None, :sequence_k]
            )
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
