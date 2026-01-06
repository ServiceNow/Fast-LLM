import typing

import torch

from fast_llm.layers.attention.config import MixerKwargs
from fast_llm.utils import Assert


def preprocess_for_varlen(
    kwargs: dict[str, typing.Any],
    device: torch.device,
    return_cu_seqlens: bool = False,
    return_max_seqlen: bool = False,
    return_seq_idx: bool = False,
    return_position_ids: bool = False,
) -> None:
    """
    Prepares cu_seqlens_q and cu_seqlens_k for flash_attn_varlen_func:
    https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py#L1375
    cu_seqlens_q and cu_seqlens_k are cumulative sequence lengths for the query and key/value tensors, respectively.
    Assumes a flattened batch of documents. In absence of sequence_data_parallelism, cu_seqlens_q = cu_seqlens_k.
    If sequence_data_parallelism > 1, query tensors contain tokens only from current micro-sequence, whereas key/value tensors additionally
    also contain previous tokens from the first document in micro-sequence.
    We use individual sequence lengths of each document to (optionally) find the micro-sequences in the batch and compute the cumulative lengths.
    """

    # TODO: ====== Fix (need to know how much first sequence was cropped) ======
    Assert.eq(kwargs[MixerKwargs.sequence_k_dim].global_size, kwargs[MixerKwargs.sequence_q_dim].global_size)

    sequence_lengths = [
        sequence_length
        for sequence_lengths in kwargs[MixerKwargs.sequence_lengths]
        for sequence_length in sequence_lengths
    ]
    if return_cu_seqlens:
        cu_seqlens_q = torch.tensor([0] + sequence_lengths, dtype=torch.int32, device=device).cumsum(
            0, dtype=torch.int32
        )
        kwargs[MixerKwargs.cu_seqlens_q] = cu_seqlens_q
        kwargs[MixerKwargs.cu_seqlens_k] = cu_seqlens_q
    if return_max_seqlen:
        max_seqlen_q = torch.full((1,), max(sequence_lengths), dtype=torch.int32, device=device)
        kwargs[MixerKwargs.max_seqlen_q] = max_seqlen_q
        kwargs[MixerKwargs.max_seqlen_k] = max_seqlen_q
    if return_seq_idx:
        kwargs[MixerKwargs.seq_idx] = torch.cat(
            [
                torch.full((sequence_length,), i, dtype=torch.int32, device=device)
                for i, sequence_length in enumerate(sequence_lengths)
            ]
        )
    if return_position_ids:
        kwargs[MixerKwargs.position_ids] = torch.cat(
            [
                torch.arange(sequence_length, dtype=torch.int32, device=device)
                for i, sequence_length in enumerate(sequence_lengths)
            ]
        )
