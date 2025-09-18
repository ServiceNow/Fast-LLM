import logging
import typing

import torch

from fast_llm.engine.base_model.config import Preprocessor
from fast_llm.engine.config_utils.tensor_dim import TensorDim, scalar_dim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.attention.config import AttentionConfig, AttentionKwargs
from fast_llm.tensor import TensorMeta

logger = logging.getLogger(__name__)


class BackupAttentionPreprocessor(Preprocessor):
    _head_size_dim: TensorDim
    _rotary_embedding_frequencies: torch.Tensor
    _mask: torch.Tensor
    _mask_value: torch.Tensor
    _tensor_cache_max_sequence_length: int = -1

    def __init__(self, config: AttentionConfig, distributed_config: DistributedConfig):
        self._config = config
        self._distributed_config = distributed_config
        assert not self._config.do_use_flash_attention(self._distributed_config)

    def _create_tensors(self, sequence_length: int, device: torch.device) -> None:
        if sequence_length <= self._tensor_cache_max_sequence_length:
            return
        self._tensor_cache_max_sequence_length = sequence_length

        self._mask = torch.ones(
            (sequence_length, sequence_length),
            dtype=torch.bool,
            device=device,
        ).tril_()

        if self._config.window_size is not None:
            self._mask.triu_(-self._config.window_size + 1)
        self._mask_value = torch.full(
            [],
            torch.finfo(self._distributed_config.compute_dtype.torch).min,
            dtype=self._distributed_config.compute_dtype.torch,
            device=device,
        )

    def preprocess(self, batch: torch.Tensor, kwargs: dict[str, typing.Any]) -> None:
        self._create_tensors(kwargs[AttentionKwargs.sequence_length], batch.device)
        sequence_k = kwargs[AttentionKwargs.sequence_k_dim].size
        sequence_q = kwargs[AttentionKwargs.sequence_q_dim].size
        kwargs[AttentionKwargs.attention_mask] = self._mask[
            None, None, sequence_k - sequence_q : sequence_k, None, :sequence_k
        ]
        if (sequence_lengths := kwargs.get(AttentionKwargs.sequence_lengths, None)) is not None:
            seq_ids = torch.stack(
                [
                    torch.cat([torch.full((x,), i) for i, x in enumerate(sample_lens)])
                    for sample_lens in sequence_lengths
                ]
            )
            document_mask = (seq_ids[:, None, :] == seq_ids[:, :, None]).to(batch.device)
            kwargs[AttentionKwargs.attention_mask] = (
                kwargs[AttentionKwargs.attention_mask]
                & document_mask[:, None, sequence_k - sequence_q : sequence_k, None, :sequence_k]
            )
        kwargs[AttentionKwargs.attention_mask_value] = self._mask_value

    def preprocess_meta(self, kwargs: dict[str, typing.Any]) -> None:
        kwargs[AttentionKwargs.attention_mask] = TensorMeta.from_dims(
            (
                scalar_dim,
                scalar_dim,
                kwargs[AttentionKwargs.sequence_q_dim],
                scalar_dim,
                kwargs[AttentionKwargs.sequence_k_dim],
            ),
            tensor_name=AttentionKwargs.attention_mask,
            dtype=torch.bool,
        )
        kwargs[AttentionKwargs.attention_mask_value] = TensorMeta.from_dims(
            (scalar_dim,),
            tensor_name=AttentionKwargs.attention_mask_value,
            dtype=self._distributed_config.compute_dtype.torch,
        )


class FlashAttnVarlenPreprocessor(Preprocessor):
    def __init__(self, config: AttentionConfig, distributed_config: DistributedConfig):
        assert config.do_use_flash_attention(distributed_config)

    def preprocess(self, batch: torch.Tensor, kwargs: dict[str, typing.Any]) -> None:
        """
        Prepares cu_seqlens_q and cu_seqlens_k for flash_attn_varlen_func:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py#L1375
        cu_seqlens_q and cu_seqlens_k are cumulative sequence lengths for the query and key/value tensors, respectively.
        Assumes a flattened batch of documents. In absence of sequence_data_parallelism, cu_seqlens_q = cu_seqlens_k.
        If sequence_data_parallelism > 1, query tensors contain tokens only from current micro-sequence, whereas key/value tensors additionally
        also contain previous tokens from the first document in micro-sequence.
        We use individual sequence lengths of each document to (optionally) find the micro-sequences in the batch and compute the cumulative lengths.
        """
        if AttentionKwargs.sequence_lengths not in kwargs:
            return
        sequence_lengths = kwargs[AttentionKwargs.sequence_lengths]
        sequence_k = kwargs[AttentionKwargs.sequence_k_dim].size
        sequence_q = kwargs[AttentionKwargs.sequence_q_dim].size
        if sequence_q < kwargs[AttentionKwargs.sequence_length]:
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
        kwargs[AttentionKwargs.cu_seqlens_q] = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=batch.device),
                torch.cumsum(seqlens_q, dim=0, dtype=torch.int32).to(batch.device),
            )
        )
        kwargs[AttentionKwargs.cu_seqlens_k] = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=batch.device),
                torch.cumsum(seqlens_k, dim=0, dtype=torch.int32).to(batch.device),
            )
        )
        kwargs[AttentionKwargs.max_seqlen_q] = seqlens_q.max()
        kwargs[AttentionKwargs.max_seqlen_k] = seqlens_k.max()
