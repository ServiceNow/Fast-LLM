"""
A collection of external kernels and kernel wrappers used within Fast llm.
TODO: Is is still useful to keep them in one place? (Some like layer norm are added elsewhere.)
Todo: Move all core methods elsewhere (functional?).
"""

import torch

from fast_llm.core.distributed import set_generator

try:
    from amp_C import multi_tensor_adam as _multi_tensor_adam  # noqa
    from amp_C import multi_tensor_l2norm as _multi_tensor_l2norm  # noqa
    from amp_C import multi_tensor_scale as _multi_tensor_scale  # noqa
    from apex.multi_tensor_apply import multi_tensor_applier as _multi_tensor_applier  # noqa

    _apex_available = True
except ImportError:
    _apex_available = False

try:
    from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func  # noqa
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func

    _flash_available = True
except ImportError:
    _flash_available = False


def l2_norm(tensors: list[torch.Tensor], noop_flag: torch.Tensor) -> torch.Tensor:
    assert _apex_available
    norm, _ = _multi_tensor_applier(
        _multi_tensor_l2norm,
        noop_flag,
        [tensors],
        False,  # no per-parameter norm
    )
    return norm


def scale_(tensors: list[torch.Tensor], noop_flag: torch.Tensor, scale: torch.Tensor | float) -> None:
    assert _apex_available
    _multi_tensor_applier(
        _multi_tensor_scale,
        noop_flag,
        [tensors, tensors],
        scale,
    )


# TODO: Same as torch._fused_adam_?
def fused_adam(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    exp_avgs: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    noop_flag: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    wd: float,
    eps: float,
    step: int,
) -> None:
    _multi_tensor_applier(
        _multi_tensor_adam,
        noop_flag,
        [grads, params, exp_avgs, exp_avg_sqs],
        lr,
        beta1,
        beta2,
        eps,
        step,
        1,  # adamw
        1,  # bias correction
        wd,
    )


def flash_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
    *,
    window_size: int | None,
    causal: bool = False,
    generator: torch.Generator | None,
    softmax_scale: float | None = None,
    position_ids: torch.Tensor | None = None,
    prevent_cross_document_attention: bool = False,
) -> torch.Tensor:
    assert _flash_available
    with set_generator(generator):
        if prevent_cross_document_attention:
            out_dims = query.size()
            query, key, value, indices_q, cu_seq_lens, max_seq_lens = prepare_fa2_from_position_ids(
                query, key, value, position_ids
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            return _flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout_p,
                window_size=(-1, -1) if window_size is None else (window_size - 1, 0),
                causal=causal,
                softmax_scale=softmax_scale,
            ).view(*out_dims)
        else:
            return _flash_attn_func(
                query,
                key,
                value,
                window_size=(-1, -1) if window_size is None else (window_size - 1, 0),
                dropout_p=dropout_p,
                causal=causal,
                softmax_scale=softmax_scale,
            )


def prepare_fa2_from_position_ids(query, key, value, position_ids):
    """
    This function returns necessary arguments to call `flash_attn_varlen_func`.
    All three query, key, value states will be flattened.
    Cummulative lengths of each examples in the batch will be extracted from position_ids.
    NOTE: ideally cummulative lengths should be prepared at the data collator stage
    Arguments:
        query (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        position_ids (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
    Return:
        query (`torch.Tensor):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    query = query.view(-1, query.size(-2), query.size(-1))
    key = key.view(-1, key.size(-2), key.size(-1))
    value = value.view(-1, value.size(-2), value.size(-1))
    position_ids = position_ids.flatten()
    indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)

    cu_seq_lens = torch.cat(
        (
            indices_q[position_ids == 0],
            torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
        )
    )

    max_length = position_ids.max() + 1

    return (query, key, value, indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length))
