import pytest
import torch

from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.language_model import LanguageModelBatch
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.attention.attention import Attention, _flash_available
from fast_llm.layers.attention.config import AttentionConfig, AttentionKwargs
from fast_llm.layers.attention.rotary.rotary import rotary_embeddings_real
from fast_llm.utils import Assert
from tests.utils.utils import get_stage


@pytest.mark.parametrize(("causal", "window_size"), ((True, None), (True, 50), (False, None)))
@pytest.mark.parametrize(
    "lengths",
    (
        [20, 32, 10, 11, 9, 18],
        [100],
        [2, 8, 22, 7, 6, 5, 1, 10, 4, 11, 3, 8, 4, 9],
        [5 for _ in range(20)],
    ),
)
@pytest.mark.skipif(not _flash_available, reason="Flash attention not available")
def test_attention_implementations(causal: bool, window_size: int | None, lengths: list[int]):
    """
    Check that the flash and backup attention implementation give the same result.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention: Attention = AttentionConfig(
        head_size=32,
        heads=4,
        head_groups=2,
        window_size=window_size,
        causal=causal,
    ).get_layer(
        distributed_config := DistributedConfig(compute_dtype="bfloat16"),
        TensorDim("hidden_size", 256),
        lr_scale=None,
        peft=None,
    )
    num_tokens = sum(lengths)

    query = torch.empty(num_tokens, 4, 32, dtype=torch.bfloat16, device=device).normal_()
    key = torch.empty(num_tokens, 2, 32, dtype=torch.bfloat16, device=device).normal_()
    value = torch.empty(num_tokens, 2, 32, dtype=torch.bfloat16, device=device).normal_()

    (model_input,) = LanguageModelBatch(
        tokens=torch.empty(num_tokens, dtype=torch.int64, device=device), lengths=lengths
    ).get_model_inputs(
        LanguageModelBatchPreprocessingConfig(
            distributed=distributed_config,
            predicted_tokens=0,
            return_cumulative_sequence_lengths=True,
            return_max_sequence_lengths=True,
            return_document_index=True,
        )
    )
    kwargs = model_input.to_kwargs()
    attention._preprocess_for_backup_attention(kwargs)

    out_backup = attention._attn_backup(query, key, value, kwargs)
    out_flash = attention._attn_flash(query, key, value, kwargs)

    Assert.rms_close(out_backup, out_flash, 2e-3)


@pytest.mark.slow
@pytest.mark.parametrize(
    "norm_flags",
    [
        ("q_only", True, False, False),
        ("k_only", False, True, False),
        ("v_only", False, False, True),
        ("qk", True, True, False),
        ("qkv", True, True, True),
    ],
)
def test_qk_norm_gradients(norm_flags):
    """
    Verify that q_norm/k_norm weight gradients are correctly accumulated, and that
    v_norm (fixed-scale, no learnable weight) correctly propagates input gradients.
    Only _query_key_value is exercised (not the full attention path), so NoRotary
    is used to avoid rotary kwargs.
    """
    _, use_q_norm, use_k_norm, use_v_norm = norm_flags
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads, head_groups, head_size, num_tokens = 4, 2, 32, 20
    hidden_size = heads * head_size
    rms_eps = 1e-5

    distributed_config = DistributedConfig(compute_dtype="float32", use_cuda=torch.cuda.is_available())
    hidden_dim = TensorDim("hidden_size", hidden_size)

    attention: Attention = AttentionConfig(
        head_size=head_size,
        heads=heads,
        head_groups=head_groups,
        rotary={"type": "none"},
        query_norm={"type": "rms_norm"} if use_q_norm else None,
        key_norm={"type": "rms_norm"} if use_k_norm else None,
        value_norm=use_v_norm,
        value_norm_eps=rms_eps,
    ).get_layer(distributed_config, hidden_dim, lr_scale=None, peft=None)

    distributed = Distributed(distributed_config)
    get_stage([attention], distributed)

    input_ = torch.randn(num_tokens, hidden_size, dtype=torch.float32, device=device, requires_grad=True)

    # Forward through the wrapped _query_key_value (differentiable via wrap_forward_backward)
    query, key_value = attention._query_key_value(input_, {})
    (query.sum() + key_value.sum()).backward()

    # --- Verify learnable norm weight grads are non-zero ---
    if use_q_norm:
        q_grad = attention.q_norm.weight.grad_buffer
        assert q_grad is not None, "q_norm weight grad_buffer is None"
        assert q_grad.abs().sum() > 0, "q_norm weight grad_buffer is all-zero"

    if use_k_norm:
        k_grad = attention.k_norm.weight.grad_buffer
        assert k_grad is not None, "k_norm weight grad_buffer is None"
        assert k_grad.abs().sum() > 0, "k_norm weight grad_buffer is all-zero"

    # --- Verify input gradient correctness against a pure-autograd reference ---
    q_w = attention.query.weight.detach()
    kv_w = attention.key_value.weight.detach()
    q_norm_w = attention.q_norm.weight.detach() if use_q_norm else None
    k_norm_w = attention.k_norm.weight.detach() if use_k_norm else None

    input_ref = input_.detach().requires_grad_(True)
    query_ref = torch.nn.functional.linear(input_ref, q_w).unflatten(1, (heads, head_size))
    kv_ref = torch.nn.functional.linear(input_ref, kv_w).unflatten(1, (2 * head_groups, head_size))

    if use_q_norm:
        query_ref = torch.rms_norm(query_ref, [head_size], q_norm_w, rms_eps)
    if use_k_norm or use_v_norm:
        k_ref, v_ref = kv_ref.chunk(2, dim=1)
        if use_k_norm:
            k_ref = torch.rms_norm(k_ref, [head_size], k_norm_w, rms_eps)
        if use_v_norm:
            v_ref = torch.rms_norm(v_ref, [head_size], None, rms_eps)
        kv_ref = torch.cat([k_ref, v_ref], dim=1)

    (query_ref.sum() + kv_ref.sum()).backward()

    Assert.rms_close_relative(input_.grad, input_ref.grad, threshold=1e-5, min_threshold=1e-6)


@pytest.mark.slow
def test_q_norm_rotary_gradients():
    """
    q_norm backward must see the unrotated q_norm output. Triton rotary is
    in-place, so this catches storage aliasing between q_norm and RoPE.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads, head_groups, head_size, num_tokens = 4, 2, 8, 6
    hidden_size = heads * head_size
    rms_eps = 1e-5

    distributed_config = DistributedConfig(compute_dtype="float32", use_cuda=torch.cuda.is_available())
    hidden_dim = TensorDim("hidden_size", hidden_size)
    attention: Attention = AttentionConfig(
        head_size=head_size,
        heads=heads,
        head_groups=head_groups,
        rotary={"type": "default"},
        query_norm={"type": "rms_norm"},
        value_norm=True,
        value_norm_eps=rms_eps,
    ).get_layer(distributed_config, hidden_dim, lr_scale=None, peft=None)

    distributed = Distributed(distributed_config)
    get_stage([attention], distributed)

    (model_input,) = LanguageModelBatch(
        tokens=torch.empty(num_tokens, dtype=torch.int64, device=device),
        lengths=[num_tokens],
    ).get_model_inputs(
        LanguageModelBatchPreprocessingConfig(
            distributed=distributed_config,
            predicted_tokens=0,
            return_document_index=True,
        )
    )
    kwargs = model_input.to_kwargs()
    attention.preprocess(kwargs)

    input_ = torch.randn(num_tokens, hidden_size, dtype=torch.float32, device=device, requires_grad=True)
    query, key_value = attention._query_key_value(input_, kwargs)
    query_grad = torch.randn_like(query)
    query_grad_ref = query_grad.clone()
    torch.autograd.backward((query, key_value), (query_grad, torch.zeros_like(key_value)))

    q_w = attention.query.weight.detach()
    q_norm_w = attention.q_norm.weight.detach()
    input_ref = input_.detach().requires_grad_(True)
    q_norm_w_ref = q_norm_w.detach().clone().requires_grad_(True)
    query_ref = torch.nn.functional.linear(input_ref, q_w).unflatten(1, (heads, head_size))
    query_ref = torch.rms_norm(query_ref, [head_size], q_norm_w_ref, rms_eps)
    query_ref = rotary_embeddings_real(query_ref, kwargs[AttentionKwargs.rotary_freq])
    query_ref.backward(query_grad_ref)

    Assert.rms_close_relative(input_.grad, input_ref.grad, threshold=1e-5, min_threshold=1e-6)
    Assert.rms_close_relative(attention.q_norm.weight.grad_buffer, q_norm_w_ref.grad, threshold=1e-5, min_threshold=1e-6)


@pytest.mark.slow
def test_attention_k_eq_v_gradients():
    """
    Verify Gemma4-style full attention uses one shared K=V projection and that
    gradients from the K and V branches are accumulated into that projection.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads, head_groups, head_size, num_tokens = 4, 2, 8, 6
    hidden_size = heads * head_size

    distributed_config = DistributedConfig(compute_dtype="float32", use_cuda=torch.cuda.is_available())
    hidden_dim = TensorDim("hidden_size", hidden_size)

    attention: Attention = AttentionConfig(
        head_size=head_size,
        heads=heads,
        head_groups=head_groups,
        rotary={"type": "none"},
        query_norm={"type": "rms_norm"},
        key_norm={"type": "rms_norm"},
        value_norm=True,
        attention_k_eq_v=True,
    ).get_layer(distributed_config, hidden_dim, lr_scale=None, peft=None)

    distributed = Distributed(distributed_config)
    get_stage([attention], distributed)
    attention.to(device)

    input_ = torch.randn(num_tokens, hidden_size, dtype=torch.float32, device=device, requires_grad=True)
    query, key_value = attention._query_key_value(input_, {})
    assert key_value.shape == (num_tokens, 2 * head_groups, head_size)
    assert attention.key_value.weight.shape == (head_groups * head_size, hidden_size)
    (query.sum() + key_value.sum()).backward()

    q_w = attention.query.weight.detach()
    kv_w = attention.key_value.weight.detach()
    q_norm_w = attention.q_norm.weight.detach()
    k_norm_w = attention.k_norm.weight.detach()

    input_ref = input_.detach().requires_grad_(True)
    query_ref = torch.nn.functional.linear(input_ref, q_w).unflatten(1, (heads, head_size))
    kv_raw_ref = torch.nn.functional.linear(input_ref, kv_w).unflatten(1, (head_groups, head_size))
    query_ref = torch.rms_norm(query_ref, [head_size], q_norm_w, 1e-5)
    k_ref = torch.rms_norm(kv_raw_ref, [head_size], k_norm_w, 1e-5)
    v_ref = torch.rms_norm(kv_raw_ref, [head_size], None, 1e-5)
    kv_ref = torch.cat([k_ref, v_ref], dim=1)
    (query_ref.sum() + kv_ref.sum()).backward()

    Assert.rms_close_relative(input_.grad, input_ref.grad, threshold=1e-5, min_threshold=1e-6)
