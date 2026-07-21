import dataclasses

import pytest
import torch
import torch.nn.functional

from fast_llm.data.document.block import LengthModelInputPreprocessor
from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig, LengthPreprocessingConfig
from fast_llm.data.document.language_model import LanguageModelBatch, LanguageModelInput
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.attention.attention import Attention, _flash_available, _KVCacheSlot
from fast_llm.layers.attention.config import AttentionConfig, AttentionKwargs
from fast_llm.utils import Assert
from tests.utils.utils import get_stage, no_tf32

_HEADS = 4
_KV_HEADS = 2
_HEAD_SIZE = 16


def _compute_rotary_freqs(
    seq_len: int,
    head_size: int,
    theta: float,
    device: torch.device,
    partial_rotary_factor: float | None = None,
) -> torch.Tensor:
    angle_scales = theta ** (-torch.arange(0, 1, 2 / head_size, dtype=torch.float64, device=device))
    if partial_rotary_factor is not None:
        rotary_pairs = round(head_size * partial_rotary_factor) // 2
        angle_scales[rotary_pairs:] = 0
    positions = torch.arange(seq_len, dtype=torch.float64, device=device)
    angles = torch.outer(positions, angle_scales)
    freqs = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1).to(torch.float32)
    return freqs.unsqueeze(1)  # (seq_len, 1, head_size)


def _apply_rotary(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    half = tensor.shape[-1] // 2
    re, im = tensor[..., :half], tensor[..., half:]
    cos, sin = freqs[..., :half], freqs[..., half:]
    return torch.cat([re * cos - im * sin, im * cos + re * sin], dim=-1)


@dataclasses.dataclass
class AttentionTestConfig:
    name: str
    heads: int = _HEADS
    kv_heads: int = _KV_HEADS
    head_size: int = _HEAD_SIZE
    causal: bool = True
    window_size: int | None = None
    query_norm: bool = False
    key_norm: bool = False
    value_norm: bool = False
    shared_key_value: bool = False
    rotary: bool = False
    rotary_theta: float = 10000.0
    rotary_partial_rotary_factor: float | None = None

    @property
    def hidden_size(self) -> int:
        return self.heads * self.head_size

    @property
    def heads_per_group(self) -> int:
        return self.heads // self.kv_heads

    def get_attention_config(self, implementation: str = "backup") -> AttentionConfig:
        config: dict = {
            "heads": self.heads,
            "head_groups": self.kv_heads,
            "head_size": self.head_size,
            "add_linear_biases": False,
            "causal": self.causal,
            "implementation": implementation,
        }
        if self.window_size is not None:
            config["window_size"] = self.window_size
        if self.query_norm:
            config["query_norm"] = {"type": "rms_norm"}
        if self.key_norm:
            config["key_norm"] = {"type": "rms_norm"}
        if self.value_norm:
            config["value_norm"] = {"type": "fixed_rms_norm"}
        if self.shared_key_value:
            config["shared_key_value"] = True
        if self.rotary:
            if self.rotary_partial_rotary_factor is not None:
                config["rotary"] = {
                    "type": "proportional",
                    "theta": self.rotary_theta,
                    "partial_rotary_factor": self.rotary_partial_rotary_factor,
                }
            else:
                config["rotary"] = {"type": "default", "theta": self.rotary_theta}
        return AttentionConfig.from_dict(config)

    def expected_output(
        self,
        input_: torch.Tensor,
        attention: Attention,
        lengths: list[int],
    ) -> torch.Tensor:
        """
        Independent reference: plain F.linear + torch.rms_norm + rotary + per-document einsum attention.
        No calls to Fast-LLM attention or norm internals.
        """
        with torch.no_grad():
            q = torch.nn.functional.linear(input_, attention.query.weight.detach()).unflatten(
                1, (self.heads, self.head_size)
            )
            if self.shared_key_value:
                key_projected = torch.nn.functional.linear(input_, attention.key_value.weight.detach()).unflatten(
                    1, (self.kv_heads, self.head_size)
                )
                kv = torch.cat([key_projected, key_projected], dim=1)
            else:
                kv = torch.nn.functional.linear(input_, attention.key_value.weight.detach()).unflatten(
                    1, (2 * self.kv_heads, self.head_size)
                )

            if self.query_norm:
                q = torch.rms_norm(q, (self.head_size,), attention.query_norm.weight.detach(), 1e-5)
            if self.key_norm:
                key_normed = torch.rms_norm(
                    kv[:, : self.kv_heads, :], (self.head_size,), attention.key_norm.weight.detach(), 1e-5
                )
                kv = torch.cat([key_normed, kv[:, self.kv_heads :, :]], dim=1)
            if self.value_norm:
                value_normed = torch.rms_norm(kv[:, self.kv_heads :, :], (self.head_size,), None, 1e-5)
                kv = torch.cat([kv[:, : self.kv_heads, :], value_normed], dim=1)

            if self.rotary:
                freqs = _compute_rotary_freqs(
                    input_.shape[0],
                    self.head_size,
                    self.rotary_theta,
                    input_.device,
                    partial_rotary_factor=self.rotary_partial_rotary_factor,
                )
                q = _apply_rotary(q, freqs)
                k_rotated = _apply_rotary(kv[:, : self.kv_heads, :], freqs)
                kv = torch.cat([k_rotated, kv[:, self.kv_heads :, :]], dim=1)

            k, v = kv[:, : self.kv_heads, :], kv[:, self.kv_heads :, :]
            scale = self.head_size**-0.5

            attn_out = torch.empty_like(q)
            offset = 0
            for length in lengths:
                q_doc = q[offset : offset + length]
                k_doc = k[offset : offset + length]
                v_doc = v[offset : offset + length]

                head_outputs = []
                for head_index in range(self.heads):
                    group = head_index // self.heads_per_group
                    scores = (q_doc[:, head_index, :].float() @ k_doc[:, group, :].float().T) * scale
                    if self.causal:
                        positions = torch.arange(length, device=input_.device)
                        mask = positions.unsqueeze(1) >= positions.unsqueeze(0)
                        if self.window_size is not None:
                            mask = mask & (positions.unsqueeze(1) - positions.unsqueeze(0) < self.window_size)
                        scores = scores.masked_fill(~mask, float("-inf"))
                    head_outputs.append((torch.softmax(scores, dim=-1) @ v_doc[:, group, :].float()).to(q.dtype))

                attn_out[offset : offset + length] = torch.stack(head_outputs, dim=1)
                offset += length

            return torch.nn.functional.linear(attn_out.flatten(1), attention.dense.weight.detach())


_LENGTHS_FULL = [[15], [6, 9], [4, 1, 10], [20, 32, 10, 11, 9, 18]]
_LENGTHS_SHORT = [[15], [4, 1, 10]]
_LENGTHS_SINGLE = [[15]]

_attention_test_cases: list[tuple[AttentionTestConfig, list[int]]] = []

# Mask, group, and window base cases — no norms, swept over all length sets.
for name, kwargs in (
    ("causal", {"causal": True}),
    ("noncausal", {"causal": False}),
    ("window", {"causal": True, "window_size": 4}),
    ("mqa", {"causal": True, "kv_heads": 1}),
    ("mha", {"causal": True, "kv_heads": _HEADS}),
):
    for lengths in _LENGTHS_FULL:
        _attention_test_cases.append((AttentionTestConfig(name=f"{name}_no_norm", **kwargs), lengths))

# Per-head norm variants on causal and shared key/value bases. Rotary bases use single-doc
# inputs because the packed and per-sequence rotary references diverge across boundaries.
for base_name, base_kwargs, variants, length_set in (
    (
        "causal",
        {"causal": True},
        (
            ("query_norm", {"query_norm": True}),
            ("key_norm", {"key_norm": True}),
            ("value_norm", {"value_norm": True}),
            ("both_norms", {"query_norm": True, "key_norm": True}),
            ("all_norms", {"query_norm": True, "key_norm": True, "value_norm": True}),
        ),
        _LENGTHS_SHORT,
    ),
    (
        "causal_rotary",
        {"causal": True, "rotary": True},
        (
            ("no_norm", {}),
            ("query_norm", {"query_norm": True}),
            ("key_norm", {"key_norm": True}),
            ("value_norm", {"value_norm": True}),
            ("both_norms", {"query_norm": True, "key_norm": True}),
            ("all_norms", {"query_norm": True, "key_norm": True, "value_norm": True}),
        ),
        _LENGTHS_SINGLE,
    ),
    (
        "shared_key_value",
        {"shared_key_value": True},
        (
            ("no_norm", {}),
            ("key_norm", {"key_norm": True}),
            ("value_norm", {"value_norm": True}),
            ("all_norms", {"query_norm": True, "key_norm": True, "value_norm": True}),
        ),
        _LENGTHS_SHORT,
    ),
    (
        "shared_key_value_rotary",
        {"shared_key_value": True, "rotary": True},
        (
            ("no_norm", {}),
            ("key_norm", {"key_norm": True}),
            ("value_norm", {"value_norm": True}),
            ("all_norms", {"query_norm": True, "key_norm": True, "value_norm": True}),
        ),
        _LENGTHS_SINGLE,
    ),
    (
        "shared_key_value_proportional_rotary",
        {"shared_key_value": True, "rotary": True, "rotary_partial_rotary_factor": 0.5},
        (
            ("no_norm", {}),
            ("key_norm", {"key_norm": True}),
            ("value_norm", {"value_norm": True}),
            ("all_norms", {"query_norm": True, "key_norm": True, "value_norm": True}),
        ),
        _LENGTHS_SINGLE,
    ),
):
    for variant_name, variant_kwargs in variants:
        for lengths in length_set:
            _attention_test_cases.append(
                (
                    AttentionTestConfig(name=f"{base_name}_{variant_name}", **base_kwargs, **variant_kwargs),
                    lengths,
                )
            )

# head_size > 256 — exercises the SDPA-only regime (flash caps at 256).
for name, kwargs in (
    ("large_head_causal_no_norm", {"causal": True, "head_size": 320}),
    ("large_head_mqa_no_norm", {"causal": True, "head_size": 320, "kv_heads": 1}),
):
    for lengths in _LENGTHS_SHORT:
        _attention_test_cases.append((AttentionTestConfig(name=name, **kwargs), lengths))

# Regression for the cu_seqlens_k canonicalization: attention with a non-zero leading K/V
# prefix from earlier documents (first_document_begin > 0) must match attention on the
# active documents alone, and the K/V grad for the inactive prefix must be exactly zero.
_attention_test_cases.append((AttentionTestConfig(name="first_document_begin"), [4, 1, 10]))


def _check_packed(
    implementation: str,
    config: AttentionTestConfig,
    hidden_dim: TensorDim,
    lengths: list[int],
    attention_f32: Attention,
    distributed_config: DistributedConfig,
    distributed: Distributed,
    hidden_states: torch.Tensor,
    out_ref: torch.Tensor,
    grads_ref: list[torch.Tensor],
    out_rtol: float,
    grad_rtol: float,
) -> None:
    attention_impl: Attention = config.get_attention_config(implementation).get_layer(
        distributed_config, hidden_dim, lr_scale=None, peft=None, return_bias=False
    )
    stage_impl = get_stage([attention_impl], distributed)
    for param_impl, param_f32 in zip(attention_impl.parameters(), attention_f32.parameters(), strict=True):
        param_impl.data.copy_(param_f32.data)
    (model_input,) = LanguageModelBatch(
        tokens=torch.empty(sum(lengths), dtype=torch.int64, device=hidden_states.device), lengths=lengths
    ).get_model_inputs(
        LanguageModelBatchPreprocessingConfig(
            distributed=distributed_config,
            predicted_tokens=0,
            **attention_impl.get_preprocessing_config(),
        )
    )
    kwargs_impl = model_input.to_kwargs()
    attention_impl.preprocess(kwargs_impl)
    out_impl, context = stage_impl.forward(hidden_states, kwargs_impl)
    stage_impl.backward(torch.ones_like(out_impl), context)
    Assert.rms_close_relative(out_impl, out_ref, out_rtol, 1e-7)
    for param_impl, grad_ref in zip(attention_impl.parameters(), grads_ref, strict=True):
        Assert.rms_close_relative(param_impl.grad_buffer, grad_ref, grad_rtol, 1e-7, msg=implementation)


def _run_per_seq_reference(
    attention: Attention,
    stage,
    distributed_config: DistributedConfig,
    hidden_states: torch.Tensor,
    lengths: list[int],
    device: torch.device,
) -> torch.Tensor:
    out_refs = []
    for length, hidden_slice in zip(lengths, torch.split(hidden_states, lengths, dim=0), strict=True):
        (model_input,) = LanguageModelBatch(
            tokens=torch.empty(length, dtype=torch.int64, device=device), lengths=[length]
        ).get_model_inputs(
            LanguageModelBatchPreprocessingConfig(
                distributed=distributed_config,
                predicted_tokens=0,
                **attention.get_preprocessing_config(),
            )
        )
        kwargs = model_input.to_kwargs()
        attention.preprocess(kwargs)
        out, context = stage.forward(hidden_slice, kwargs)
        stage.backward(torch.ones_like(out), context)
        out_refs.append(out.detach())
    return torch.cat(out_refs, dim=0)


def _test_attention(config: AttentionTestConfig, lengths: list[int]) -> None:
    num_tokens = sum(lengths)
    hidden_dim = TensorDim("hidden", config.hidden_size)

    distributed_config = DistributedConfig(use_cuda=torch.cuda.is_available())
    distributed = Distributed(distributed_config)
    device = distributed.device

    attention: Attention = config.get_attention_config("backup").get_layer(
        distributed_config, hidden_dim, lr_scale=None, peft=None, return_bias=False
    )
    stage = get_stage([attention], distributed)

    # Independent reference check with TF32 disabled on GPU.
    # Backup attention uses baddbmm over head groups; the reference uses per-head matmul — a
    # different summation order. With TF32 enabled these can differ by ~3e-4 on A100/H100;
    # disabling TF32 reduces the gap to ~1e-7, well within rtol=1e-4.
    attention.eval()
    input_for_ref = torch.randn(num_tokens, config.hidden_size, device=device)
    (model_input_ref,) = LanguageModelBatch(
        tokens=torch.empty(num_tokens, dtype=torch.int64, device=device), lengths=lengths
    ).get_model_inputs(
        LanguageModelBatchPreprocessingConfig(
            distributed=distributed_config,
            predicted_tokens=0,
            **attention.get_preprocessing_config(),
        )
    )
    kwargs_ref = model_input_ref.to_kwargs()
    attention.preprocess(kwargs_ref)
    with torch.no_grad():
        out_fastllm = attention(input_for_ref, kwargs_ref)
    expected = config.expected_output(input_for_ref, attention, lengths)
    Assert.rms_close_relative(out_fastllm, expected, 1e-5, 1e-7)
    attention.train()

    # Rotary uses global positions across packed docs; per-sequence uses local positions.
    # Packing equivalence and flash checks are only valid for single-document inputs with rotary.
    if config.rotary and len(lengths) > 1:
        return

    # Packing equivalence check: packed backup must match per-sequence backup, forward and backward.
    # TF32 disabled so that linear projections are row-by-row identical regardless of batch size;
    # different batch sizes can trigger different CUDA tiling strategies, causing ~1e-4 drift with TF32.
    hidden_states = torch.randn(num_tokens, config.hidden_size, device=device, requires_grad=True)

    out_ref = _run_per_seq_reference(attention, stage, distributed_config, hidden_states, lengths, device)
    names, params = zip(*list(attention.named_parameters()))
    grads_ref = [param.grad_buffer.clone() for param in params]
    stage.reset_gradients()

    (model_input_packed,) = LanguageModelBatch(
        tokens=torch.empty(num_tokens, dtype=torch.int64, device=device), lengths=lengths
    ).get_model_inputs(
        LanguageModelBatchPreprocessingConfig(
            distributed=distributed_config,
            predicted_tokens=0,
            **attention.get_preprocessing_config(),
        )
    )
    kwargs_packed = model_input_packed.to_kwargs()
    attention.preprocess(kwargs_packed)
    out_backup, context = stage.forward(hidden_states, kwargs_packed)
    stage.backward(torch.ones_like(out_backup), context)

    Assert.rms_close_relative(out_backup, out_ref, 1e-5, 1e-7)
    for name, param, grad_ref in zip(names, params, grads_ref, strict=True):
        Assert.rms_close_relative(param.grad_buffer, grad_ref, 1e-5, 1e-7, msg=name)
    stage.reset_gradients()

    _check_packed(
        "sdpa_dense",
        config,
        hidden_dim,
        lengths,
        attention,
        distributed_config,
        distributed,
        hidden_states,
        out_ref,
        grads_ref,
        1e-5,
        1e-5,
    )

    if not torch.cuda.is_available():
        return

    distributed_config_bf16 = DistributedConfig(compute_dtype=DataType.bfloat16, use_cuda=True)
    distributed_bf16 = Distributed(distributed_config_bf16)

    attention_backup_bf16: Attention = config.get_attention_config("backup").get_layer(
        distributed_config_bf16, hidden_dim, lr_scale=None, peft=None, return_bias=False
    )
    stage_backup_bf16 = get_stage([attention_backup_bf16], distributed_bf16)
    for param_bf16, param_f32 in zip(attention_backup_bf16.parameters(), attention.parameters(), strict=True):
        param_bf16.data.copy_(param_f32.data)

    hidden_states_bf16 = hidden_states.detach().to(torch.bfloat16).requires_grad_()
    out_ref_bf16 = _run_per_seq_reference(
        attention_backup_bf16, stage_backup_bf16, distributed_config_bf16, hidden_states_bf16, lengths, device
    )
    grads_ref_bf16 = [param.grad_buffer.clone() for param in attention_backup_bf16.parameters()]

    # bf16 grad rtol is looser than forward: reduction-order noise compounds through backward.
    for implementation in ("sdpa_dense", "flash", "sdpa_nested"):
        if implementation == "flash" and (not _flash_available or config.head_size > 256):
            continue
        if implementation == "sdpa_nested" and config.window_size is not None:
            continue
        _check_packed(
            implementation,
            config,
            hidden_dim,
            lengths,
            attention,
            distributed_config_bf16,
            distributed_bf16,
            hidden_states_bf16,
            out_ref_bf16,
            grads_ref_bf16,
            5e-3,
            1.5e-2,
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("config", "lengths"),
    [pytest.param(config, lengths, id=f"{config.name}-{lengths}") for config, lengths in _attention_test_cases],
)
def test_attention(config: AttentionTestConfig, lengths: list[int]) -> None:
    with no_tf32():
        if config.name == "first_document_begin":
            _test_first_document_begin(config, lengths)
        else:
            _test_attention(config, lengths)


def _check_first_document_begin(
    implementation: str,
    config: AttentionTestConfig,
    active_lengths: list[int],
    past_length: int,
    distributed_config: DistributedConfig,
    distributed: Distributed,
    hidden_dim: TensorDim,
    hidden_states: torch.Tensor,
    out_ref: torch.Tensor,
    grads_ref: list[torch.Tensor],
    ref_params: list[torch.Tensor],
    out_rtol: float,
    grad_rtol: float,
) -> None:
    """Run attention with `first_document_begin > 0` via a fake past slot, compare against
    the per-doc reference, and verify that the inactive prefix's K/V grad is exactly zero."""
    active_total = sum(active_lengths)
    total_k = past_length + active_total

    attention: Attention = config.get_attention_config(implementation).get_layer(
        distributed_config, hidden_dim, lr_scale=None, peft=None, return_bias=False
    )
    stage = get_stage([attention], distributed)
    for param, ref_param in zip(attention.parameters(), ref_params, strict=True):
        param.data.copy_(ref_param.data)

    # Build kwargs directly so we can inject `sequence_k_past` / `first_document_begin` > 0.
    model_input = LanguageModelInput(tokens=torch.empty(active_total, dtype=torch.int64, device=hidden_states.device))
    LengthModelInputPreprocessor(
        lengths=active_lengths,
        sequence_k_past=past_length,
        first_document_begin=past_length,
        last_document_end=total_k,
        device=hidden_states.device,
        unpadded_length=active_total,
        sequence_length=total_k,
    ).preprocess(
        model_input,
        LengthPreprocessingConfig(distributed=distributed_config, **attention.get_preprocessing_config()),
    )

    # Fake "past" K/V slot with arbitrary data in the leading-prefix region; the narrow must
    # drop those positions, so their contents should not influence output or gradients.
    slot = _KVCacheSlot(
        buffer=torch.randn(
            total_k, 2 * config.kv_heads, config.head_size, device=hidden_states.device, dtype=hidden_states.dtype
        ),
        frontier=past_length,
    )
    kwargs = model_input.to_kwargs()
    kwargs[AttentionKwargs.past_key_values] = [slot]
    attention.preprocess(kwargs)

    hidden_states_test = hidden_states.detach().clone().requires_grad_()
    out, context = stage.forward(hidden_states_test, kwargs)
    stage.backward(torch.ones_like(out), context)

    Assert.rms_close_relative(out, out_ref, out_rtol, 1e-7, msg=implementation)
    for param, grad_ref in zip(attention.parameters(), grads_ref, strict=True):
        Assert.rms_close_relative(param.grad_buffer, grad_ref, grad_rtol, 1e-7, msg=implementation)
    # Specific guarantee of the fix: K/V grad for the inactive leading prefix must be zero.
    assert slot.grad_buffer is not None, f"{implementation}: grad_buffer not populated"
    assert (slot.grad_buffer[:past_length] == 0).all(), f"{implementation}: dK/dV for inactive prefix not zero"


def _test_first_document_begin(config: AttentionTestConfig, lengths: list[int]) -> None:
    """Regression for the cu_seqlens_k canonicalization. Attention with
    `first_document_begin > 0` (a non-zero K/V prefix from earlier documents that the current
    micro-sequence does not attend to) must produce the same output and gradients as
    attention on the active documents alone, and the K/V grad for the inactive prefix must
    be exactly zero — the specific guarantee of the fix.
    """
    past_length = 7
    active_total = sum(lengths)

    distributed_config = DistributedConfig(use_cuda=torch.cuda.is_available())
    distributed = Distributed(distributed_config)
    device = distributed.device
    hidden_dim = TensorDim("hidden", config.hidden_size)

    # fp32 reference: per-doc backup attention on the active documents alone.
    attention_ref: Attention = config.get_attention_config("backup").get_layer(
        distributed_config, hidden_dim, lr_scale=None, peft=None, return_bias=False
    )
    stage_ref = get_stage([attention_ref], distributed)
    hidden_states = torch.randn(active_total, config.hidden_size, device=device, requires_grad=True)
    (model_input_ref,) = LanguageModelBatch(
        tokens=torch.empty(active_total, dtype=torch.int64, device=device), lengths=lengths
    ).get_model_inputs(
        LanguageModelBatchPreprocessingConfig(
            distributed=distributed_config,
            predicted_tokens=0,
            **attention_ref.get_preprocessing_config(),
        )
    )
    kwargs_ref = model_input_ref.to_kwargs()
    attention_ref.preprocess(kwargs_ref)
    out_ref, context_ref = stage_ref.forward(hidden_states, kwargs_ref)
    stage_ref.backward(torch.ones_like(out_ref), context_ref)
    grads_ref = [param.grad_buffer.clone() for param in attention_ref.parameters()]
    ref_params = list(attention_ref.parameters())

    for implementation in ("backup", "sdpa_dense"):
        _check_first_document_begin(
            implementation,
            config,
            lengths,
            past_length,
            distributed_config,
            distributed,
            hidden_dim,
            hidden_states,
            out_ref.detach(),
            grads_ref,
            ref_params,
            1e-5,
            1e-5,
        )

    if not torch.cuda.is_available():
        return

    # bf16 reference for flash + sdpa_nested (flash rejects fp32).
    distributed_config_bf16 = DistributedConfig(compute_dtype=DataType.bfloat16, use_cuda=True)
    distributed_bf16 = Distributed(distributed_config_bf16)
    attention_ref_bf16: Attention = config.get_attention_config("backup").get_layer(
        distributed_config_bf16, hidden_dim, lr_scale=None, peft=None, return_bias=False
    )
    stage_ref_bf16 = get_stage([attention_ref_bf16], distributed_bf16)
    for param_bf16, param_f32 in zip(attention_ref_bf16.parameters(), ref_params, strict=True):
        param_bf16.data.copy_(param_f32.data)
    hidden_states_bf16 = hidden_states.detach().to(torch.bfloat16).requires_grad_()
    (model_input_ref_bf16,) = LanguageModelBatch(
        tokens=torch.empty(active_total, dtype=torch.int64, device=device), lengths=lengths
    ).get_model_inputs(
        LanguageModelBatchPreprocessingConfig(
            distributed=distributed_config_bf16,
            predicted_tokens=0,
            **attention_ref_bf16.get_preprocessing_config(),
        )
    )
    kwargs_ref_bf16 = model_input_ref_bf16.to_kwargs()
    attention_ref_bf16.preprocess(kwargs_ref_bf16)
    out_ref_bf16, context_ref_bf16 = stage_ref_bf16.forward(hidden_states_bf16, kwargs_ref_bf16)
    stage_ref_bf16.backward(torch.ones_like(out_ref_bf16), context_ref_bf16)
    grads_ref_bf16 = [param.grad_buffer.clone() for param in attention_ref_bf16.parameters()]
    ref_params_bf16 = list(attention_ref_bf16.parameters())

    for implementation in ("flash", "sdpa_nested"):
        if implementation == "flash" and (not _flash_available or config.head_size > 256):
            continue
        _check_first_document_begin(
            implementation,
            config,
            lengths,
            past_length,
            distributed_config_bf16,
            distributed_bf16,
            hidden_dim,
            hidden_states_bf16,
            out_ref_bf16.detach(),
            grads_ref_bf16,
            ref_params_bf16,
            5e-3,
            1.5e-2,
        )
