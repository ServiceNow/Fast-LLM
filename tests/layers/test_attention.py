import contextlib
import dataclasses

import pytest
import torch
import torch.nn.functional

from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.language_model import LanguageModelBatch
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.attention.attention import Attention, _flash_available
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.utils import Assert
from tests.utils.utils import get_stage

_HEADS = 4
_KV_HEADS = 2
_HEAD_SIZE = 16


def _compute_rotary_freqs(seq_len: int, head_size: int, theta: float, device: torch.device) -> torch.Tensor:
    angle_scales = theta ** (-torch.arange(0, 1, 2 / head_size, dtype=torch.float64, device=device))
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
    rotary: bool = False
    rotary_theta: float = 10000.0

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
        if self.rotary:
            config["rotary"] = {"type": "default", "theta": self.rotary_theta}
        return AttentionConfig.from_dict(config)

    def expected_output(
        self,
        input_: torch.Tensor,
        attention: Attention,
        lengths: list[int],
    ) -> torch.Tensor:
        """
        Independent reference: plain F.linear + rotary + per-document einsum attention.
        No calls to Fast-LLM attention internals.
        """
        with torch.no_grad():
            q = torch.nn.functional.linear(input_, attention.query.weight.detach()).unflatten(
                1, (self.heads, self.head_size)
            )
            kv = torch.nn.functional.linear(input_, attention.key_value.weight.detach()).unflatten(
                1, (2 * self.kv_heads, self.head_size)
            )

            if self.rotary:
                freqs = _compute_rotary_freqs(input_.shape[0], self.head_size, self.rotary_theta, input_.device)
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


_base_attention_cases = [
    ("causal", {"causal": True}),
    ("noncausal", {"causal": False}),
    ("window", {"causal": True, "window_size": 4}),
    ("mqa", {"causal": True, "kv_heads": 1}),
    ("mha", {"causal": True, "kv_heads": _HEADS}),
]

_attention_rotary_cases = [
    # Rotary: packing equivalence is skipped for multi-document inputs (packed rotary uses global
    # positions; per-sequence reference uses per-doc positions). All three checks run for single-doc inputs.
    ("causal_rotary", {"causal": True, "rotary": True}),
]

_attention_test_configs = [
    AttentionTestConfig(name=base_name, **base_kwargs)
    for base_name, base_kwargs in _base_attention_cases + _attention_rotary_cases
]

_attention_lengths = [
    [15],
    [6, 9],
    [4, 1, 10],
    [20, 32, 10, 11, 9, 18],
]


def _run_per_seq_reference(
    attention: Attention,
    stage,
    distributed_config: DistributedConfig,
    hidden_states: torch.Tensor,
    lengths: list[int],
    device: torch.device,
    with_backward: bool = True,
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
        if with_backward:
            stage.backward(torch.ones_like(out), context)
        out_refs.append(out.detach())
    return torch.cat(out_refs, dim=0)


@contextlib.contextmanager
def _no_tf32():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


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

    # Flash equivalence check: packed flash output must match per-sequence bfloat16 backup reference.
    if _flash_available:
        distributed_config_bf16 = DistributedConfig(compute_dtype=DataType.bfloat16, use_cuda=True)
        distributed_bf16 = Distributed(distributed_config_bf16)

        attention_backup_bf16: Attention = config.get_attention_config("backup").get_layer(
            distributed_config_bf16, hidden_dim, lr_scale=None, peft=None, return_bias=False
        )
        stage_backup_bf16 = get_stage([attention_backup_bf16], distributed_bf16)
        for param_bf16, param_f32 in zip(attention_backup_bf16.parameters(), attention.parameters(), strict=True):
            param_bf16.data.copy_(param_f32.data)

        hidden_states_bf16 = hidden_states.detach().to(torch.bfloat16)
        out_ref_bf16 = _run_per_seq_reference(
            attention_backup_bf16,
            stage_backup_bf16,
            distributed_config_bf16,
            hidden_states_bf16,
            lengths,
            device,
            with_backward=False,
        )

        attention_flash: Attention = config.get_attention_config("flash").get_layer(
            distributed_config_bf16, hidden_dim, lr_scale=None, peft=None, return_bias=False
        )
        stage_flash = get_stage([attention_flash], distributed_bf16)
        for param_flash, param_f32 in zip(attention_flash.parameters(), attention.parameters(), strict=True):
            param_flash.data.copy_(param_f32.data)

        (model_input_flash,) = LanguageModelBatch(
            tokens=torch.empty(num_tokens, dtype=torch.int64, device=device), lengths=lengths
        ).get_model_inputs(
            LanguageModelBatchPreprocessingConfig(
                distributed=distributed_config_bf16,
                predicted_tokens=0,
                **attention_flash.get_preprocessing_config(),
            )
        )
        kwargs_flash = model_input_flash.to_kwargs()
        attention_flash.preprocess(kwargs_flash)
        out_flash, _ = stage_flash.forward(hidden_states_bf16, kwargs_flash)

        Assert.rms_close_relative(out_flash, out_ref_bf16, 4e-3, 1e-7)


@pytest.mark.slow
@pytest.mark.parametrize(
    "lengths",
    [pytest.param(lengths, id=str(lengths)) for lengths in _attention_lengths],
)
@pytest.mark.parametrize(
    "config",
    [pytest.param(config, id=config.name) for config in _attention_test_configs],
)
def test_attention(config: AttentionTestConfig, lengths: list[int]) -> None:
    with _no_tf32():
        _test_attention(config, lengths)
