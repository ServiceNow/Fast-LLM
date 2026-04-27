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

_HIDDEN_SIZE = 64
_HEADS = 4
_KV_HEADS = 2
_HEAD_SIZE = 16
_HEADS_PER_GROUP = _HEADS // _KV_HEADS


@dataclasses.dataclass
class AttentionTestConfig:
    name: str
    causal: bool = True
    window_size: int | None = None
    query_norm: bool = False
    key_norm: bool = False

    def get_attention_config(self, implementation: str = "backup") -> AttentionConfig:
        config: dict = {
            "heads": _HEADS,
            "head_groups": _KV_HEADS,
            "head_size": _HEAD_SIZE,
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
        return AttentionConfig.from_dict(config)

    def expected_output(
        self,
        input_: torch.Tensor,
        attention: Attention,
        lengths: list[int],
    ) -> torch.Tensor:
        """
        Independent reference: plain F.linear + torch.rms_norm + per-document einsum attention.
        No calls to Fast-LLM attention or norm internals.
        """
        with torch.no_grad():
            q = torch.nn.functional.linear(input_, attention.query.weight.detach()).unflatten(1, (_HEADS, _HEAD_SIZE))
            kv = torch.nn.functional.linear(input_, attention.key_value.weight.detach()).unflatten(
                1, (2 * _KV_HEADS, _HEAD_SIZE)
            )

            if self.query_norm:
                q = torch.rms_norm(q, (_HEAD_SIZE,), attention.query_norm.weight.detach(), 1e-5)
            if self.key_norm:
                key_normed = torch.rms_norm(
                    kv[:, :_KV_HEADS, :], (_HEAD_SIZE,), attention.key_norm.weight.detach(), 1e-5
                )
                kv = torch.cat([key_normed, kv[:, _KV_HEADS:, :]], dim=1)

            k, v = kv[:, :_KV_HEADS, :], kv[:, _KV_HEADS:, :]
            scale = _HEAD_SIZE**-0.5

            attn_out = torch.empty_like(q)
            offset = 0
            for length in lengths:
                q_doc = q[offset : offset + length]
                k_doc = k[offset : offset + length]
                v_doc = v[offset : offset + length]

                head_outputs = []
                for head_index in range(_HEADS):
                    group = head_index // _HEADS_PER_GROUP
                    scores = (q_doc[:, head_index, :].float() @ k_doc[:, group, :].float().T) * scale
                    if self.causal:
                        positions = torch.arange(length, device=input_.device)
                        # mask[i, j] = True if j <= i (query i can attend to key j)
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
]

_attention_norm_variants = [
    ("no_norm", {}),
    ("query_norm", {"query_norm": True}),
    ("key_norm", {"key_norm": True}),
    ("both_norms", {"query_norm": True, "key_norm": True}),
]

_attention_test_configs = [
    AttentionTestConfig(name=f"{base_name}_{variant_name}", **base_kwargs, **variant_kwargs)
    for base_name, base_kwargs in _base_attention_cases
    for variant_name, variant_kwargs in _attention_norm_variants
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

    # Float32 for precise comparison throughout.
    distributed_config = DistributedConfig(use_cuda=torch.cuda.is_available())
    distributed = Distributed(distributed_config)
    device = distributed.device

    hidden_dim = TensorDim("hidden", _HIDDEN_SIZE)
    attention: Attention = config.get_attention_config("backup").get_layer(
        distributed_config, hidden_dim, lr_scale=None, peft=None, return_bias=False
    )
    stage = get_stage([attention], distributed)

    # Independent reference check: compare the full forward pass to a plain-PyTorch reference
    # that uses F.linear, torch.rms_norm, and an explicit per-document attention loop.
    attention.eval()
    input_for_ref = torch.randn(num_tokens, _HIDDEN_SIZE, device=device)
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
    torch.testing.assert_close(
        out_fastllm, config.expected_output(input_for_ref, attention, lengths), rtol=1e-4, atol=1e-5
    )
    attention.train()

    # Packing equivalence check: packed backup must match per-sequence backup, forward and backward.
    hidden_states = torch.randn(num_tokens, _HIDDEN_SIZE, device=device, requires_grad=True)

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
            attention_backup_bf16, stage_backup_bf16, distributed_config_bf16, hidden_states_bf16, lengths, device
        )
        stage_backup_bf16.reset_gradients()

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

        Assert.rms_close(out_flash, out_ref_bf16, 2e-3)


@pytest.mark.parametrize(
    "lengths",
    [pytest.param(lengths, id=str(lengths)) for lengths in _attention_lengths],
)
@pytest.mark.parametrize(
    "config",
    [pytest.param(config, id=config.name) for config in _attention_test_configs],
)
def test_attention(config: AttentionTestConfig, lengths: list[int]) -> None:
    _test_attention(config, lengths)
