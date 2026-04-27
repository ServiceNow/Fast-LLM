import dataclasses
import functools

import pytest
import torch

from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.language_model import LanguageModelBatch
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.attention.attention import Attention, _flash_available
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.utils import Assert
from tests.utils.utils import get_stage

_NUM_TOKENS = 32
_HIDDEN_SIZE = 64
_HEADS = 4
_KV_HEADS = 2
_HEAD_SIZE = 16


@dataclasses.dataclass
class QKNormTestConfig:
    name: str
    query_norm: bool = False
    key_norm: bool = False

    def get_attention_config(self) -> AttentionConfig:
        config: dict = {
            "heads": _HEADS,
            "head_groups": _KV_HEADS,
            "head_size": _HEAD_SIZE,
            "add_linear_biases": False,
            "implementation": "backup",
        }
        if self.query_norm:
            config["query_norm"] = {"type": "rms_norm"}
        if self.key_norm:
            config["key_norm"] = {"type": "rms_norm"}
        return AttentionConfig.from_dict(config)

    @functools.cached_property
    def threshold(self) -> float:
        return 1e-5

    def expected_query(self, attention: Attention, q_flat: torch.Tensor) -> torch.Tensor:
        q = q_flat.unflatten(1, (_HEADS, _HEAD_SIZE))
        if self.query_norm:
            q = torch.rms_norm(q, (_HEAD_SIZE,), attention.query_norm.weight.detach(), 1e-5)
        return q

    def expected_key_value(self, attention: Attention, kv_flat: torch.Tensor) -> torch.Tensor:
        kv = kv_flat.unflatten(1, (2 * _KV_HEADS, _HEAD_SIZE))
        if self.key_norm:
            key = torch.rms_norm(kv[:, :_KV_HEADS, :], (_HEAD_SIZE,), attention.key_norm.weight.detach(), 1e-5)
            kv = torch.cat([key, kv[:, _KV_HEADS:, :]], dim=1)
        return kv


_base_qk_norm_cases = [
    ("no_norm", {}),
    ("query_norm", {"query_norm": True}),
    ("key_norm", {"key_norm": True}),
    ("both_norms", {"query_norm": True, "key_norm": True}),
]


def _make_qk_norm_name(name: str) -> str:
    return name


_qk_norm_test_configs = [
    QKNormTestConfig(name=_make_qk_norm_name(name), **kwargs) for name, kwargs in _base_qk_norm_cases
]


@pytest.mark.parametrize(
    "test_config",
    [pytest.param(c, id=c.name) for c in _qk_norm_test_configs],
)
def test_qk_norm(test_config: QKNormTestConfig):
    distributed_config = DistributedConfig(use_cuda=torch.cuda.is_available())
    distributed = Distributed(distributed_config)
    hidden_dim = TensorDim("hidden", _HIDDEN_SIZE)
    attention: Attention = test_config.get_attention_config().get_layer(
        distributed_config, hidden_dim, lr_scale=None, peft=None
    )
    get_stage([attention], distributed)
    attention.eval()

    device = distributed.device
    input_ = torch.randn(_NUM_TOKENS, _HIDDEN_SIZE, device=device)

    query, key_value, _ = attention._query_key_value_forward(input_, {})

    q_weight = attention.query.weight.detach()
    kv_weight = attention.key_value.weight.detach()
    ref_query = test_config.expected_query(attention, torch.nn.functional.linear(input_, q_weight))
    ref_key_value = test_config.expected_key_value(attention, torch.nn.functional.linear(input_, kv_weight))

    torch.testing.assert_close(query, ref_query, rtol=test_config.threshold, atol=test_config.threshold)
    torch.testing.assert_close(key_value, ref_key_value, rtol=test_config.threshold, atol=test_config.threshold)


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
