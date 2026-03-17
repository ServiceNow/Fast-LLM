import pytest
import torch

from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.language_model import LanguageModelBatch
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.attention.attention import Attention, _flash_available
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.utils import Assert


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
