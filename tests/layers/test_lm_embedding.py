import pytest
import torch

from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.utils import Assert
from tests.utils.utils import get_base_model, get_stage

NUM_TOKENS = 64
HIDDEN_SIZE = 128
VOCAB_SIZE = 256


def _make_config(scale: bool) -> GPTModelConfig:
    return GPTModelConfig.from_dict(
        {
            "base_model": {
                "decoder": {"num_blocks": 0},
                "embeddings": {
                    "vocab_size": VOCAB_SIZE,
                    "scale_by_sqrt_hidden_size": scale,
                },
                "hidden_size": HIDDEN_SIZE,
            },
            "distributed": {"use_cuda": torch.cuda.is_available()},
        }
    )


@pytest.mark.slow
def test_embedding_no_scale():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, distributed = get_base_model(_make_config(scale=False))
    emb: LanguageModelEmbedding = model.embeddings
    get_stage([emb], distributed)
    emb.eval()

    token_ids = torch.randint(0, VOCAB_SIZE, (NUM_TOKENS,), device=device)
    actual = emb._forward(None, token_ids, None, False, None)

    weight = emb.word_embeddings_weight.detach()
    expected = torch.embedding(weight, token_ids).to(actual.dtype)
    Assert.rms_close(actual, expected, threshold=1e-5)


@pytest.mark.slow
def test_embedding_scale_by_sqrt_hidden_size():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, distributed = get_base_model(_make_config(scale=True))
    emb: LanguageModelEmbedding = model.embeddings
    get_stage([emb], distributed)
    emb.eval()

    token_ids = torch.randint(0, VOCAB_SIZE, (NUM_TOKENS,), device=device)
    actual = emb._forward(None, token_ids, None, False, None)

    weight = emb.word_embeddings_weight.detach()
    expected = torch.embedding(weight, token_ids).to(actual.dtype) * HIDDEN_SIZE**0.5
    Assert.rms_close(actual, expected, threshold=1e-5)
