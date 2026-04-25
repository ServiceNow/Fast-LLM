import dataclasses
import functools

import pytest
import torch

from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
from fast_llm.models.gpt.config import GPTModelConfig
from tests.utils.utils import get_base_model, get_stage

NUM_TOKENS = 64
VOCAB_SIZE = 256
HIDDEN_SIZE = 128
NUM_POSITION_EMBEDDINGS = 128


@dataclasses.dataclass
class EmbeddingTestConfig:
    name: str
    embedding_scale: float = 1.0
    compute_dtype: DataType = DataType.float32
    full_precision_residual: bool = False
    with_position_embeddings: bool = False
    with_padding: bool = False

    @functools.cached_property
    def residual_dtype(self) -> torch.dtype:
        return (DataType.float32 if self.full_precision_residual else self.compute_dtype).torch

    def get_config(self) -> GPTModelConfig:
        embeddings: dict = {
            "vocab_size": VOCAB_SIZE,
            "embedding_scale": self.embedding_scale,
            "full_precision_residual": self.full_precision_residual,
        }
        if self.with_position_embeddings:
            embeddings["position_embeddings"] = {"enabled": True}
            embeddings["num_position_embeddings"] = NUM_POSITION_EMBEDDINGS
        return GPTModelConfig.from_dict(
            {
                "base_model": {
                    "decoder": {"num_blocks": 0},
                    "embeddings": embeddings,
                    "head": {"normalization": {"type": "rms_norm"}},
                    "hidden_size": HIDDEN_SIZE,
                },
                "distributed": {
                    "compute_dtype": self.compute_dtype,
                    "use_cuda": torch.cuda.is_available(),
                },
            }
        )

    def get_inputs(self, device: torch.device) -> dict:
        num_real_tokens = NUM_TOKENS // 2 if self.with_padding else NUM_TOKENS
        token_ids = torch.randint(0, VOCAB_SIZE, (NUM_TOKENS,), device=device)
        if self.with_padding:
            token_ids[num_real_tokens:] = -1
        token_dim = TensorDim("token", NUM_TOKENS)
        kwargs = {
            LanguageModelKwargs.token_ids: token_ids,
            LanguageModelKwargs.token_dim: token_dim,
            LanguageModelKwargs.hidden_token_dim: token_dim,
            LanguageModelKwargs.num_tokens: num_real_tokens,
        }
        if self.with_position_embeddings:
            kwargs[LanguageModelKwargs.position_ids] = torch.arange(NUM_TOKENS, device=device)
        return kwargs

    def get_reference_output(self, layer: LanguageModelEmbedding, kwargs: dict) -> torch.Tensor:
        token_ids = kwargs[LanguageModelKwargs.token_ids]
        mask_inputs = kwargs[LanguageModelKwargs.num_tokens] < kwargs[LanguageModelKwargs.token_dim].size

        word_weight = layer.word_embeddings_weight.detach()
        if mask_inputs:
            token_mask = token_ids >= 0
            embeddings = torch.embedding(word_weight, token_ids * token_mask)
        else:
            embeddings = torch.embedding(word_weight, token_ids)

        if layer.position_embeddings_weight is not None:
            embeddings = embeddings + torch.nn.functional.embedding(
                kwargs[LanguageModelKwargs.position_ids], layer.position_embeddings_weight.detach()
            )

        if mask_inputs:
            embeddings = embeddings * token_mask.unsqueeze(-1)

        if self.embedding_scale != 1.0:
            embeddings = embeddings * self.embedding_scale

        return embeddings.to(dtype=self.residual_dtype)


_base_cases = [
    ("default", {}),
    ("with_padding", {"with_padding": True}),
    ("with_position_embeddings", {"with_position_embeddings": True}),
]

_variants = [
    ("", {}),
    ("bfloat16", {"compute_dtype": DataType.bfloat16}),
    ("full_precision_residual", {"full_precision_residual": True}),
    ("embedding_scale", {"embedding_scale": 2.0}),
]


def _make_name(base_name: str, variant_name: str) -> str:
    return f"{base_name}_{variant_name}" if variant_name else base_name


_embedding_test_configs = [
    EmbeddingTestConfig(name=_make_name(base_name, variant_name), **base_kwargs, **variant_kwargs)
    for base_name, base_kwargs in _base_cases
    for variant_name, variant_kwargs in _variants
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_config",
    [pytest.param(c, id=c.name) for c in _embedding_test_configs],
)
def test_embedding(test_config: EmbeddingTestConfig):
    model, distributed = get_base_model(test_config.get_config())
    stage = get_stage([model.embeddings], distributed)
    layer: LanguageModelEmbedding = stage._layers[0]

    kwargs = test_config.get_inputs(distributed.device)
    output, _ = stage.forward(torch.empty(0, device=distributed.device), kwargs)

    expected = test_config.get_reference_output(layer, kwargs)
    threshold = 1e-5 if test_config.compute_dtype == DataType.float32 else 5e-3
    torch.testing.assert_close(output, expected, rtol=threshold, atol=threshold)
