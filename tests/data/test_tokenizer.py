import pytest

from fast_llm.data.preprocessing.tokenizer import Tokenizer, TokenizerConfig
from fast_llm.utils import Assert
from tests.utils.dataset import download_santacoder_tokenizer
from tests.utils.global_variables import TOKENIZER_PATH


@pytest.fixture(scope="session")
def common_tokenizer() -> Tokenizer:
    download_santacoder_tokenizer()
    return TokenizerConfig(path=TOKENIZER_PATH).get_tokenizer()


TEXT = "hello world"


@pytest.mark.parametrize("extra_tokens", (False, True))
@pytest.mark.parametrize(
    ("spans", "expected_token_spans", "expected_tokens"),
    (
        ([], [], [7196, 5297]),  # No span
        ([(1, 3)], [(1, 2)], [71, 325, 303, 5297]),  # Simple span
        ([(2, 2)], [(1, 1)], [284, 47443, 5297]),  # Empty span
        ([(0, 11)], [(0, 2)], [7196, 5297]),  # Full span
        ([(1, 4), (6, 7)], [(1, 2), (4, 5)], [71, 1498, 78, 207, 86, 2231]),  # Two spans
        ([(1, 6), (4, 7)], [(1, 4), (2, 5)], [71, 1498, 78, 207, 86, 2231]),  # Overlapping spans
        ([(1, 7), (4, 6)], [(1, 5), (2, 4)], [71, 1498, 78, 207, 86, 2231]),  # Nested spans
        ([(1, 5), (5, 7)], [(1, 3), (3, 4)], [71, 325, 303, 365, 2231]),  # Consecutive spans
        ([(2, 4), (2, 4)], [(1, 2), (1, 2)], [284, 683, 78, 5297]),  # Duplicate spans
        ([(2, 3), (5, 8), (9, 11)], [(1, 2), (3, 4), (5, 6)], [284, 75, 303, 48485, 81, 1382]),  # Three spans
    ),
)
def test_tokenize_with_spans(common_tokenizer, spans, expected_token_spans, expected_tokens, extra_tokens):
    tokens, token_spans = common_tokenizer.tokenize_with_spans(
        TEXT, begin=extra_tokens, end=extra_tokens, text_spans=spans
    )
    if extra_tokens:
        expected_tokens = [common_tokenizer.bod_id, *expected_tokens, common_tokenizer.eod_id]
        expected_token_spans = [(begin + 1, end + 1) for begin, end in expected_token_spans]
    Assert.eq(tokens.tolist(), expected_tokens)
    Assert.eq(token_spans, expected_token_spans)
