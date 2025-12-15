import pytest

from fast_llm.data.preprocessing.tokenizer import TokenizerConfig
from fast_llm.utils import Assert
from tests.utils.dataset import download_santacoder_tokenizer
from tests.utils.global_variables import TOKENIZER_PATH

TEXT = "hello world"


@pytest.fixture(scope="module")
def tokenizer():
    download_santacoder_tokenizer()
    return TokenizerConfig(path=TOKENIZER_PATH).get_tokenizer()


@pytest.mark.parametrize("extra_tokens", (False, True))
@pytest.mark.parametrize(
    ("spans", "expected_token_spans", "expected_tokens"),
    (
        ([], [], [7196, 5297]),
        ([(1, 3)], [(1, 2)], [71, 325, 303, 5297]),
        ([(2, 2)], [(1, 1)], [284, 47443, 5297]),
        ([(0, 11)], [(0, 2)], [7196, 5297]),
        ([(1, 4), (6, 7)], [(1, 2), (4, 5)], [71, 1498, 78, 207, 86, 2231]),
        ([(1, 6), (4, 7)], [(1, 4), (2, 5)], [71, 1498, 78, 207, 86, 2231]),
        ([(1, 7), (4, 6)], [(1, 5), (2, 4)], [71, 1498, 78, 207, 86, 2231]),
        ([(1, 5), (5, 7)], [(1, 3), (3, 4)], [71, 325, 303, 365, 2231]),
        ([(2, 4), (2, 4)], [(1, 2), (1, 2)], [284, 683, 78, 5297]),
        ([(2, 3), (5, 8), (9, 11)], [(1, 2), (3, 4), (5, 6)], [284, 75, 303, 48485, 81, 1382]),
    ),
)
def test_tokenize_with_spans(tokenizer, spans, expected_token_spans, expected_tokens, extra_tokens):
    tokens, token_spans = tokenizer.tokenize_with_spans(TEXT, begin=extra_tokens, end=extra_tokens, text_spans=spans)
    if extra_tokens:
        expected_tokens = [tokenizer.bod_id, *expected_tokens, tokenizer.eod_id]
        expected_token_spans = [(begin + 1, end + 1) for begin, end in expected_token_spans]
    Assert.eq(tokens.tolist(), expected_tokens)
    Assert.eq(token_spans, expected_token_spans)


def test_validate_chat_template_no_template(tokenizer):
    """Tokenizer without chat template raises."""
    with pytest.raises(ValueError, match="does not have a chat template"):
        tokenizer.validate_chat_template()


def test_validate_chat_template_no_markers(tokenizer):
    """Tokenizer with chat template but no markers raises."""
    tokenizer.tokenizer.chat_template = "{{ messages }}"
    with pytest.raises(ValueError, match="does not contain.*generation"):
        tokenizer.validate_chat_template()


def test_validate_chat_template_with_markers(tokenizer):
    """Tokenizer with generation markers validates."""
    tokenizer.tokenizer.chat_template = "{% generation %}{{ m }}{% endgeneration %}"
    tokenizer.validate_chat_template()


CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message.role == 'assistant' %}"
    "<assistant>{% generation %}{{ message.content }}{% endgeneration %}</assistant>"
    "{% else %}"
    "<{{ message.role }}>{{ message.content }}</{{ message.role }}>"
    "{% endif %}"
    "{% endfor %}"
)


@pytest.mark.parametrize(
    ("messages", "expected_text", "expected_spans"),
    (
        ([], "", []),
        (
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
            "<user>Hi</user><assistant>Hello</assistant>",
            [(0, 26), (31, 43)],
        ),
        (
            [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}, {"role": "user", "content": "C"}, {"role": "assistant", "content": "D"}],
            "<user>A</user><assistant>B</assistant><user>C</user><assistant>D</assistant>",
            [(0, 25), (26, 63), (64, 76)],
        ),
    ),
)
def test_apply_chat_template_with_spans(tokenizer, messages, expected_text, expected_spans):
    """Chat template produces correct text and masking spans."""
    tokenizer.tokenizer.chat_template = CHAT_TEMPLATE
    text, spans = tokenizer.apply_chat_template_with_spans(messages)
    Assert.eq(text, expected_text)
    Assert.eq(spans, expected_spans)
