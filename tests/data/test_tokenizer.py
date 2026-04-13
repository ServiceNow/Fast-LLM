import pytest

from fast_llm.data.preparation.tokenizer import Tokenizer, TokenizerConfig
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
        ([], [], [31373, 995]),  # No span
        ([(1, 3)], [(1, 2)], [71, 417, 5439, 995]),  # Simple span
        ([(2, 2)], [(1, 1)], [258, 18798, 995]),  # Empty span
        ([(0, 11)], [(0, 2)], [31373, 995]),  # Full span
        ([(1, 4), (6, 7)], [(1, 2), (4, 5)], [71, 695, 78, 220, 86, 1764]),  # Two spans
        ([(1, 6), (4, 7)], [(1, 4), (2, 5)], [71, 695, 78, 220, 86, 1764]),  # Overlapping spans
        ([(1, 7), (4, 6)], [(1, 5), (2, 4)], [71, 695, 78, 220, 86, 1764]),  # Nested spans
        ([(1, 5), (5, 7)], [(1, 2), (2, 3)], [71, 11109, 266, 1764]),  # Consecutive spans
        ([(2, 4), (2, 4)], [(1, 2), (1, 2)], [258, 297, 78, 995]),  # Duplicate spans
        ([(2, 3), (5, 8), (9, 11)], [(1, 2), (3, 4), (5, 6)], [258, 75, 5439, 24486, 81, 335]),  # Three spans
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


def test_validate_chat_template_no_template(common_tokenizer):
    """Tokenizer without chat template raises."""
    with pytest.raises(ValueError, match="does not have a chat template"):
        common_tokenizer.validate_chat_template()


def test_validate_chat_template_no_markers(common_tokenizer):
    """Tokenizer with chat template but no markers raises."""
    common_tokenizer.tokenizer.chat_template = "{{ messages }}"
    with pytest.raises(ValueError, match="does not contain.*generation"):
        common_tokenizer.validate_chat_template()


def test_validate_chat_template_with_markers(common_tokenizer):
    """Tokenizer with generation markers validates."""
    common_tokenizer.tokenizer.chat_template = "{% generation %}{{ m }}{% endgeneration %}"
    common_tokenizer.validate_chat_template()


# Realistic chat template following HF conventions (e.g., SmolLM3):
# The generation block includes the full assistant turn: opening tag, content, and closing tag.
# This ensures the model learns to emit the closing tag.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message.role == 'assistant' %}"
    "{% generation %}<assistant>{{ message.content }}</assistant>{% endgeneration %}"
    "{% else %}"
    "<{{ message.role }}>{{ message.content }}</{{ message.role }}>"
    "{% endif %}"
    "{% endfor %}"
)


@pytest.mark.parametrize(
    ("messages", "expected_tokens", "expected_loss_masking_spans"),
    (
        # Single turn: full assistant turn (<assistant>Hello</assistant>) is trainable
        # 17 tokens, loss mask spans cover 0-7 and 16
        (
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
            [50256, 27, 7220, 29, 17250, 3556, 7220, 6927, 562, 10167, 29, 15496, 3556, 562, 10167, 29, 50256],
            [(0, 7), (16, 17)],
        ),
        # Multi-turn: both assistant turns are fully trainable
        (
            [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
                {"role": "user", "content": "C"},
                {"role": "assistant", "content": "D"},
            ],
            [
                50256,
                27,
                7220,
                29,
                32,
                3556,
                7220,
                6927,
                562,
                10167,
                29,
                33,
                3556,
                562,
                10167,
                6927,
                7220,
                29,
                34,
                3556,
                7220,
                6927,
                562,
                10167,
                29,
                35,
                3556,
                562,
                10167,
                29,
                50256,
            ],
            [(0, 7), (16, 21), (30, 31)],
        ),
        # System + user + assistant: full assistant turn trainable
        (
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            [
                50256,
                27,
                10057,
                29,
                1639,
                389,
                7613,
                25970,
                10057,
                6927,
                7220,
                29,
                17250,
                3556,
                7220,
                6927,
                562,
                10167,
                29,
                15496,
                3556,
                562,
                10167,
                29,
                50256,
            ],
            [(0, 15), (24, 25)],
        ),
        # User only: no trainable tokens
        (
            [{"role": "user", "content": "Hi"}],
            [50256, 27, 7220, 29, 17250, 3556, 7220, 29, 50256],
            [(0, 9)],
        ),
        # Long multi-turn (3 assistant responses with tags, tests span machinery)
        (
            [
                {"role": "system", "content": "You are a helpful assistant that answers questions."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
                {"role": "user", "content": "What about Germany?"},
                {"role": "assistant", "content": "The capital of Germany is Berlin."},
                {"role": "user", "content": "And Italy?"},
                {"role": "assistant", "content": "The capital of Italy is Rome."},
            ],
            [
                50256,
                27,
                10057,
                29,
                1639,
                389,
                257,
                7613,
                8796,
                326,
                7429,
                2683,
                25970,
                10057,
                6927,
                7220,
                29,
                2061,
                318,
                262,
                3139,
                286,
                4881,
                30,
                3556,
                7220,
                6927,
                562,
                10167,
                29,
                464,
                3139,
                286,
                4881,
                318,
                6342,
                25970,
                562,
                10167,
                6927,
                7220,
                29,
                2061,
                546,
                4486,
                30,
                3556,
                7220,
                6927,
                562,
                10167,
                29,
                464,
                3139,
                286,
                4486,
                318,
                11307,
                25970,
                562,
                10167,
                6927,
                7220,
                29,
                1870,
                8031,
                30,
                3556,
                7220,
                6927,
                562,
                10167,
                29,
                464,
                3139,
                286,
                8031,
                318,
                10598,
                25970,
                562,
                10167,
                29,
                50256,
            ],
            [(0, 26), (40, 48), (62, 69), (83, 84)],
        ),
    ),
)
def test_tokenize_chat(common_tokenizer, messages, expected_tokens, expected_loss_masking_spans):
    common_tokenizer.tokenizer.chat_template = CHAT_TEMPLATE
    tokens, loss_masking_spans = common_tokenizer.tokenize_chat(messages)
    Assert.eq(tokens.tolist(), expected_tokens)
    Assert.eq(loss_masking_spans, expected_loss_masking_spans)


@pytest.mark.parametrize(
    ("train_mask", "expected_loss_spans"),
    (
        # All masked (no trainable tokens)
        ([False, False, False], [(0, 3)]),
        # All trainable (no spans)
        ([True, True, True], []),
        # Single trainable at start
        ([True, False, False], [(1, 3)]),
        # Single trainable at end
        ([False, False, True], [(0, 2)]),
        # Single trainable in middle
        ([False, True, False], [(0, 1), (2, 3)]),
        # Multiple trainable regions (simulates multi-turn conversation)
        ([False, False, True, True, False, False, True, True, True, False], [(0, 2), (4, 6), (9, 10)]),
        # Alternating
        ([False, True, False, True, False], [(0, 1), (2, 3), (4, 5)]),
    ),
)
def test_train_mask_to_loss_spans(train_mask, expected_loss_spans):
    from fast_llm.data.preparation.tokenizer import _train_mask_to_loss_spans

    Assert.eq(_train_mask_to_loss_spans(train_mask), expected_loss_spans)
