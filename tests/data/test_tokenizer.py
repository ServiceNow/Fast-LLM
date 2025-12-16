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
    ("messages", "expected_tokens", "expected_trainable_indices"),
    (
        # Single turn: full assistant turn (<assistant>Hello</assistant>) is trainable
        (
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
            [49152, 27, 789, 29, 16946, 750, 789, 2293, 17822, 29, 7371, 750, 17822, 29, 49152],
            [7, 8, 9, 10, 11, 12, 13],
        ),
        # Multi-turn: both assistant turns are fully trainable
        (
            [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
                {"role": "user", "content": "C"},
                {"role": "assistant", "content": "D"},
            ],
            [49152, 27, 789, 29, 32, 750, 789, 2293, 17822, 29, 33, 750, 17822, 2293, 789, 29, 34, 750, 789, 2293, 17822, 29, 35, 750, 17822, 29, 49152],
            [7, 8, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25],
        ),
        # System + user + assistant: full assistant turn trainable
        (
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            [49152, 27, 3144, 29, 5815, 1139, 44569, 6928, 3144, 2293, 789, 29, 16946, 750, 789, 2293, 17822, 29, 7371, 750, 17822, 29, 49152],
            [15, 16, 17, 18, 19, 20, 21],
        ),
        # User only: no trainable tokens
        (
            [{"role": "user", "content": "Hi"}],
            [49152, 27, 789, 29, 16946, 750, 789, 29, 49152],
            [],
        ),
        # Long multi-turn (85 tokens, 3 assistant responses with tags, tests span machinery)
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
            [49152, 27, 3144, 29, 5815, 1139, 373, 44569, 2424, 11886, 954, 15737, 14516, 6928, 3144, 2293, 789, 29, 13938, 438, 331, 25016, 457, 12409, 562, 35838, 789, 2293, 17822, 29, 2111, 25016, 457, 12409, 562, 438, 4235, 280, 6928, 17822, 2293, 789, 29, 13938, 5028, 759, 42226, 35838, 789, 2293, 17822, 29, 2111, 25016, 457, 759, 42226, 438, 29784, 3556, 6928, 17822, 2293, 789, 29, 1996, 4413, 3326, 35838, 789, 2293, 17822, 29, 2111, 25016, 457, 4413, 3326, 438, 613, 1361, 6928, 17822, 29, 49152],
            list(range(27, 41)) + list(range(49, 63)) + list(range(70, 84)),
        ),
    ),
)
def test_tokenize_chat(common_tokenizer, messages, expected_tokens, expected_trainable_indices):
    common_tokenizer.tokenizer.chat_template = CHAT_TEMPLATE
    tokens, train_mask = common_tokenizer.tokenize_chat(messages)
    Assert.eq(tokens.tolist(), expected_tokens)
    Assert.eq([i for i, m in enumerate(train_mask) if m], expected_trainable_indices)


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
def test_mask_to_spans(train_mask, expected_loss_spans):
    from fast_llm.data.preparator.gpt_memmap.prepare import _mask_to_spans

    Assert.eq(_mask_to_spans(train_mask), expected_loss_spans)
