import pytest
import tokenizers
import torch
from transformers import PreTrainedTokenizerFast

from fast_llm.data.preparation.gpt_memmap.config import GPTMemmapDatasetPreparatorConfig
from fast_llm.data.preparation.gpt_memmap.prepare import GPTMemmapDatasetPreparator
from fast_llm.engine.config_utils.data_type import DataType


@pytest.fixture
def special_token_preparator(tmp_path):
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            vocab={"<unk>": 0, "<bos>": 1, "<eos>": 2, "hello": 3, "world": 4, "chosen": 5, "rejected": 6},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<bos>", eos_token="<eos>"
    )
    tokenizer.save_pretrained(tmp_path)

    def build(add_bos=None, add_eos=None, schema=None):
        fields = {
            "output_path": str(tmp_path / "prepared"),
            "dataset": {"path": "unused", "source_schema": schema or {"type": "document"}},
            "tokenizer": {"path": str(tmp_path)},
        }
        if add_bos is not None or add_eos is not None:
            fields["special_tokens"] = {}
        if add_bos is not None:
            fields["special_tokens"]["add_bos"] = add_bos
        if add_eos is not None:
            fields["special_tokens"]["add_eos"] = add_eos
        config = GPTMemmapDatasetPreparatorConfig.from_dict(fields)
        preparator = GPTMemmapDatasetPreparator(config)
        preparator._tokenizer = config.tokenizer.get_tokenizer()
        preparator._data_type = DataType.int32
        return preparator

    return build


@pytest.mark.parametrize(
    "add_bos,add_eos,expected",
    [
        (None, None, [1, 3, 4, 2]),
        (True, True, [1, 3, 4, 2]),
        (False, True, [3, 4, 2]),
        (True, False, [1, 3, 4]),
        (False, False, [3, 4]),
    ],
)
def test_document_special_token_options(special_token_preparator, add_bos, add_eos, expected):
    preparator = special_token_preparator(add_bos, add_eos, {"type": "document", "loss_masking_spans": "mask"})
    doc = preparator._prepare_sample({"text": "hello world", "mask": [[0, 4]]})
    assert doc.tokens.tolist() == expected
    offset = int(add_bos is not False)
    assert doc.loss_masking_spans.ranges == [(offset, offset + 1)]


def test_source_special_tokens_are_retained(special_token_preparator):
    doc = special_token_preparator(False, False)._prepare_sample({"text": "<bos> hello <eos>"})
    assert doc.tokens.tolist() == [1, 3, 2]


@pytest.mark.parametrize("add_bos", [True, False])
@pytest.mark.parametrize("add_eos", [True, False])
@pytest.mark.parametrize("template_bos", [True, False])
@pytest.mark.parametrize("template_eos", [True, False])
def test_conversation_special_token_options(special_token_preparator, add_bos, add_eos, template_bos, template_eos):
    preparator = special_token_preparator(add_bos, add_eos, {"type": "conversation"})
    preparator._tokenizer.tokenizer.chat_template = (
        ("{{ bos_token }}" if template_bos else "")
        + "{% generation %}{{ messages[0]['content'] }}{% endgeneration %}"
        + ("{{ eos_token }}" if template_eos else "")
    )
    doc = preparator._prepare_sample({"messages": [{"role": "assistant", "content": "hello"}]})
    assert doc.tokens.tolist() == (
        ([1] if add_bos or template_bos else []) + [3] + ([2] if add_eos or template_eos else [])
    )


@pytest.mark.parametrize("add_bos", [True, False])
@pytest.mark.parametrize("add_eos", [True, False])
def test_preference_special_token_options(special_token_preparator, add_bos, add_eos):
    preparator = special_token_preparator(
        add_bos,
        add_eos,
        {"type": "document", "chosen_span": "chosen", "rejected_span": "rejected"},
    )
    doc = preparator._prepare_sample({"text": "hello ", "chosen": "chosen", "rejected": "rejected"})
    assert int((doc.tokens == 1).sum()) == (2 if add_bos else 0)
    assert int((doc.tokens == 2).sum()) == (2 if add_eos else 0)
    for ranges, expected in [
        (doc.chosen_spans.ranges, [5] + ([2] if add_eos else [])),
        (doc.rejected_spans.ranges, [6] + ([2] if add_eos else [])),
    ]:
        begin, end = ranges[0]
        assert torch.equal(doc.tokens[begin:end], torch.tensor(expected, dtype=torch.int32))
