import datasets
import pytest

from fast_llm.data.dataset.gpt.config import GPTDatasetFromFileConfig
from fast_llm.data.dataset.memmap.memmap import MemmapDataset
from fast_llm.data.document.language_model import LanguageModelDocument
from fast_llm.data.preparation.tokenizer import TokenizerConfig
from fast_llm.utils import Assert
from tests.data.common import get_dataset_config
from tests.data.test_preparator import COMMON_DATASET_LENGTH, COMMON_DATASET_TEXT
from tests.utils.dataset import get_test_dataset_with_loss_masking_spans
from tests.utils.global_variables import TOKENIZER_NAME

DATASET_WITH_SPAN_TOKENS = 47782
DATASET_WITH_SPAN_SAMPLES = {
    27: [50256, 63, 82, 11, 7456, 50256],
    30: [50256, 31, 85, 78, 27, 8220, 62, 43, 50256],
    31: [50256, 60, 55, 80, 30, 85, 22, 18, 50256],
    77: [50256, 73, 80, 85, 52, 22, 46, 5, 88, 78, 50256],
    87: [50256, 52, 48274, 11, 71, 50256],
}
HF_LOSS_MASKING_SPANS = {
    27: [[0, 1]],
    30: [[0, 1]],
    31: [[0, 0], [2, 2], [5, 5]],
    77: [[0, 0], [2, 2], [5, 5], [7, 7]],
    87: [[0, 0], [3, 3]],
}
TOKEN_LOSS_MASKING_SPANS = {
    27: [(1, 3)],
    30: [(1, 3)],
    31: [(1, 2), (3, 4), (6, 7)],
    77: [(1, 2), (3, 4), (6, 7), (8, 9)],
    87: [(1, 2), (3, 4)],
}


@pytest.mark.slow
def test_gpt_data_with_loss_masking_spans():
    _, config, hf_path, _ = get_test_dataset_with_loss_masking_spans()
    dataset: MemmapDataset[LanguageModelDocument] = get_dataset_config(config, GPTDatasetFromFileConfig).build()

    hf_dataset = datasets.load_from_disk(hf_path)["train"]
    tokenizer = TokenizerConfig(path=TOKENIZER_NAME).get_tokenizer()

    # Check global stats.
    Assert.eq(len(dataset), len(hf_dataset), COMMON_DATASET_LENGTH)
    Assert.eq(dataset.num_tokens, DATASET_WITH_SPAN_TOKENS)

    for index in range(0, 200, 8):
        expected_text = hf_dataset[index]["text"]
        expected_text_spans = [(begin, last + 1) for begin, last in hf_dataset[index]["loss_masking_spans"]]
        expected_tokens, expected_spans = tokenizer.tokenize_with_spans(
            hf_dataset[index]["text"],
            text_spans=[(begin, last + 1) for begin, last in hf_dataset[index]["loss_masking_spans"]],
        )
        document = dataset.get_document(index)

        # Compare tokens and token spans.
        Assert.all_equal(document.tokens, expected_tokens)
        Assert.eq(document.loss_masking_spans.ranges, expected_spans)

        # Compare text.
        text, text_spans = tokenizer.detokenize_with_spans(
            document.tokens, True, True, token_spans=document.loss_masking_spans.ranges
        )
        Assert.eq(text, expected_text)
        Assert.eq(text_spans, expected_text_spans)

    # Check some numerical values.
    for index in DATASET_WITH_SPAN_SAMPLES:
        Assert.eq(hf_dataset[index]["text"], COMMON_DATASET_TEXT[index])
        Assert.eq(hf_dataset[index]["loss_masking_spans"], HF_LOSS_MASKING_SPANS[index])
        document = dataset.get_document(index)
        Assert.eq(document.tokens.tolist(), DATASET_WITH_SPAN_SAMPLES[index])
        Assert.eq(document.loss_masking_spans.ranges, TOKEN_LOSS_MASKING_SPANS[index])
