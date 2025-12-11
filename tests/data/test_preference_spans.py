import datasets
import numpy as np
import pytest
import torch

from fast_llm.data.dataset.config import SamplingParameters
from fast_llm.data.dataset.gpt.config import GPTDatasetFromFileConfig
from fast_llm.data.dataset.memmap import MemmapDataset
from fast_llm.data.preprocessing.tokenizer import TokenizerConfig
from fast_llm.data.sample.language_model import LanguageModelSample
from fast_llm.utils import Assert
from tests.data.common import get_dataset_config
from tests.data.test_preparator import COMMON_DATASET_LENGTH
from tests.utils.dataset import get_common_test_dataset, get_test_dataset_with_preference_spans
from tests.utils.global_variables import TOKENIZER_NAME

DATASET_WITH_PREFERENCE_SPAN_TOKENS = 62163
DATASET_WITH_PREFERENCE_SPAN_TEXT = {
    27: ["`", "s,", "uh"],
    30: ["@v", "o<C", "O_L"],
    31: ["]Xq?", "v", "73"],
    77: ["j", "qvU7O", "&yo"],
    87: ["Uz", "l", ",h"],
}
DATASET_WITH_PREFERENCE_SPAN_SAMPLES = {
    27: [49152, 63, 82, 11, 49152, 49152, 63, 27799, 49152],
    30: [49152, 31, 85, 78, 27, 34, 49152, 49152, 31, 85, 46, 62, 43, 49152],
    31: [49152, 60, 55, 80, 30, 85, 49152, 49152, 60, 55, 80, 30, 22, 18, 49152],
    77: [49152, 73, 80, 85, 52, 22, 46, 49152, 49152, 73, 5, 11807, 49152],
    87: [49152, 52, 89, 75, 49152, 49152, 52, 89, 11, 71, 49152],
}
TOKEN_PREFERENCE_SPANS = {
    27: [(2, 5), (7, 9)],
    30: [(3, 7), (10, 14)],
    31: [(5, 7), (12, 15)],
    77: [(2, 8), (10, 13)],
    87: [(3, 5), (8, 11)],
}


@pytest.mark.slow
def test_gpt_data_with_spans():
    _, config, hf_path, preprocessing = get_test_dataset_with_preference_spans()
    dataset: MemmapDataset[LanguageModelSample] = get_dataset_config(config, GPTDatasetFromFileConfig).build(
        preprocessing
    )

    hf_dataset = datasets.load_from_disk(hf_path)["train"]
    tokenizer = TokenizerConfig(path=TOKENIZER_NAME).get_tokenizer()

    # Check global stats.
    Assert.eq(len(dataset), len(hf_dataset), COMMON_DATASET_LENGTH)

    for index in range(0, 200, 8):
        expected_text_split = [
            hf_dataset[index]["text"],
            hf_dataset[index]["chosen_span"],
            tokenizer.tokenizer.eos_token,
            tokenizer.tokenizer.bos_token,
            hf_dataset[index]["text"],
            hf_dataset[index]["rejected_span"],
        ]
        expected_text = "".join(expected_text_split)
        text_length_cumsum = np.cumsum([len(text) for text in expected_text_split]).tolist()
        expected_text_spans = [
            (text_length_cumsum[0], text_length_cumsum[2]),
            (text_length_cumsum[4], text_length_cumsum[5]),
        ]

        expected_tokens_split = [
            tokenizer.tokenize(expected_text_split[0], True, False),
            tokenizer.tokenize(expected_text_split[1], False, False),
            torch.tensor([tokenizer.eod_id], dtype=torch.int64),
            torch.tensor([tokenizer.bod_id], dtype=torch.int64),
            tokenizer.tokenize(expected_text_split[4], False, False),
            tokenizer.tokenize(expected_text_split[5], False, True),
        ]
        token_length_cumsum = np.cumsum([len(tokens) for tokens in expected_tokens_split])
        expected_tokens = torch.cat(expected_tokens_split)
        expected_token_spans = [
            (token_length_cumsum[0], token_length_cumsum[2]),
            (token_length_cumsum[4], token_length_cumsum[5]),
        ]

        document = dataset.get_document(index, parameters=SamplingParameters(num_samples=0, sequence_length=0))
        token_spans = document.chosen_spans.ranges + document.rejected_spans.ranges

        # Compare tokens and token spans.
        Assert.all_equal(document.tokens.tokens, expected_tokens)
        Assert.eq(token_spans, expected_token_spans)

        # Compare text.
        text, text_spans = tokenizer.detokenize_with_spans(document.tokens.tokens, True, True, token_spans=token_spans)
        Assert.eq(text, expected_text)
        Assert.eq(text_spans, expected_text_spans)

    # Check some numerical values.
    for index in DATASET_WITH_PREFERENCE_SPAN_SAMPLES:
        Assert.eq(
            [hf_dataset[index]["text"], hf_dataset[index]["chosen_span"], hf_dataset[index]["rejected_span"]],
            DATASET_WITH_PREFERENCE_SPAN_TEXT[index],
        )

        document = dataset.get_document(index, parameters=SamplingParameters(num_samples=0, sequence_length=0))
        Assert.eq(document.tokens.tokens.tolist(), DATASET_WITH_PREFERENCE_SPAN_SAMPLES[index])
        Assert.eq(document.chosen_spans.ranges + document.rejected_spans.ranges, TOKEN_PREFERENCE_SPANS[index])


@pytest.mark.slow
def test_gpt_data_with_missing_preference_spans():
    path, config, hf_path, _ = get_common_test_dataset()
    _, _, _, preprocessing = get_test_dataset_with_preference_spans(config_only=True)
    with pytest.raises(AssertionError, match="The dataset is missing required preference spans"):
        get_dataset_config(config, GPTDatasetFromFileConfig).build(preprocessing)
