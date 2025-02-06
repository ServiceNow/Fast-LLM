from fast_llm.data.dataset.gpt.config import GPTMemmapDatasetConfig
from tests.common import DATASET_PREFIX, get_test_dataset_with_spans
from tests.data.common import compare_indexed_dataset_with_spans, get_dataset_config
from tests.data.test_memmap import MEMMAP_DATASET_LENGTH, MEMMAP_DATASET_TOKENS

_DATASET_PREFIX_SPANS = DATASET_PREFIX.with_name("with_spans")

SPANS_DATASET_EXPECTED_SAMPLES = {
    9: ([], []),
    10: ([80, 85, 4295, 4182, 489, 727, 84, 698, 1197, 583], [[4, 6], [8, 9]]),
    13: ([78, 727, 74, 317, 1358, 89], []),
    15: ([78], [[0, 0]]),
}


def test_gpt_data_with_spans():
    get_test_dataset_with_spans(prefix=_DATASET_PREFIX_SPANS)
    dataset = get_dataset_config(
        {
            "type": "memmap",
            "path": _DATASET_PREFIX_SPANS,
        },
        GPTMemmapDatasetConfig,
    ).build()
    compare_indexed_dataset_with_spans(
        dataset, MEMMAP_DATASET_LENGTH, MEMMAP_DATASET_TOKENS, SPANS_DATASET_EXPECTED_SAMPLES
    )
    # for i, sample in SPANS_DATASET_EXPECTED_SAMPLES.items():
    #     Assert.all_equal(np.array(sample[0], dtype=np.uint16), dataset.get(i, use_loss_masking_spans=True).token_ids)
    #     Assert.all_equal(
    #         np.array(sample[1]).reshape(-1, 2), dataset.get(i, use_loss_masking_spans=True).loss_masking_spans
    #     )
