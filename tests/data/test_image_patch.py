import hashlib
import io

import datasets
import numpy as np
import PIL.Image
import pytest

from fast_llm.data.dataset.gpt.config import GPTDatasetFromFileConfig, GPTSamplingParameters
from fast_llm.data.dataset.memmap import MemmapDataset
from fast_llm.data.sample.language_model import LanguageModelSample
from fast_llm.utils import Assert
from tests.data.common import get_dataset_config
from tests.data.test_preparator import COMMON_DATASET_LENGTH, COMMON_DATASET_TEXT
from tests.utils.dataset import get_test_dataset_with_image_patches

DATASET_WITH_IMAGE_PATCHES_TOKENS = [55750, 56809, 59145, 59145]
DATASET_WITH_IMAGE_PATCHES_IMAGE_MD5 = {
    27: [],
    30: ["a2c34e404506fe664efcdb520642f260"],
    31: ["3aca101f63e09f75b070fcc53ee44895", "0e027a3d45b34767a0cf5c67280dc825"],
    77: [],
    87: ["65597f534c1e2a257ac1b5282100d541", "612404822bc61a0b2a293890e7246621"],
}
DATASET_WITH_IMAGE_PATCHES_IMAGE_POSITIONS = {
    27: [],
    30: [3],
    31: [2, 4],
    77: [],
    87: [1, 2],
}
DATASET_WITH_IMAGE_PATCHES_IMAGE_SHAPES = {
    27: [],
    30: [(30, 4)],
    31: [(7, 22), (14, 24)],
    77: [],
    87: [(17, 4), (15, 12)],
}
DATASET_WITH_IMAGE_PATCHES_SAMPLES = {
    27: [49152, 63, 82, 11, 27799, 49152],
    30: [49152, 31, 2327, (4, 1), 27, 1448, 62, 43, 49152],
    31: [49152, 60, 55, (2, 4), 80, 30, (3, 4), 85, 22, 18, 49152],
    77: [49152, 13736, 85, 52, 22, 46, 5, 11807, 49152],
    87: [49152, 52, (4, 1), 89, (4, 3), 75, 11, 71, 49152],
}


def _shifted_range(begin: int, height_patches: int, width_patches: int, shift: int = 1):
    return [
        i
        for row in range(height_patches)
        for i in range(begin + row * (width_patches + shift), begin + row * (width_patches + shift) + width_patches)
    ]


DATASET_WITH_IMAGE_PATCHES_TOKEN_MAP = {
    27: [[] for _ in range(4)],
    30: [
        list(range(3, 7)),
        list(range(3, 7)),
        _shifted_range(3, 4, 1),
        _shifted_range(3, 4, 1),
    ],
    31: [
        [*range(3, 11), *range(13, 25)],
        [*range(3, 11), *range(14, 26)],
        _shifted_range(3, 2, 4) + _shifted_range(15, 3, 4),
        _shifted_range(3, 2, 4) + _shifted_range(15, 3, 4),
    ],
    77: [[] for _ in range(4)],
    87: [
        [*range(2, 6), *range(7, 19)],
        [*range(2, 6), *range(8, 20)],
        _shifted_range(2, 4, 1) + _shifted_range(11, 4, 3),
        _shifted_range(2, 4, 1) + _shifted_range(11, 4, 3),
    ],
}


def _position_ids(height_patches: int, width_patches: int):
    return [[i, j] for i in range(height_patches) for j in range(width_patches)]


DATASET_WITH_IMAGE_PATCHES_POSITIONS = {
    27: [],
    30: _position_ids(4, 1),
    31: _position_ids(2, 4) + _position_ids(3, 4),
    77: [],
    87: _position_ids(4, 1) + _position_ids(4, 3),
}
DATASET_WITH_IMAGE_PATCHES_LENGTHS = {
    27: [],
    30: [4],
    31: [8, 12],
    77: [],
    87: [4, 12],
}
DATASET_WITH_IMAGE_PATCHES_PATCHES_MD5 = {
    27: "d41d8cd98f00b204e9800998ecf8427e",
    30: "f9e5a216990b1a3646677195532dddec",
    31: "bd469b52ddd4f8f2bea4af5c7d843da9",
    77: "d41d8cd98f00b204e9800998ecf8427e",
    87: "946d6363c3440c4d3d7b5c684c6efcee",
}


def _get_image_tokens(
    height_patches: int, width_patches: int, image_break_token: int | None, image_end_token: int | None
):
    return ([-100] * width_patches + ([] if image_break_token is None else [image_break_token])) * (
        height_patches - 1
    ) + (
        [-100] * width_patches
        + (
            [image_end_token]
            if image_end_token is not None
            else [] if image_break_token is None else [image_break_token]
        )
    )


@pytest.mark.slow
@pytest.mark.parametrize("image_break_token", (None, 55))
@pytest.mark.parametrize("image_end_token", (None, 132))
def test_gpt_data_with_image_patches(image_break_token, image_end_token):
    _, config, hf_path = get_test_dataset_with_image_patches(image_break_token, image_end_token)
    dataset: MemmapDataset[LanguageModelSample] = get_dataset_config(config, GPTDatasetFromFileConfig).build()
    test_index = 2 * (image_break_token is not None) + (image_end_token is not None)

    hf_dataset = datasets.load_from_disk(hf_path)["train"]

    # Check global stats.
    Assert.eq(len(dataset), len(hf_dataset), COMMON_DATASET_LENGTH)
    Assert.eq(dataset.num_tokens, DATASET_WITH_IMAGE_PATCHES_TOKENS[test_index])

    # Check some numerical values.
    for index in DATASET_WITH_IMAGE_PATCHES_SAMPLES:
        Assert.eq(hf_dataset[index]["text"], COMMON_DATASET_TEXT[index])
        Assert.eq(
            [hashlib.md5(image).hexdigest() for image in hf_dataset[index]["images"]],
            DATASET_WITH_IMAGE_PATCHES_IMAGE_MD5[index],
        )
        Assert.eq(
            [np.array(PIL.Image.open(io.BytesIO(image))).shape[:2] for image in hf_dataset[index]["images"]],
            DATASET_WITH_IMAGE_PATCHES_IMAGE_SHAPES[index],
        )
        Assert.eq(hf_dataset[index]["image_positions"], DATASET_WITH_IMAGE_PATCHES_IMAGE_POSITIONS[index])

        document = dataset.get_document(
            index, parameters=GPTSamplingParameters(num_samples=0, sequence_length=0, use_images=True)
        )
        expected_tokens = [
            tokens
            for token_or_patches in DATASET_WITH_IMAGE_PATCHES_SAMPLES[index]
            for tokens in (
                _get_image_tokens(*token_or_patches, image_break_token, image_end_token)
                if isinstance(token_or_patches, tuple)
                else [token_or_patches]
            )
        ]
        Assert.eq(document.tokens.tokens.tolist(), expected_tokens)
        Assert.eq(document.image_patches.token_map.tolist(), DATASET_WITH_IMAGE_PATCHES_TOKEN_MAP[index][test_index])
        Assert.eq(document.image_patches.positions.tolist(), DATASET_WITH_IMAGE_PATCHES_POSITIONS[index])
        Assert.eq(document.image_patches.lengths, DATASET_WITH_IMAGE_PATCHES_LENGTHS[index])
        Assert.eq(
            hashlib.md5(document.image_patches.patches.numpy().tobytes()).hexdigest(),
            DATASET_WITH_IMAGE_PATCHES_PATCHES_MD5[index],
        )
