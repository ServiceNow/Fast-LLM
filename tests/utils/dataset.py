import io
import pathlib
import typing

import datasets
import numpy as np
import PIL.Image

from fast_llm.data.preparator.gpt_memmap.config import GPTMemmapDatasetPreparatorConfig
from fast_llm.data.preprocessing.abstract import NullPreprocessingConfig
from fast_llm.data.preprocessing.image_patch import ImagePatchConfig
from fast_llm.data.preprocessing.language_model import LanguageModelPreprocessingConfig
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.utils import padded_cumsum
from tests.utils.global_variables import DATASET_CACHE, MODEL_TEST_VOCAB_SIZE, TOKENIZER_FILE, TOKENIZER_PATH


def download_santacoder_tokenizer():
    if not TOKENIZER_FILE.is_file():
        import transformers

        transformers.AutoTokenizer.from_pretrained("bigcode/santacoder").save_pretrained(TOKENIZER_PATH)


def get_random_text(
    num_documents: int = 1000,
    min_document_size: int = 5,
    max_document_size: int = 99,
    random_state: np.random.RandomState = np.random,
):
    # Randomize document sizes
    document_sizes = random_state.randint(min_document_size, max_document_size + 1, num_documents)
    size_cumsums = padded_cumsum(document_sizes)
    # Generate random ascii characters.
    random_text = random_state.randint(32, 127, document_sizes.sum(), dtype=np.uint8).tobytes().decode()
    # Gather text by documents.
    texts = [
        random_text[size_cumsums[document_index] : size_cumsums[document_index + 1]]
        for document_index in range(num_documents)
    ]
    return texts, document_sizes


def get_random_spans(
    document_sizes: np.ndarray,
    min_spans: int,
    max_spans: int,
    random_state: np.random.RandomState = np.random,
    use_last_format: bool = False,
):
    # Randomize span counts. Actual count may be lower for small documents.
    span_counts = random_state.randint(min_spans, max_spans + 1, len(document_sizes))
    # Generate random spans.
    return [
        [
            (begin, end - use_last_format)
            for begin, end in np.sort(
                random_state.choice(range(length), min(num_spans, length // 2) * 2, replace=False)
            )
            .reshape([-1, 2])
            .tolist()
        ]
        for length, num_spans in zip(document_sizes, span_counts, strict=True)
    ]


def get_random_preference_spans(texts, random_state: np.random.RandomState = np.random) -> dict[str, str]:
    texts_ = []
    chosen_spans = []
    rejected_spans = []
    for text in texts:
        # Split in three non-empty_chunks
        splits = np.sort(random_state.choice(range(1, len(text) - 1), 2, replace=False)).tolist()
        texts_.append(text[: splits[0]])
        chosen_spans.append(text[splits[0] : splits[1]])
        rejected_spans.append(text[splits[1] :])
    return {"text": texts_, "chosen_span": chosen_spans, "rejected_span": rejected_spans}


def _save_image_to_bytes(image: np.ndarray, format="PNG", mode="RGB"):
    buffer = io.BytesIO()
    PIL.Image.fromarray(image, mode).save(buffer, format=format)
    return buffer.getvalue()


def get_random_images(
    document_sizes: np.ndarray,
    min_images: int,
    max_images: int,
    min_image_size: int,
    max_image_size: int,
    random_state: np.random.RandomState = np.random,
):
    # Randomize image count for each sample.
    image_counts = random_state.randint(min_images, max_images + 1, num_documents := len(document_sizes))
    image_count_cumsums = padded_cumsum(image_counts)
    num_images = image_count_cumsums[-1]
    # Randomize image shapes.
    image_shapes = random_state.randint(min_image_size, max_image_size + 1, [num_images, 2])
    pixel_count_cumsum = padded_cumsum(image_shapes.prod(1) * 3)
    # Generate random pixels.
    pixels = random_state.randint(0, 256, pixel_count_cumsum[-1], dtype=np.uint8)
    # Convert pixels to image byte buffers.
    images = [
        _save_image_to_bytes(
            pixels[pixel_count_cumsum[image_index] : pixel_count_cumsum[image_index + 1]].reshape(
                [*image_shapes[image_index], 3]
            )
        )
        for image_index in range(num_images)
    ]
    # Gather images by documents.
    images = [
        images[image_count_cumsums[document_index] : image_count_cumsums[document_index + 1]]
        for document_index in range(num_documents)
    ]
    # Generate random image positions.
    image_positions = [
        np.sort(random_state.choice(range(document_size), image_counts[document_index], replace=False)).tolist()
        for document_index, document_size in enumerate(document_sizes)
    ]
    return images, image_positions


def _get_hf_test_dataset(
    seed: int = 1234,
    num_documents: int = 1000,
    min_document_size: int = 5,
    max_document_size: int = 99,
    min_loss_masking_spans: int = 0,
    max_loss_masking_spans: int = 0,
    has_preference_spans: bool = False,
    has_grpo_data: bool = False,
    min_images: int = 0,
    max_images: int = 0,
    min_image_size: int = 4,
    max_image_size: int = 32,
):
    random_state = np.random.RandomState(seed)
    # Generate random document sizes (character count).
    texts, document_sizes = get_random_text(num_documents, min_document_size, max_document_size, random_state)

    if has_preference_spans:
        dataset_dict = get_random_preference_spans(texts, random_state)
    else:
        dataset_dict: dict[str, typing.Any] = {"text": texts}

    if max_loss_masking_spans > 0:
        dataset_dict["loss_masking_spans"] = get_random_spans(
            document_sizes, min_loss_masking_spans, max_loss_masking_spans, random_state, use_last_format=True
        )

    if max_images > 0:
        dataset_dict["images"], dataset_dict["image_positions"] = get_random_images(
            document_sizes, min_images, max_images, min_image_size, max_image_size, random_state
        )

    if has_grpo_data:
        dataset_dict["advantages"] = random_state.randn(num_documents).tolist()

    return datasets.Dataset.from_dict(dataset_dict)


def _get_test_dataset(
    path: pathlib.Path,
    seed: int,
    tokenizer_path: str = TOKENIZER_PATH,
    max_vocab_size: int | None = None,
    documents_per_shard: int = 10**6,
    num_documents: int = 1000,
    min_document_size: int = 5,
    max_document_size: int = 99,
    min_loss_masking_spans: int = 0,
    max_loss_masking_spans: int = 0,
    has_preference_spans: bool = False,
    has_grpo_data: bool = False,
    splits: dict[str, float] | None = None,
    min_images: int = 0,
    max_images: int = 0,
    image_patch_config: ImagePatchConfig | None = None,
    min_image_size: int = 4,
    max_image_size: int = 32,
    config_only: bool = False,
) -> tuple[pathlib.Path, dict[str, typing.Any], pathlib.Path, LanguageModelPreprocessingConfig]:
    config_paths = (
        [path / "fast_llm_config.yaml"]
        if splits is None
        else [path / f"fast_llm_config_{split}.yaml" for split in splits]
    )
    hf_path = path / "hf"

    if not config_only and not all(config_path.is_file() for config_path in config_paths):
        # Not supported for parallel tests, but dataset should already exist anyway.
        assert DistributedConfig.default_world_size == 1
        dataset = _get_hf_test_dataset(
            seed=seed,
            num_documents=num_documents,
            min_document_size=min_document_size,
            max_document_size=max_document_size,
            min_loss_masking_spans=min_loss_masking_spans,
            max_loss_masking_spans=max_loss_masking_spans,
            has_preference_spans=has_preference_spans,
            has_grpo_data=has_grpo_data,
            min_images=min_images,
            max_images=max_images,
            min_image_size=min_image_size,
            max_image_size=max_image_size,
        )
        datasets.DatasetDict({"train": dataset}).save_to_disk(hf_path)
        source_schema = {"text": "text"}
        if max_loss_masking_spans > 0:
            source_schema["loss_masking_spans"] = "loss_masking_spans"
        if has_preference_spans:
            source_schema["chosen_span"] = "chosen_span"
            source_schema["rejected_span"] = "rejected_span"
        if max_images > 0:
            source_schema["images"] = "images"
            source_schema["image_positions"] = "image_positions"
        if has_grpo_data:
            source_schema["advantages"] = "advantages"

        download_santacoder_tokenizer()
        preparator_config = GPTMemmapDatasetPreparatorConfig.from_dict(
            {
                "dataset": {
                    "path": hf_path,
                    "load_from_disk": True,
                    "source_schema": source_schema,
                },
                "tokenizer": {"path": tokenizer_path, "max_vocab_size": max_vocab_size},
                "output_path": path,
                "documents_per_shard": documents_per_shard,
                "splits": splits,
                "image_patches": {} if image_patch_config is None else image_patch_config,
            }
        )
        preparator_config.run()

    config = (
        {"type": "file", "path": config_paths[0]}
        if splits is None
        else {
            split: {"type": "file", "path": config_path}
            for split, config_path in zip(splits, config_paths, strict=True)
        }
    )
    preprocessing = LanguageModelPreprocessingConfig(
        tokenizer={"type": "tokenizer", "path": tokenizer_path, "max_vocab_size": max_vocab_size},
        image_patches=NullPreprocessingConfig() if image_patch_config is None else image_patch_config,
        vocab_size=max_vocab_size,
        use_loss_masking_spans=max_loss_masking_spans > 0,
        use_preference_spans=has_preference_spans,
        use_grpo_data=has_grpo_data,
    )
    return path, config, hf_path, preprocessing


def get_common_test_dataset() -> (
    tuple[pathlib.Path, dict[str, typing.Any], pathlib.Path, LanguageModelPreprocessingConfig]
):
    return _get_test_dataset(DATASET_CACHE / "common_dataset", seed=1234)


def get_alt_test_dataset() -> (
    tuple[pathlib.Path, dict[str, typing.Any], pathlib.Path, LanguageModelPreprocessingConfig]
):
    return _get_test_dataset(DATASET_CACHE / "other_dataset", seed=2345)


def get_sharded_test_dataset() -> (
    tuple[pathlib.Path, dict[str, typing.Any], pathlib.Path, LanguageModelPreprocessingConfig]
):
    return _get_test_dataset(DATASET_CACHE / "common_dataset_sharded", seed=1234, documents_per_shard=350)


def get_split_test_dataset() -> (
    tuple[pathlib.Path, dict[str, typing.Any], pathlib.Path, LanguageModelPreprocessingConfig]
):
    return _get_test_dataset(
        DATASET_CACHE / "common_dataset_split", seed=1234, splits={"training": 1, "validation": 1}
    )


def get_split_sharded_test_dataset() -> (
    tuple[pathlib.Path, dict[str, typing.Any], pathlib.Path, LanguageModelPreprocessingConfig]
):
    return _get_test_dataset(
        DATASET_CACHE / "common_dataset_split_sharded",
        seed=1234,
        documents_per_shard=350,
        splits={"training": 1, "validation": 1},
    )


def get_test_dataset_with_loss_masking_spans(
    config_only: bool = False,
) -> tuple[pathlib.Path, dict[str, typing.Any], pathlib.Path, LanguageModelPreprocessingConfig]:
    return _get_test_dataset(
        DATASET_CACHE / "dataset_with_loss_masking_spans",
        seed=1234,
        max_loss_masking_spans=5,
        config_only=config_only,
    )


def get_test_dataset_with_preference_spans(
    config_only: bool = False,
) -> tuple[pathlib.Path, dict[str, typing.Any], pathlib.Path, LanguageModelPreprocessingConfig]:
    return _get_test_dataset(
        DATASET_CACHE / "dataset_with_preference_spans", seed=1234, has_preference_spans=True, config_only=config_only
    )


def get_test_dataset_with_image_patches(
    image_break_token: int | None = None, image_end_token: int | None = None, config_only: bool = False
) -> tuple[pathlib.Path, dict[str, typing.Any], pathlib.Path, LanguageModelPreprocessingConfig]:
    return _get_test_dataset(
        DATASET_CACHE / f"dataset_with_image_patches_{image_break_token}_{image_end_token}",
        seed=1234,
        max_images=2,
        image_patch_config=ImagePatchConfig(
            height=4,
            width=4,
            max_image_height=16,
            max_image_width=16,
            image_break_token=image_break_token,
            image_end_token=image_end_token,
        ),
        config_only=config_only,
    )


def get_model_test_dataset(config_only: bool = False):
    return _get_test_dataset(
        DATASET_CACHE / "model_dataset",
        seed=1234,
        num_documents=200,
        max_loss_masking_spans=5,
        has_grpo_data=True,
        max_vocab_size=MODEL_TEST_VOCAB_SIZE,
        splits={"training": 180, "validation": 18, "test": 2},
        config_only=config_only,
    )


def get_multimodal_test_dataset(config_only: bool = False):
    return _get_test_dataset(
        DATASET_CACHE / "model_dataset_multimodal",
        seed=1234,
        num_documents=200,
        max_vocab_size=MODEL_TEST_VOCAB_SIZE,
        max_images=2,
        image_patch_config=ImagePatchConfig(
            height=4,
            width=4,
            max_image_height=16,
            max_image_width=16,
            image_break_token=None,
            image_end_token=None,
        ),
        splits={"training": 180, "validation": 18, "test": 2},
        config_only=config_only,
    )
