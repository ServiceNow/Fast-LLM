import os
import pathlib
import struct
import typing

import datasets
import numpy as np
import pytest
import torch
import yaml

from fast_llm.config import Field, FieldHint, config_class
from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.config import MemmapDatasetConfig, SampledDatasetConfig
from fast_llm.data.dataset.gpt.config import GPTSamplingData
from fast_llm.data.dataset.gpt.legacy_memmap import MEMMAP_DTYPES, MEMMAP_INDEX_HEADER, LegacyMemmapDataset
from fast_llm.data.dataset.sampled import logger
from fast_llm.data.preprocessing.abstract import PreprocessingConfig
from fast_llm.data.preprocessing.tokenizer import TokenizerConfig
from fast_llm.data.sample.language_model import LanguageModelSample
from fast_llm.data.sample.token import TokenSample
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert
from tests.utils.compare_tensor_logs import CompareConfig
from tests.utils.dataset import get_common_test_dataset
from tests.utils.distributed_configs import DistributedTestingConfig
from tests.utils.global_variables import DATASET_CACHE, MODEL_TEST_VOCAB_SIZE, TOKENIZER_NAME
from tests.utils.model_configs import ModelTestingGroup
from tests.utils.utils import requires_cuda

try:
    from fast_llm.csrc.data import build_sample_idx  # noqa

    _extension_available = True
except ImportError:
    _extension_available = False

MEGATRON_DATASET_PREFIX = DATASET_CACHE / "megatron_dataset/dataset"


def get_megatron_test_dataset(prefix: pathlib.Path = MEGATRON_DATASET_PREFIX):
    if not (
        prefix.with_suffix(".idx").is_file()
        and prefix.with_suffix(".bin").is_file()
        and prefix.parent.joinpath("fast_llm_config.yaml").is_file()
    ):
        _, _, hf_path, _ = get_common_test_dataset()
        hf_dataset = datasets.load_from_disk(hf_path)["train"]
        tokenizer = TokenizerConfig(path=TOKENIZER_NAME).get_tokenizer()
        samples = [
            LanguageModelSample(
                TokenSample((tokenizer.tokenize(document["text"]) % MODEL_TEST_VOCAB_SIZE).to(torch.uint16))
            )
            for document in hf_dataset
        ]

        MegatronMemmapDataset.write_dataset(prefix, samples)
        yaml.safe_dump(
            {"type": "memmap", "path": prefix.name}, prefix.parent.joinpath("fast_llm_config.yaml").open("w")
        )


@requires_cuda
@pytest.mark.model_testing_group(ModelTestingGroup.megatron)
def test_megatron(run_distributed_script, model_testing_config, run_test_script_base_path):
    path = run_test_script_base_path / "megatron"
    env = os.environ.copy()
    # Prevent Megatron from complaining.
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["NVTE_FLASH_ATTN"] = "0"
    get_megatron_test_dataset()
    run_distributed_script(
        [
            "Megatron-LM/pretrain_gpt.py",
            *model_testing_config.megatron_args,
            f"--data-path={MEGATRON_DATASET_PREFIX}",
            f"--structured-logs-dir={path}",
            f"--data-cache-path={path}",
        ],
        num_gpus=1,
        env=env,
    )


@requires_cuda
@pytest.mark.depends_on(on=["test_megatron[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.megatron)
def test_match_megatron(run_test_script_for_all_models, model_testing_config, compare_results_for_all_models):
    assert model_testing_config.megatron_args is not None

    ignore_tensors = (
        ".mixer.query_key_value.",
        ".mixer.query.",
        ".mixer.key_value.",
        ".mlp.layer_2.weight",
        ".mlp.experts.",
    )
    if model_testing_config.name == "mixtral":
        ignore_tensors += (".mlp.experts.", ".mlp.layer_1.weight")

    distributed_testing_config = DistributedTestingConfig(
        name="match_megatron",
        compare="megatron",
        config_args=[
            "model.distributed.compute_dtype=fp32",
            f'data.datasets.training={{"type":"megatron","path":{MEGATRON_DATASET_PREFIX}}}',
            "data.sampling.seed=1234",
            "model.base_model.use_megatron_initialization=True",
        ],
        num_gpus=1,
        compare_config=CompareConfig(sub_configs={(None, ignore_tensors): CompareConfig(ignore_tensors=True)}),
    )

    run_test_script_for_all_models(distributed_testing_config)
    compare_results_for_all_models(distributed_testing_config)


@config_class(dynamic_type={SampledDatasetConfig: "megatron"})
class MegatronDatasetConfig[SampleType: LanguageModelSample](MemmapDatasetConfig[SampleType]):
    _abstract: typing.ClassVar[bool] = False
    path: str = Field(
        desc="Dataset path (prefix).",
        hint=FieldHint.core,
    )

    def build(self, preprocessing: PreprocessingConfig) -> "LegacyMemmapDataset[SampleType]":
        return MegatronMemmapDataset(str(self.path).replace("/", "__"), self.path, preprocessing)


class MegatronMemmapDataset(LegacyMemmapDataset):
    def sample(self, sampling: GPTSamplingData) -> "MegatronSampledIndexedDataset":
        return MegatronSampledIndexedDataset(self, sampling)

    @classmethod
    def write_dataset(
        cls,
        prefix: pathlib.Path | str,
        documents: typing.Iterable[LanguageModelSample],
    ) -> None:
        # Initialize metadata
        dtype = None
        num_documents = 0
        lengths = []
        pointers = []
        offset = 0

        prefix = pathlib.Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        # Write the binary data file (.bin) lazily
        with prefix.with_suffix(".bin").open("wb") as bin_stream:
            for document in documents:
                token_ids = document.tokens.tokens
                # Infer dtype from the first document
                if dtype is None:
                    dtype = token_ids.dtype
                    assert dtype is not None, "Document dtype could not be inferred from the data."

                # Ensure all documents have the same dtype
                assert token_ids.dtype == dtype, f"Expected dtype {dtype}, got {token_ids.dtype}."

                # Write document to binary file
                bin_stream.write(token_ids.numpy().tobytes(order="C"))

                # Update metadata
                doc_length = len(token_ids)
                lengths.append(doc_length)
                pointers.append(offset)
                offset += doc_length * dtype.itemsize
                num_documents += 1

        # Finalize metadata arrays
        lengths = np.array(lengths, dtype=np.int32)
        pointers = np.array(pointers, dtype=np.int64)

        # Write the index file (.idx)
        with prefix.with_suffix(".idx").open("wb") as idx_stream:
            idx_stream.write(MEMMAP_INDEX_HEADER)
            # Version
            idx_stream.write(struct.pack("<Q", 1))
            # Data type
            idx_stream.write(struct.pack("<B", {y: x for x, y in MEMMAP_DTYPES.items()}[DataType.from_torch(dtype)]))
            # Number of sequences, same as documents in our case
            idx_stream.write(struct.pack("<Q", num_documents))
            # Number of documents, needs a +1 for some reason
            idx_stream.write(struct.pack("<Q", num_documents + 1))
            # Sequence (document) lengths
            idx_stream.write(lengths.tobytes(order="C"))
            # Sequence (document) begin offsets in the bin file
            idx_stream.write(pointers.tobytes(order="C"))
            # Document indices, unused but needed for compatibility with Megatron-LM
            idx_stream.write(np.arange(num_documents + 1, dtype=np.int64).tobytes(order="C"))


class MegatronSampledIndexedDataset(SampledDataset):
    """
    A GPT sampled dataset that exactly matches Megatron-LM, for testing purposes.
    Minimalistic implementation, implements only the required features.
    """

    def __init__(
        self,
        indexed_dataset: MegatronMemmapDataset,
        sampling: GPTSamplingData,
    ):
        assert isinstance(sampling, GPTSamplingData)
        self._indexed_dataset = indexed_dataset
        self._num_samples = sampling.parameters.num_samples
        self._sequence_length = sampling.parameters.sequence_length

        logger.info(f" > Sampling dataset {self._indexed_dataset.name} ...")
        document_sizes = self._indexed_dataset.get_document_sizes()
        num_documents = len(document_sizes)
        num_tokens = document_sizes.sum()
        np_rng = np.random.RandomState(seed=sampling.config.seed)

        # Assume less than one epoch.
        Assert.lt(self._sequence_length * self._num_samples, num_tokens)

        self._doc_idx = np.arange(num_documents, dtype=np.int32)
        np_rng.shuffle(self._doc_idx)

        assert _extension_available, (
            "The C++ extension for dataset sampling is missing." " Please make sure Fast-LLM is installed correctly."
        )

        self._sample_idx = build_sample_idx(document_sizes, self._doc_idx, self._sequence_length, 1, num_tokens, True)
        self._shuffle_idx = np.arange(0, self._sample_idx.shape[0] - 1, dtype=np.uint32)
        np_rng.shuffle(self._shuffle_idx)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> typing.Any:
        shuffled_idx = self._shuffle_idx[idx]
        doc_f, offset_f = self._sample_idx[shuffled_idx]
        doc_l, offset_l = self._sample_idx[shuffled_idx + 1]
        return LanguageModelSample.from_documents(
            [
                self._indexed_dataset.get_document(
                    self._doc_idx[doc].item(),
                    begin=(doc == doc_f) * offset_f,
                    end=offset_l + 1 if doc == doc_l else None,
                )
                for doc in range(doc_f, doc_l + 1)
            ]
        )

    @property
    def name(self) -> str:
        return self._indexed_dataset.name
