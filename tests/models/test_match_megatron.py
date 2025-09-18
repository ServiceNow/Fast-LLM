import os
import typing

import numpy as np
import pytest

from fast_llm.config import Field, FieldHint, config_class
from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.gpt.config import GPTMemmapDatasetConfig, GPTSampledDatasetConfig, GPTSamplingData
from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset
from fast_llm.data.dataset.gpt.sampled import GPTSample, logger
from fast_llm.utils import Assert
from tests.utils.compare_tensor_logs import CompareConfig
from tests.utils.dataset import get_model_test_dataset
from tests.utils.distributed_configs import DistributedTestingConfig
from tests.utils.global_variables import MODEL_DATASET_PREFIX
from tests.utils.model_configs import ModelTestingGroup
from tests.utils.utils import requires_cuda

try:
    from fast_llm.csrc.data import build_sample_idx  # noqa

    _extension_available = True
except ImportError:
    _extension_available = False


@requires_cuda
@pytest.mark.model_testing_group(ModelTestingGroup.megatron)
def test_megatron(run_distributed_script, model_testing_config, run_test_script_base_path):
    path = run_test_script_base_path / "megatron"
    env = os.environ.copy()
    # Prevent Megatron from complaining.
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["NVTE_FLASH_ATTN"] = "0"
    get_model_test_dataset()
    run_distributed_script(
        [
            "Megatron-LM/pretrain_gpt.py",
            *model_testing_config.megatron_args,
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
            f'data.datasets.training={{"type":"megatron","path":{MODEL_DATASET_PREFIX}}}',
            "data.sampling.seed=1234",
            "model.base_model.use_megatron_initialization=True",
        ],
        num_gpus=1,
        compare_config=CompareConfig(sub_configs={(None, ignore_tensors): CompareConfig(ignore_tensors=True)}),
    )

    run_test_script_for_all_models(distributed_testing_config)
    compare_results_for_all_models(distributed_testing_config)


@config_class(dynamic_type={GPTSampledDatasetConfig: "megatron"})
class GPTMegatronDatasetConfig(GPTMemmapDatasetConfig):
    _abstract: typing.ClassVar[bool] = False
    path: str = Field(
        desc="Dataset path (prefix).",
        hint=FieldHint.core,
    )

    def build(self) -> "GPTMemmapDataset":
        return GPTMegatronMemmapDataset(
            str(self.path).replace("/", "__"), self.path, self.num_documents, self.num_tokens
        )


class GPTMegatronMemmapDataset(GPTMemmapDataset):
    def sample(self, sampling: GPTSamplingData) -> "MegatronGPTSampledIndexedDataset":
        return MegatronGPTSampledIndexedDataset(self, sampling)


class MegatronGPTSampledIndexedDataset(SampledDataset):
    """
    A GPT sampled dataset that exactly matches Megatron-LM, for testing purposes.
    Minimalistic implementation, implements only the required features.
    """

    def __init__(
        self,
        indexed_dataset: GPTMegatronMemmapDataset,
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
        sample_list = [
            self._indexed_dataset.get(
                self._doc_idx[doc].item(),
                offset=(doc == doc_f) * offset_f,
                length=offset_l + 1 - (doc == doc_f) * offset_f if doc == doc_l else None,
            )
            for doc in range(doc_f, doc_l + 1)
        ]
        token_ids = np.concatenate([sample.token_ids for sample in sample_list], dtype=np.int64)
        Assert.eq(len(token_ids), self._sequence_length + 1)

        return GPTSample(token_ids=token_ids)

    @property
    def name(self) -> str:
        return self._indexed_dataset.name
