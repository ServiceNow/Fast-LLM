import json
import logging
import math
import pathlib
import typing
import warnings
import numpy as np

from fast_llm.models.stardoc.config import StarDocDataConfig
from fast_llm.models.stardoc.stardoc_dataset import StarDocDataset
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.config_utils.run import get_run, log_main_rank
from fast_llm.data.data import Data
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.data.dataset import BlendedDataset, SampledDataset, Sampler
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


def normalize_probs(p: list[float]) -> list[float]:
    p = np.array(p)
    Assert.custom(lambda x: np.all(x >= 0), p)
    p_sum = p.sum()
    Assert.gt(p_sum, 0)
    return (p / p_sum).tolist()


class StarDocData(Data):
    """
    A class for all dataset needs for StarDoc.
    """
    _sampled_datasets: dict[PhaseType, dict[str, SampledDataset]]
    _blended_datasets: dict[PhaseType, SampledDataset]
    _tokenizer: Tokenizer | None
    _distributed: Distributed
    _cache_dir: pathlib.Path | None
    _samples_per_phase: dict[PhaseType, int]
    _phases: typing.ClassVar[tuple[PhaseType, ...]] = (PhaseType.training, PhaseType.validation, PhaseType.test)

    def __init__(
        self,
        config: StarDocDataConfig,
        distributed_config: DistributedConfig,
        vocab_size: int,
        max_sequence_length: int,
    ):
        """
        Create the data and gather some basic information on the dataset(s).
        Should be `setup` before use.
        """
        self._config = config.validate()
        self._distributed_config = distributed_config.validate()
        self._vocab_size = vocab_size
        self._max_sequence_length = max_sequence_length
        Assert.eq(len(self._config.split), len(self._phases))
        self._phase_split = {
            phase: ratio for phase, ratio in zip(self._phases, normalize_probs(self._config.split)) if ratio > 0
        }
        data_base_path = None
        Assert.eq(len(self._config.path), 1)
        data_path = pathlib.Path(self._config.path[0])
        dataset_defs = json.load(data_path.open("r"))
        data_base_path = data_path.parent
        dataset_prefixes = [dataset_def["prefix"] for dataset_def in dataset_defs["datasets"]]
        dataset_weights = normalize_probs([dataset_def["weight"] for dataset_def in dataset_defs["datasets"]])
        self._build_and_sample_dataset = self._build_and_sample_stardoc_dataset

        dataset_names = [
            f"dataset_{i}_{'dummy' if prefix is None else prefix.replace('/','__')}"
            for i, prefix in enumerate(dataset_prefixes)
        ]
        self._num_datasets = len(dataset_names)
        self._dataset_prefixes = {
            name: (
                None
                if prefix is None
                else (
                    pathlib.Path(prefix).resolve()
                    if data_base_path is None
                    else (pathlib.Path(data_base_path) / prefix).resolve()
                )
            )
            for name, prefix in zip(dataset_names, dataset_prefixes)
        }
        self._dataset_weights = {name: weight for name, weight in zip(dataset_names, dataset_weights)}
    
    def setup(self, distributed: Distributed, samples_per_phase: dict[PhaseType, int]):
        """
        Load the datasets. This may take a while and a significant amount of cpu memory.
        """
        run = get_run()
        Assert.leq(set(samples_per_phase), set(self._phase_split))
        log_main_rank(f"Preparing {self._num_datasets} datasets. This may take several minutes.")
        self._tokenizer = Tokenizer(self._config.tokenizer, max_sequence_length=self._max_sequence_length)
        self._distributed = distributed
        self._cache_dir = run.dataset_cache_dir
        self._samples_per_phase = samples_per_phase
        if self._cache_dir is None:
            warnings.warn(f"Using the dataset directory for the index cache.")

        # Build and split datasets.
        self._sampled_datasets = {phase: {} for phase in self._samples_per_phase}
        for i, (name, weight) in enumerate(self._dataset_weights.items()):
            if i % 100 == 0 and i > 0:
                log_main_rank(f"Prepared {i} of {self._num_datasets} datasets.")
            dataset_samples_per_phase = {}
            for phase, samples_per_phase in self._samples_per_phase.items():
                expected_samples = self._dataset_weights[name] * samples_per_phase
                # Add 5 times the standard deviation (of a binomial distribution)
                # so the probability of sampling more than this amount during blending is negligible.
                dataset_samples_per_phase[phase] = math.ceil(
                    expected_samples
                    + 5 * math.sqrt(expected_samples * self._dataset_weights[name] * (1 - self._dataset_weights[name]))
                )
            sampled_datasets = self._build_and_sample_dataset(name, dataset_samples_per_phase)
            for phase, dataset in sampled_datasets.items():
                self._sampled_datasets[phase][name] = dataset

        self._blended_datasets = {
            phase: (
                list(datasets.values())[0]
                if len(datasets) == 1
                else BlendedDataset(
                    list(datasets.values()),
                    weights=[self._dataset_weights[name] for name in datasets],
                    name=phase.value,
                    num_samples=self._samples_per_phase[phase],
                    cache_dir=self._cache_dir,
                    group=self._distributed.world_group,
                    verbose=run.is_main_rank,
                    data_sample_warn_time_ms=self._config.data_sample_warn_time_ms,
                )
            )
            for phase, datasets in self._sampled_datasets.items()
        }

    def get_iterator(
        self,
        batch_config: BatchConfig,
        phase: PhaseType,
        *,
        consumed_samples: int,
        num_workers: int,
        prefetch_factor: int | None = None,
    ):
        # TODO: Adjust or reimplement.
        return super().get_iterator(
            batch_config,
            phase,
            consumed_samples=consumed_samples,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    def _build_and_sample_stardoc_dataset(self, name: str, dataset_samples_per_phase: dict[PhaseType, int]):
        sampled_datasets = {}
        for phase, num_samples in dataset_samples_per_phase.items():
            if num_samples == 0:
                continue

            # TODO: Get image handling parameters from config 
            sampled_datasets[phase] = StarDocDataset(
                self._dataset_prefixes[name],
                im_size=224,
                num_samples=num_samples,
                num_im_tokens=256,
                transforms=False,
                multi_imgs=True,
                split=phase,
                tokenizer=self._tokenizer,
                config=self._config,
            )

        return sampled_datasets