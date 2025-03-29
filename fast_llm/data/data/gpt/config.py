import logging
import typing

from fast_llm.config import Field, FieldHint, FieldUpdate, check_field, config_class
from fast_llm.data.config import MultiprocessingContext, TokenizerConfig
from fast_llm.data.data.config import DataConfig, SamplingDefaultConfig
from fast_llm.data.dataset.gpt.config import (
    GPTLegacyConfig,
    GPTLegacyDatasetConfig,
    GPTSampledDatasetConfig,
    GPTSamplingConfig,
    ShufflingType,
)
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


@config_class()
class GPTSamplingDefaultConfig(SamplingDefaultConfig, GPTSamplingConfig):
    gpu: bool = FieldUpdate(default=True)
    use_loss_masking_spans: bool = FieldUpdate(default=False)
    shuffle: ShufflingType = FieldUpdate(default=ShufflingType.epoch)


@config_class()
class GPTDataConfig(DataConfig, GPTLegacyConfig):
    """
    Configuration for the dataset(s), split and sampling.
    Currently hard-coded to a GPT dataset.
    TODO: Extract generalizable content.
    """

    _abstract = False

    tokenizer: TokenizerConfig = Field(
        default_factory=TokenizerConfig,
        desc="Configuration for the tokenizer (for FIM).",
        hint=FieldHint.feature,
    )
    # TODO: Review field. Move closer to phase definition in training config?
    datasets: dict[str, GPTSampledDatasetConfig] = Field(
        default_factory=dict,
        desc="Configuration for the dataset(s).",
        hint=FieldHint.core,
    )
    sampling: GPTSamplingDefaultConfig = FieldUpdate(default_factory=GPTSamplingDefaultConfig)
    data_sample_warn_time_ms: float = Field(
        default=1000,
        desc="Warn if a sample takes too long to load.",
        hint=FieldHint.feature,
        valid=check_field(Assert.gt, 0),
    )
    multiprocessing_context: MultiprocessingContext = Field(
        default=MultiprocessingContext.spawn,
        desc="Multiprocessing context. Do not touch.",
        hint=FieldHint.expert,
    )
    truncate_documents: bool = Field(
        default=True,
        desc=(
            "If enabled, documents may be truncated while being packed to fit the sequence length."
            "Otherwise, sequences will be padded such that every document lies entirely within a sample"
            " (and documents exceeding the sequence length will be skipped altogether)."
        ),
        hint=FieldHint.feature,
    )

    def _validate(self) -> None:
        if not self.datasets:
            logger.warning(
                "Using the legacy dataset definition format." " Specify it through `data.datasets` instead."
            )
            self.datasets = {
                phase.value.lower(): GPTLegacyDatasetConfig.from_dict(self, strict=False)
                for phase in (PhaseType.training, PhaseType.validation, PhaseType.test)
            }
        super()._validate()

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        # TODO v0.x: Remove backward compatibility.
        if "datasets" in default:
            for phase in PhaseType:
                if phase.value in default["datasets"]:
                    rename = phase.value.lower()
                    logger.warning(f"Renaming dataset {phase.value} to {rename}")
                    assert rename not in default["datasets"]
                    default["datasets"][rename] = default["datasets"].pop(phase.value)

        cls._handle_renamed_field(default, "validation", ("evaluations", "validation"))
        return super()._from_dict(default, strict, flat)
