import functools
import logging
import typing

from fast_llm.config import Field, FieldHint, FieldUpdate, check_field, config_class
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, PretrainedFastLLMModelConfig
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.language_model.config import LanguageModelConfig, MultiTokenPredictionConfig
from fast_llm.models.gpt.conversion.config import (
    Apriel2TextCheckpointFormat,
    AprielHybridSSMCheckpointFormat,
    AutoGPTHuggingfaceCheckpointFormat,
    DiffusionDreamCheckpointFormat,
    DiffusionLlamaCheckpointFormat,
    LlamaCheckpointFormat,
    MistralCheckpointFormat,
    MixtralCheckpointFormat,
    MTPLlamaCheckpointFormat,
    Qwen2CheckpointFormat,
)
from fast_llm.models.gpt.megatron import set_megatron_distributed_seeds
from fast_llm.utils import Assert, div

if typing.TYPE_CHECKING:
    from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
    from fast_llm.models.gpt.model import GPTBaseModel, GPTInferenceRunner, GPTModel
    from fast_llm.models.gpt.trainer import GPTTrainer

logger = logging.getLogger(__name__)


@config_class()
class GPTBatchConfig(BatchConfig):
    sequence_length: int = Field(
        default=2048,
        desc="Number of tokens in a sample.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    micro_sequence_length: int = Field(
        default=None,
        desc="Number of tokens in a micro-sequence (must divide the sequence length).",
        hint=FieldHint.performance,
        valid=check_field(Assert.gt, 0),
    )
    use_loss_masking_spans: bool = Field(
        default=False,
        desc="Read loss masking spans from the dataset.",
        hint=FieldHint.feature,
    )
    truncate_documents: bool | None = Field(
        default=True,
        desc=(
            "If enabled, documents may be truncated while being packed to fit the sequence length."
            "Otherwise, sequences will be padded such that every document lies entirely within a sample"
            " (and documents exceeding the sequence length will be skipped altogether)."
        ),
        hint=FieldHint.feature,
    )

    def _validate(self) -> None:
        if self.micro_sequence_length is None:
            with self._set_implicit_default():
                self.micro_sequence_length = self.sequence_length
        super()._validate()

    @functools.cached_property
    def micro_batch_splits(self) -> int:
        assert self._validated
        return div(self.sequence_length, self.micro_sequence_length)


@config_class()
class GPTBaseModelConfig(LanguageModelConfig, BaseModelConfig):
    _abstract = False

    # TODO: Allow overriding in sub-models?
    peft: PeftConfig = Field(
        desc="Configuration for parameter-efficient fine tuning.",
        hint=FieldHint.architecture,
    )
    # Debug, to get an exact match with megatron init.
    use_megatron_initialization: bool = Field(
        default=False, desc="Exactly match the initialization of a Megatron model.", hint=FieldHint.testing
    )

    @property
    def base_model_class(self) -> type["GPTBaseModel"]:
        from fast_llm.models.gpt.model import GPTBaseModel

        return GPTBaseModel


@config_class(dynamic_type={FastLLMModelConfig: "gpt"})
class GPTModelConfig(FastLLMModelConfig):
    _abstract = False
    model_name: typing.ClassVar[str] = "gpt"
    base_model: GPTBaseModelConfig = FieldUpdate()
    checkpoint_formats: typing.ClassVar[tuple[type[CheckpointFormat], ...]] = FastLLMModelConfig.checkpoint_formats + (
        AutoGPTHuggingfaceCheckpointFormat,
        LlamaCheckpointFormat,
        Qwen2CheckpointFormat,
        MistralCheckpointFormat,
        MixtralCheckpointFormat,
        MTPLlamaCheckpointFormat,
        DiffusionDreamCheckpointFormat,
        DiffusionLlamaCheckpointFormat,
        AprielHybridSSMCheckpointFormat,
        Apriel2TextCheckpointFormat,
    )

    @classmethod
    def get_model_class(cls) -> type["GPTModel"]:
        from fast_llm.models.gpt.model import GPTModel

        return GPTModel

    @classmethod
    def get_inference_runner_class(cls) -> type["GPTInferenceRunner"]:
        from fast_llm.models.gpt.model import GPTInferenceRunner

        return GPTInferenceRunner

    @classmethod
    def get_huggingface_model_for_causal_lm_class(cls) -> type["HuggingfaceGPTModelForCausalLM"]:
        from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM

        return HuggingfaceGPTModelForCausalLM


@config_class()
class PretrainedGPTModelConfig(PretrainedFastLLMModelConfig):
    _abstract = False
    model: GPTModelConfig = FieldUpdate()


@config_class(dynamic_type={RunnableConfig: "train_gpt", TrainerConfig: "gpt"})
class GPTTrainerConfig(PretrainedGPTModelConfig, TrainerConfig):
    data: GPTDataConfig = FieldUpdate()
    batch: GPTBatchConfig = FieldUpdate()
    # TODO: Use dynamic model type?
    reference_models: dict[str, PretrainedGPTModelConfig] = FieldUpdate()

    def _validate(self) -> None:
        if self.batch.sequence_length is None:
            # TODO: Drop this.
            self.batch.sequence_length = self.model.base_model.embeddings.num_position_embeddings
        if self.model.base_model.use_megatron_initialization:
            set_megatron_distributed_seeds(self.model.distributed)
        super()._validate()

        if self.model.base_model.embeddings.position_embeddings.enabled:
            Assert.geq(self.model.base_model.embeddings.num_position_embeddings, self.batch.sequence_length)

        # TODO: Avoid digging inside the model.
        head = self.model.base_model.head
        if isinstance(head, MultiTokenPredictionConfig):
            prediction_heads = head.prediction_heads
            head = head.head
        else:
            prediction_heads = 1

        expected_names = {name for name in (head.distillation_model, head.dpo_reference_model) if name is not None}
        Assert.eq(self.reference_models.keys(), expected_names)

        for reference_model in self.reference_models.values():
            reference_head = reference_model.model.base_model.head
            if isinstance(reference_head, MultiTokenPredictionConfig):
                reference_prediction_heads = reference_head.prediction_heads
                reference_head = reference_head.heads
            else:
                reference_prediction_heads = 1
            Assert.geq(reference_prediction_heads, prediction_heads)

            Assert.none(reference_head.distillation_model)
            Assert.none(reference_head.dpo_reference_model)
            # TODO: Support more LM head features.
            Assert.none(reference_head.cross_entropy_splits)
            Assert.eq(
                reference_model.model.base_model.embeddings.vocab_parallel,
                self.model.base_model.embeddings.vocab_parallel,
            )

    @classmethod
    def get_trainer_class(cls) -> type["GPTTrainer"]:
        from fast_llm.models.gpt.trainer import GPTTrainer

        return GPTTrainer
