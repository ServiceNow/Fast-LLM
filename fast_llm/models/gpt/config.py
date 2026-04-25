import logging
import typing

from fast_llm.config import Field, FieldHint, FieldOverride, config_class
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, PretrainedFastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.language_model.config import LanguageModelConfig
from fast_llm.models.gpt.conversion.config import (
    Apriel2TextCheckpointFormat,
    AprielHybridSSMCheckpointFormat,
    AutoGPTHuggingfaceCheckpointFormat,
    DiffusionDreamCheckpointFormat,
    DiffusionLlamaCheckpointFormat,
    Gemma4CheckpointFormat,
    LlamaCheckpointFormat,
    MistralCheckpointFormat,
    MixtralCheckpointFormat,
    MTPLlamaCheckpointFormat,
    Qwen2CheckpointFormat,
)
from fast_llm.models.gpt.megatron import set_megatron_distributed_seeds
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
    from fast_llm.models.gpt.model import GPTBaseModel, GPTInferenceRunner, GPTModel
    from fast_llm.models.gpt.trainer import GPTTrainer

logger = logging.getLogger(__name__)


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
    """Configuration for the GPT model, including distributed, multi-stage, and HuggingFace checkpoint formats."""

    _abstract = False
    model_name: typing.ClassVar[str] = "gpt"
    base_model: GPTBaseModelConfig = FieldOverride()
    checkpoint_formats: typing.ClassVar[tuple[type[CheckpointFormat], ...]] = FastLLMModelConfig.checkpoint_formats + (
        AutoGPTHuggingfaceCheckpointFormat,
        LlamaCheckpointFormat,
        Qwen2CheckpointFormat,
        MistralCheckpointFormat,
        MixtralCheckpointFormat,
        Gemma4CheckpointFormat,
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
    """Configuration for a GPT model together with an optional pretrained checkpoint to load."""

    _abstract = False
    model: GPTModelConfig = FieldOverride()


@config_class(dynamic_type={RunnableConfig: "train_gpt", TrainerConfig: "gpt"})
class GPTTrainerConfig(PretrainedGPTModelConfig, TrainerConfig):
    """Top-level configuration for training a GPT model. Entry point for `fast-llm train gpt`."""

    data: GPTDataConfig = FieldOverride()
    # TODO: Use dynamic model type?
    reference_models: dict[str, PretrainedGPTModelConfig] = FieldOverride()

    def _validate(self) -> None:
        if self.model.base_model.use_megatron_initialization:
            set_megatron_distributed_seeds(self.model.distributed)
        super()._validate()

        if self.model.base_model.embeddings.position_embeddings.enabled:
            Assert.geq(self.model.base_model.embeddings.num_position_embeddings, self.data.maximum_document_length)

        # TODO: Avoid digging inside the model.
        Assert.eq(self.reference_models.keys(), self.model.base_model.get_reference_models())

        for reference_model in self.reference_models.values():
            Assert.geq(
                reference_model.model.base_model.head.prediction_heads,
                self.model.base_model.head.prediction_heads,
            )
            Assert.empty(reference_model.model.base_model.get_reference_models())
            Assert.eq(
                reference_model.model.base_model.embeddings.vocab_parallel,
                self.model.base_model.embeddings.vocab_parallel,
            )

    @classmethod
    def get_trainer_class(cls) -> type["GPTTrainer"]:
        from fast_llm.models.gpt.trainer import GPTTrainer

        return GPTTrainer
