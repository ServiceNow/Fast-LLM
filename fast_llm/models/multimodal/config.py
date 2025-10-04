import logging
import typing

from fast_llm.config import Field, FieldHint, FieldUpdate, config_class
from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.layers.vision.config import VisionEncoderConfig
from fast_llm.models.gpt.config import (
    GPTBaseModelConfig,
    GPTBatchConfig,
    GPTModelConfig,
    GPTTrainerConfig,
    PretrainedGPTModelConfig,
)

if typing.TYPE_CHECKING:
    from fast_llm.models.multimodal.huggingface import HuggingfaceMultiModalModelForCausalLM
    from fast_llm.models.multimodal.model import MultiModalBaseModel, MultiModalModel, MultiModalModelInferenceRunner
    from fast_llm.models.multimodal.trainer import MultiModalTrainer

logger = logging.getLogger(__name__)


@config_class()
class MultiModalBatchConfig(GPTBatchConfig):
    pass


@config_class()
class MultiModalBaseModelConfig(GPTBaseModelConfig):
    vision_encoder: VisionEncoderConfig = Field(
        hint=FieldHint.architecture,
        desc="Configuration for the vision encoder.",
    )

    @property
    def base_model_class(self) -> type["MultiModalBaseModel"]:
        from fast_llm.models.multimodal.model import MultiModalBaseModel

        return MultiModalBaseModel


@config_class(dynamic_type={FastLLMModelConfig: "gpt"})
class MultiModalModelConfig(GPTModelConfig):
    _abstract = False
    model_name: typing.ClassVar[str] = "gpt"
    base_model: GPTBaseModelConfig = FieldUpdate()
    # TODO: ====== Conversion ======
    checkpoint_formats: typing.ClassVar[tuple[type[CheckpointFormat], ...]] = FastLLMModelConfig.checkpoint_formats

    @classmethod
    def get_model_class(cls) -> type["MultiModalModel"]:
        from fast_llm.models.multimodal.model import MultiModalModel

        return MultiModalModel

    @classmethod
    def get_inference_runner_class(cls) -> type["MultiModalModelInferenceRunner"]:
        from fast_llm.models.multimodal.model import MultiModalModelInferenceRunner

        return MultiModalModelInferenceRunner

    @classmethod
    def get_huggingface_model_for_causal_lm_class(cls) -> type["HuggingfaceMultiModalModelForCausalLM"]:
        from fast_llm.models.multimodal.huggingface import HuggingfaceMultiModalModelForCausalLM

        return HuggingfaceMultiModalModelForCausalLM


@config_class()
class PretrainedMultiModalModelConfig(PretrainedGPTModelConfig):
    _abstract = False
    model: MultiModalModelConfig = FieldUpdate()


@config_class(dynamic_type={RunnableConfig: "train_gpt", TrainerConfig: "gpt"})
class MultiModalTrainerConfig(PretrainedMultiModalModelConfig, GPTTrainerConfig):
    data: MultiModalDataConfig = FieldUpdate()
    batch: MultiModalBatchConfig = FieldUpdate()
    # TODO: Use dynamic model type?
    reference_models: dict[str, PretrainedMultiModalModelConfig] = FieldUpdate()

    @classmethod
    def get_trainer_class(cls) -> type["MultiModalTrainer"]:
        from fast_llm.models.multimodal.trainer import MultiModalTrainer

        return MultiModalTrainer
