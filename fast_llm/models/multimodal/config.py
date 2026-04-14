import logging
import typing

from fast_llm.config import FieldOverride, config_class
from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.layers.audio_encoder.config import AudioMultiModalModelConfig
from fast_llm.layers.vision.config import VisionMultiModalModelConfig
from fast_llm.models.gpt.config import (
    GPTBaseModelConfig,
    GPTModelConfig,
    GPTTrainerConfig,
    PretrainedGPTModelConfig,
)
from fast_llm.models.multimodal.conversion.config import (
    Apriel2CheckpointFormat,
    AyraCheckpointFormat,
    LlavaCheckpointFormat,
    LlavaHybridSSMCheckpointFormat,
    WhisperCheckpointFormat,
)

if typing.TYPE_CHECKING:
    from fast_llm.models.multimodal.huggingface import HuggingfaceMultiModalModelForCausalLM
    from fast_llm.models.multimodal.model import MultiModalBaseModel, MultiModalInferenceRunner, MultiModalModel
    from fast_llm.models.multimodal.trainer import MultiModalTrainer

logger = logging.getLogger(__name__)


@config_class()
class MultiModalBaseModelConfig(VisionMultiModalModelConfig, AudioMultiModalModelConfig, GPTBaseModelConfig):
    def _validate(self) -> None:
        super()._validate()
        if self.vision_encoder is None and not self.audio_encoder.enabled:
            raise ValueError(
                "MultiModalBaseModelConfig requires at least one encoder to be enabled. "
                "Set model.base_model.vision_encoder or model.base_model.audio_encoder.encoder_type."
            )

    @property
    def base_model_class(self) -> type["MultiModalBaseModel"]:
        from fast_llm.models.multimodal.model import MultiModalBaseModel

        return MultiModalBaseModel


@config_class(dynamic_type={FastLLMModelConfig: "multimodal"})
class MultiModalModelConfig(GPTModelConfig):
    _abstract = False
    model_name: typing.ClassVar[str] = "multimodal"
    base_model: MultiModalBaseModelConfig = FieldOverride()
    checkpoint_formats: typing.ClassVar[tuple[type[CheckpointFormat], ...]] = FastLLMModelConfig.checkpoint_formats + (
        LlavaCheckpointFormat,
        LlavaHybridSSMCheckpointFormat,
        Apriel2CheckpointFormat,
        WhisperCheckpointFormat,
        AyraCheckpointFormat,
    )

    @classmethod
    def get_model_class(cls) -> type["MultiModalModel"]:
        from fast_llm.models.multimodal.model import MultiModalModel

        return MultiModalModel

    @classmethod
    def get_inference_runner_class(cls) -> type["MultiModalInferenceRunner"]:
        from fast_llm.models.multimodal.model import MultiModalInferenceRunner

        return MultiModalInferenceRunner

    @classmethod
    def get_huggingface_model_for_causal_lm_class(cls) -> type["HuggingfaceMultiModalModelForCausalLM"]:
        from fast_llm.models.multimodal.huggingface import HuggingfaceMultiModalModelForCausalLM

        return HuggingfaceMultiModalModelForCausalLM


@config_class()
class PretrainedMultiModalModelConfig(PretrainedGPTModelConfig):
    _abstract = False
    model: MultiModalModelConfig = FieldOverride()


@config_class(dynamic_type={RunnableConfig: "train_multimodal", TrainerConfig: "multimodal"})
class MultiModalTrainerConfig(PretrainedMultiModalModelConfig, GPTTrainerConfig):
    # TODO: Use dynamic model type?
    reference_models: dict[str, PretrainedMultiModalModelConfig] = FieldOverride()

    @classmethod
    def get_trainer_class(cls) -> type["MultiModalTrainer"]:
        from fast_llm.models.multimodal.trainer import MultiModalTrainer

        return MultiModalTrainer
