from fast_llm.config import Field, FieldHint, config_class
from fast_llm.data.config import DataConfig
from fast_llm.models.gpt.config import (
    GPTArchitectureConfig,
    GPTBaseModelConfig,
    GPTModelConfig,
    GPTTrainerConfig,
    PretrainedGPTModelConfig,
)


@config_class()
class CustomDataConfig(DataConfig):
    # TODO: If needed, inherit from AbstractDataConfig instead and re-implement everything.
    pass


@config_class()
class CustomArchitectureConfig(GPTArchitectureConfig):
    # TODO: Add custom base model architecture config parameters, if any.
    pass


@config_class()
class CustomBaseModelConfig(GPTBaseModelConfig, CustomArchitectureConfig):
    # TODO: Add custom other base model config parameters, if any.
    architecture_cls = CustomArchitectureConfig


@config_class()
class CustomModelConfig(GPTModelConfig):
    # TODO: Add custom model config parameters, if any (typically none).
    base_model: CustomBaseModelConfig = Field(default_factory=CustomBaseModelConfig)

    @classmethod
    def get_model_class(cls):
        from fast_llm.models.custom.model import CustomModel

        return CustomModel

    @classmethod
    def get_huggingface_model_class(cls):
        from fast_llm.models.custom.huggingface import HuggingfaceCustomModelForCausalLM

        return HuggingfaceCustomModelForCausalLM


@config_class()
class PretrainedCustomModelConfig(PretrainedGPTModelConfig):
    model: CustomModelConfig = Field(default_factory=CustomModelConfig)


@config_class()
class CustomTrainerConfig(PretrainedCustomModelConfig, GPTTrainerConfig):
    # TODO: Add custom trainer config parameters, if any (typically none).

    data: CustomDataConfig = Field(
        default_factory=CustomDataConfig,
        desc="Configuration for the dataset and model-independent preprocessing.",
        hint=FieldHint.core,
    )

    @classmethod
    def get_trainer_class(cls):
        from fast_llm.models.custom.trainer import CustomTrainer

        return CustomTrainer
