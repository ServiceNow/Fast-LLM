import typing

from fast_llm.config import FieldUpdate, config_class
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
    architecture_class = CustomArchitectureConfig


@config_class()
class CustomModelConfig(GPTModelConfig):
    # TODO: Add custom model config parameters, if any (typically none).
    model_name: typing.ClassVar[str] = "gpt_custom"
    base_model: CustomBaseModelConfig = FieldUpdate(default_factory=CustomBaseModelConfig)

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
    model: CustomModelConfig = FieldUpdate(default_factory=CustomModelConfig)


@config_class()
class CustomTrainerConfig(PretrainedCustomModelConfig, GPTTrainerConfig):
    # TODO: Add custom trainer config parameters, if any (typically none).

    data: CustomDataConfig = FieldUpdate(default_factory=CustomDataConfig)

    @classmethod
    def get_trainer_class(cls):
        from fast_llm.models.custom.trainer import CustomTrainer

        return CustomTrainer
