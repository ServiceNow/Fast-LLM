import typing

from fast_llm.config import FieldUpdate, config_class
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig, GPTTrainerConfig, PretrainedGPTModelConfig

if typing.TYPE_CHECKING:
    from fast_llm.models.custom.huggingface import HuggingfaceCustomModelForCausalLM
    from fast_llm.models.custom.model import CustomModel
    from fast_llm.models.custom.trainer import CustomTrainer


@config_class()
class CustomDataConfig(GPTDataConfig):
    # TODO: If needed, inherit from AbstractDataConfig instead and re-implement everything.
    pass


@config_class()
class CustomBaseModelConfig(GPTBaseModelConfig):
    # TODO: Add custom other base model config parameters, if any.
    pass


@config_class(dynamic_type={FastLLMModelConfig: "gpt_custom"})
class CustomModelConfig(GPTModelConfig):
    # TODO: Add custom model config parameters, if any (typically none).
    model_name: typing.ClassVar[str] = "gpt_custom"
    base_model: CustomBaseModelConfig = FieldUpdate()

    @classmethod
    def get_model_class(cls) -> type["CustomModel"]:
        from fast_llm.models.custom.model import CustomModel

        return CustomModel

    @classmethod
    def get_huggingface_model_for_causal_lm_class(cls) -> type["HuggingfaceCustomModelForCausalLM"]:
        from fast_llm.models.custom.huggingface import HuggingfaceCustomModelForCausalLM

        return HuggingfaceCustomModelForCausalLM


@config_class()
class PretrainedCustomModelConfig(PretrainedGPTModelConfig):
    model: CustomModelConfig = FieldUpdate()


@config_class(dynamic_type={RunnableConfig: "train_gpt_custom", TrainerConfig: "gpt_custom"})
class CustomTrainerConfig(PretrainedCustomModelConfig, GPTTrainerConfig):
    # TODO: Add custom trainer config parameters, if any (typically none).
    data: CustomDataConfig = FieldUpdate()
    reference_models: dict[str, PretrainedCustomModelConfig] = FieldUpdate()

    @classmethod
    def get_trainer_class(cls) -> type["CustomTrainer"]:
        from fast_llm.models.custom.trainer import CustomTrainer

        return CustomTrainer
