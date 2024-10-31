from fast_llm.models.custom.config import CustomModelConfig, CustomTrainerConfig
from fast_llm.models.gpt.config import GPTModelConfig, GPTTrainerConfig
from fast_llm.utils import Registry

model_registry = Registry(
    "Model",
    {
        model.model_name: model
        for model in [
            GPTModelConfig,
            CustomModelConfig,
        ]
    },
)

trainer_registry = Registry(
    "Model",
    {
        trainer.get_field("model").type.model_name: trainer
        for trainer in [
            GPTTrainerConfig,
            CustomTrainerConfig,
        ]
    },
)
