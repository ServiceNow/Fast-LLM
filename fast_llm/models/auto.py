from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.models.custom.config import CustomModelConfig, CustomTrainerConfig
from fast_llm.models.gpt.config import GPTModelConfig, GPTTrainerConfig
from fast_llm.models.ssm.config import HybridSSMModelConfig, HybridTrainerConfig
from fast_llm.utils import Registry

model_registry = Registry[str, FastLLMModelConfig](
    "Model",
    {
        model.model_name: model
        for model in [
            GPTModelConfig,
            CustomModelConfig,
            HybridSSMModelConfig,
        ]
    },
)

trainer_registry = Registry[str, TrainerConfig](
    "Model",
    {
        trainer.get_field("model").type.model_name: trainer
        for trainer in [
            GPTTrainerConfig,
            CustomTrainerConfig,
            HybridTrainerConfig,
        ]
    },
)
