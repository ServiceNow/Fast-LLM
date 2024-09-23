from fast_llm.models.gpt.config import GPTModelConfig, GPTTrainerConfig
from fast_llm.utils import Registry

model_registry = Registry(
    "Model",
    {
        "gpt": GPTModelConfig,
    },
)

trainer_registry = Registry(
    "Model",
    {
        "gpt": GPTTrainerConfig,
    },
)
