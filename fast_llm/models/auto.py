from fast_llm.models.custom.config import CustomModelConfig, CustomTrainerConfig
from fast_llm.models.gpt.config import GPTModelConfig, GPTTrainerConfig
from fast_llm.models.stardoc.config import StarDocModelConfig, StarDocTrainerConfig
from fast_llm.utils import Registry

model_registry = Registry(
    "Model",
    {
        "gpt": GPTModelConfig,
        "stardoc": StarDocModelConfig,
        "gpt_custom": CustomModelConfig,
    },
)

trainer_registry = Registry(
    "Model",
    {
        "gpt": GPTTrainerConfig,
        "stardoc": StarDocTrainerConfig,
        "gpt_custom": CustomTrainerConfig,
    },
)
