from fast_llm.models.grpo.config import GRPOModelConfig, GRPOTrainerConfig
from fast_llm.models.gpt.config import GPTModelConfig, GPTTrainerConfig
from fast_llm.utils import Registry

model_registry = Registry(
    "Model",
    {
        "gpt": GPTModelConfig,
        "grpo": GRPOModelConfig,
    },
)

trainer_registry = Registry(
    "Model",
    {
        "gpt": GPTTrainerConfig,
        "grpo": GRPOTrainerConfig,
    },
)
