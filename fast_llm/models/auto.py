from fast_llm.utils import LazyRegistry


def _get_gpt_model():
    from fast_llm.models.gpt.model import GPTModel

    return GPTModel


def _get_gpt_trainer():
    from fast_llm.models.gpt.trainer import GPTTrainer

    return GPTTrainer


def _get_gpt_huggingface():
    from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM

    return HuggingfaceGPTModelForCausalLM


model_registry = LazyRegistry(
    "Model",
    {
        "gpt": _get_gpt_model,
    },
)

trainer_registry = LazyRegistry(
    "Trainer",
    {
        "gpt": _get_gpt_trainer,
    },
)

huggingface_model_registry = LazyRegistry(
    "Fast-LLM Huggingface Interface",
    {
        "gpt": _get_gpt_huggingface,
    },
)
