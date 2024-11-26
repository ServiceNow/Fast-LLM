from fast_llm.models.custom.config import CustomModelConfig
from fast_llm.models.custom.model import CustomModel
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelConfig, HuggingfaceGPTModelForCausalLM


class HuggingfaceCustomModelConfig(HuggingfaceGPTModelConfig):
    model_type = "fast_llm_gpt_custom"
    model_config_class = CustomModelConfig
    fast_llm_config: CustomModelConfig


class HuggingfaceCustomModelForCausalLM(HuggingfaceGPTModelForCausalLM):
    # TODO: Implement changes in huggingface interface, if any.
    #   Ex.: Return predictions instead of logits.
    config_class = HuggingfaceCustomModelConfig
    config: HuggingfaceCustomModelConfig
    model_class = CustomModel
    _fast_llm_model: CustomModel
