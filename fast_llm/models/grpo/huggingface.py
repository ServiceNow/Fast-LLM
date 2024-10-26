from fast_llm.models.grpo.config import GRPOModelConfig
from fast_llm.models.grpo.model import GRPOModel
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelConfig, HuggingfaceGPTModelForCausalLM


class HuggingfaceCustomModelConfig(HuggingfaceGPTModelConfig):
    model_type = "fast_llm_gpt_custom"
    model_config_class = GRPOModelConfig
    fast_llm_config: GRPOModelConfig


class HuggingfaceCustomModelForCausalLM(HuggingfaceGPTModelForCausalLM):
    # TODO: Implement changes in huggingface interface, if any.
    #   Ex.: Return predictions instead of logits.
    config_class = HuggingfaceCustomModelConfig
    config: HuggingfaceCustomModelConfig
    model_class = GRPOModel
    _fast_llm_model: GRPOModel
