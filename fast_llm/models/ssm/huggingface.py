import logging

from fast_llm.engine.huggingface.config import HuggingfaceModelConfig
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
from fast_llm.models.ssm.config import HybridModelConfig
from fast_llm.models.ssm.model import HybridModel

logger = logging.getLogger(__name__)


class HuggingfaceSSMModelConfig(HuggingfaceModelConfig):
    model_type = "fast_llm_ssm"
    model_config_class = HybridModelConfig
    fast_llm_config: HybridModelConfig


class HuggingfaceSSMModelForCausalLM(HuggingfaceGPTModelForCausalLM):
    config_class = HuggingfaceSSMModelConfig
    config: HuggingfaceSSMModelConfig
    model_class = HybridModel
    _fast_llm_model: HybridModel
