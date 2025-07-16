import logging
import typing

from fast_llm.engine.inference.config import HuggingfaceModelConfig
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
from fast_llm.models.ssm.config import HybridSSMModelConfig
from fast_llm.models.ssm.model import HybridSSMInferenceRunner, HybridSSMModel

logger = logging.getLogger(__name__)


class HuggingfaceSSMModelConfig(HuggingfaceModelConfig):
    model_type = "fast_llm_ssm"
    model_config_class = HybridSSMModelConfig
    fast_llm_config: HybridSSMModelConfig


class HuggingfaceHybridSSMModelForCausalLM(HuggingfaceGPTModelForCausalLM):
    config_class = HuggingfaceSSMModelConfig
    config: HuggingfaceSSMModelConfig
    model_class = HybridSSMModel
    runner_class: typing.ClassVar[type[HybridSSMInferenceRunner]] = HybridSSMInferenceRunner
    _fast_llm_model: HybridSSMModel
