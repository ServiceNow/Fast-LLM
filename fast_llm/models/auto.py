"""
Import these submodules to ensure classes are added to the dynamic class registry.
"""

from fast_llm.models.custom.config import CustomModelConfig, CustomTrainerConfig  # isort: skip
from fast_llm.models.gpt.config import GPTModelConfig, GPTTrainerConfig  # isort: skip
from fast_llm.models.ssm.config import HybridSSMModelConfig, HybridSSMTrainerConfig  # isort: skip
