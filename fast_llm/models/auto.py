"""
Import these submodules to ensure classes are added to the dynamic class registry.
"""

from fast_llm.layers.ssm.config import MambaConfig, Mamba2Config, DiscreteMamba2Config  # isort: skip
from fast_llm.models.gpt.config import GPTModelConfig, GPTTrainerConfig  # isort: skip
from fast_llm.engine.evaluation.evaluators import EvaluatorsConfig  # isort: skip
