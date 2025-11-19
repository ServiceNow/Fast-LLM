"""
Import these submodules to ensure classes are added to the dynamic class registry.
"""

from fast_llm.layers.ssm.config import (  # isort: skip
    DiscreteMamba2Config,
    GatedDeltaNetConfig,
    Mamba2Config,
    MambaConfig,
)

from fast_llm.layers.attention.config import AttentionConfig  # isort: skip
from fast_llm.models.gpt.config import GPTModelConfig, GPTTrainerConfig  # isort: skip
from fast_llm.engine.evaluation.evaluators import EvaluatorsConfig  # isort: skip
