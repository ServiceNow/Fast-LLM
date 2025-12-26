"""Qwen2/Qwen2.5 to Apriel2 conversion module."""

from fast_llm_external_models.apriel2.conversion.qwen2.config import convert_config
from fast_llm_external_models.apriel2.conversion.qwen2.plan import plan_qwen2_to_apriel2

__all__ = ["convert_config", "plan_qwen2_to_apriel2"]
