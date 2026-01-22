"""vLLM model implementation for Apriel2.

This module provides vLLM-optimized implementations of Apriel2 models.
See README.md for usage instructions.

Placement switching (for stochastic mixer models):
    placements = llm.collective_rpc("get_layer_placements")
    llm.collective_rpc("set_layer_placements", args=(new_placement,))

Plugin usage (for vLLM subprocess registration):
    Set VLLM_PLUGINS=fast_llm_external_models.apriel2.vllm.config_convertor
    before starting vLLM to ensure registration in subprocesses.
"""

from fast_llm_external_models.apriel2.vllm.modeling_apriel2 import Apriel2ForCausalLM

__all__ = ["Apriel2ForCausalLM"]
