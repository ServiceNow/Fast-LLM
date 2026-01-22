"""vLLM model implementation for Apriel2.

This module provides vLLM-optimized implementations of Apriel2 models.
See README.md for usage instructions.

Placement switching (for stochastic mixer models):
    placements = llm.collective_rpc("get_layer_placements")
    llm.collective_rpc("set_layer_placements", args=(new_placement,))
"""

from fast_llm_external_models.apriel2.vllm.modeling_apriel2 import Apriel2ForCausalLM


def register():
    """Register Apriel2 models with vLLM's ModelRegistry."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Apriel2ForCausalLM",
        "fast_llm_external_models.apriel2.vllm:Apriel2ForCausalLM",
    )


__all__ = ["Apriel2ForCausalLM", "register"]
