"""Component converters for Apriel2 model surgery.

This module provides a registry of converters for transforming model components
(mixers, MLPs, normalizations) between different types. Each converter takes
source weights and configs and produces target weights.

Converter paths:
- Identity: forall a. a -> a
- Attention family: attention <-> sliding_window (bidirectional)
- One-way: attention -> mamba (random init, no inverse)

When no converter is registered for a (source, target) pair, random initialization
is required.
"""

import logging
from typing import Callable, Protocol

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# =============================================================================
# Converter Protocol
# =============================================================================


class ComponentConverter(Protocol):
    """Protocol for component converters.

    A converter takes source weights and configs and produces target weights.
    The weights dict uses relative keys (e.g., "self_attn.q_proj.weight").
    """

    def __call__(
        self,
        source_weights: dict[str, Tensor],
        source_config: dict,
        target_config: dict,
        hidden_size: int,
    ) -> dict[str, Tensor]:
        """Convert source weights to target format.

        Args:
            source_weights: Source component weights with relative keys.
            source_config: Source component configuration.
            target_config: Target component configuration.
            hidden_size: Model hidden size (for initialization).

        Returns:
            Target component weights with relative keys.
        """
        ...


# =============================================================================
# Converter Registry
# =============================================================================

# Registry: (source_type, target_type) -> converter function
_CONVERTERS: dict[tuple[str, str], ComponentConverter] = {}


def register_converter(source_type: str, target_type: str):
    """Decorator to register a converter for a (source, target) type pair."""

    def decorator(fn: ComponentConverter) -> ComponentConverter:
        _CONVERTERS[(source_type, target_type)] = fn
        return fn

    return decorator


def get_converter(source_type: str, target_type: str) -> ComponentConverter | None:
    """Get converter for (source, target) pair.

    Returns None if no converter is registered (caller must use random init).
    For same types, returns identity converter.
    """
    if source_type == target_type:
        return _identity_converter

    return _CONVERTERS.get((source_type, target_type))


def has_converter(source_type: str, target_type: str) -> bool:
    """Check if a converter exists for the given type pair."""
    return source_type == target_type or (source_type, target_type) in _CONVERTERS


def list_converters() -> list[tuple[str, str]]:
    """List all registered converter pairs."""
    return list(_CONVERTERS.keys())


# =============================================================================
# Identity Converter
# =============================================================================


def _identity_converter(
    source_weights: dict[str, Tensor],
    source_config: dict,
    target_config: dict,
    hidden_size: int,
) -> dict[str, Tensor]:
    """Identity converter - return source weights unchanged."""
    return {k: v.clone() for k, v in source_weights.items()}


# =============================================================================
# Attention Family Converters
# =============================================================================


@register_converter("attention", "sliding_window")
def _attention_to_sliding_window(
    source_weights: dict[str, Tensor],
    source_config: dict,
    target_config: dict,
    hidden_size: int,
) -> dict[str, Tensor]:
    """Convert attention to sliding window attention.

    These share the same architecture - sliding window just adds a window_size
    parameter that affects the attention mask, not the weights.
    """
    return {k: v.clone() for k, v in source_weights.items()}


@register_converter("sliding_window", "attention")
def _sliding_window_to_attention(
    source_weights: dict[str, Tensor],
    source_config: dict,
    target_config: dict,
    hidden_size: int,
) -> dict[str, Tensor]:
    """Convert sliding window attention back to full attention.

    Same weights, just removes the window constraint.
    """
    return {k: v.clone() for k, v in source_weights.items()}


# =============================================================================
# Random Initialization
# =============================================================================


def random_init_mixer(
    target_config: dict,
    hidden_size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """Initialize mixer weights randomly based on config.

    Uses the actual model classes to ensure correct initialization.
    """
    mixer_type = target_config.get("type", "attention")

    if mixer_type == "attention" or mixer_type == "sliding_window":
        return _init_attention_weights(target_config, hidden_size, device, dtype)
    elif mixer_type == "mamba":
        return _init_mamba_weights(target_config, hidden_size, device, dtype)
    elif mixer_type == "gated_delta_net":
        return _init_gated_delta_net_weights(target_config, hidden_size, device, dtype)
    else:
        raise ValueError(f"Unknown mixer type for random init: {mixer_type}")


def _init_attention_weights(
    config: dict,
    hidden_size: int,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Initialize attention weights."""
    heads = config.get("heads", 32)
    head_groups = config.get("head_groups", heads)
    head_size = config.get("head_size", hidden_size // heads)

    q_size = heads * head_size
    kv_size = head_groups * head_size

    weights = {}

    # Q, K, V, O projections
    weights["self_attn.q_proj.weight"] = _kaiming_init((q_size, hidden_size), device, dtype)
    weights["self_attn.k_proj.weight"] = _kaiming_init((kv_size, hidden_size), device, dtype)
    weights["self_attn.v_proj.weight"] = _kaiming_init((kv_size, hidden_size), device, dtype)
    weights["self_attn.o_proj.weight"] = _kaiming_init((hidden_size, q_size), device, dtype)

    # Add biases if configured
    if config.get("add_linear_biases", False):
        weights["self_attn.q_proj.bias"] = torch.zeros(q_size, device=device, dtype=dtype)
        weights["self_attn.k_proj.bias"] = torch.zeros(kv_size, device=device, dtype=dtype)
        weights["self_attn.v_proj.bias"] = torch.zeros(kv_size, device=device, dtype=dtype)
        weights["self_attn.o_proj.bias"] = torch.zeros(hidden_size, device=device, dtype=dtype)

    return weights


def _init_mamba_weights(
    config: dict,
    hidden_size: int,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Initialize Mamba (SSM) weights.

    Uses standard Mamba initialization conventions.
    """
    # Mamba hyperparameters
    d_state = config.get("d_state", 16)
    d_conv = config.get("d_conv", 4)
    expand = config.get("expand", 2)
    d_inner = int(expand * hidden_size)
    dt_rank = config.get("dt_rank", "auto")
    if dt_rank == "auto":
        dt_rank = max(1, hidden_size // 16)

    weights = {}

    # Input projection (hidden_size -> 2 * d_inner for x and z)
    weights["in_proj.weight"] = _kaiming_init((2 * d_inner, hidden_size), device, dtype)

    # Conv1d
    weights["conv1d.weight"] = _kaiming_init((d_inner, 1, d_conv), device, dtype)
    if config.get("conv_bias", True):
        weights["conv1d.bias"] = torch.zeros(d_inner, device=device, dtype=dtype)

    # SSM parameters
    weights["x_proj.weight"] = _kaiming_init((dt_rank + d_state * 2, d_inner), device, dtype)
    weights["dt_proj.weight"] = _kaiming_init((d_inner, dt_rank), device, dtype)
    if config.get("dt_proj_bias", True):
        # Initialize dt_proj bias with inverse softplus of dt_init
        dt_init = config.get("dt_init", 0.001)
        dt_bias = torch.ones(d_inner, device=device, dtype=dtype) * (
            dt_init + torch.log(torch.expm1(torch.tensor(dt_init))).item()
        )
        weights["dt_proj.bias"] = dt_bias

    # A is typically initialized as -exp(linspace(...))
    A = torch.arange(1, d_state + 1, device=device, dtype=dtype).unsqueeze(0).expand(d_inner, -1)
    weights["A_log"] = torch.log(A)

    # D is initialized to ones
    weights["D"] = torch.ones(d_inner, device=device, dtype=dtype)

    # Output projection
    weights["out_proj.weight"] = _kaiming_init((hidden_size, d_inner), device, dtype)

    return weights


def _init_gated_delta_net_weights(
    config: dict,
    hidden_size: int,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Initialize Gated Delta Net weights."""
    heads = config.get("heads", 32)
    head_size = config.get("head_size", hidden_size // heads)

    weights = {}

    # Similar structure to attention but with gating
    q_size = heads * head_size
    weights["q_proj.weight"] = _kaiming_init((q_size, hidden_size), device, dtype)
    weights["k_proj.weight"] = _kaiming_init((q_size, hidden_size), device, dtype)
    weights["v_proj.weight"] = _kaiming_init((q_size, hidden_size), device, dtype)
    weights["o_proj.weight"] = _kaiming_init((hidden_size, q_size), device, dtype)

    # Gate projections
    weights["beta_proj.weight"] = _kaiming_init((heads, hidden_size), device, dtype)

    return weights


def random_init_mlp(
    target_config: dict,
    hidden_size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """Initialize MLP weights randomly."""
    intermediate_size = target_config.get("intermediate_size", hidden_size * 4)
    gated = target_config.get("gated", True)
    add_bias = target_config.get("add_linear_biases", False)

    weights = {}

    if gated:
        weights["gate_proj.weight"] = _kaiming_init(
            (intermediate_size, hidden_size), device, dtype
        )
        weights["up_proj.weight"] = _kaiming_init(
            (intermediate_size, hidden_size), device, dtype
        )
    else:
        weights["up_proj.weight"] = _kaiming_init(
            (intermediate_size, hidden_size), device, dtype
        )

    weights["down_proj.weight"] = _kaiming_init(
        (hidden_size, intermediate_size), device, dtype
    )

    if add_bias:
        if gated:
            weights["gate_proj.bias"] = torch.zeros(intermediate_size, device=device, dtype=dtype)
        weights["up_proj.bias"] = torch.zeros(intermediate_size, device=device, dtype=dtype)
        weights["down_proj.bias"] = torch.zeros(hidden_size, device=device, dtype=dtype)

    return weights


def random_init_norm(
    target_config: dict,
    hidden_size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """Initialize normalization weights."""
    norm_type = target_config.get("type", "rms_norm")

    if norm_type == "rms_norm":
        return {"weight": torch.ones(hidden_size, device=device, dtype=dtype)}
    elif norm_type == "layer_norm":
        return {
            "weight": torch.ones(hidden_size, device=device, dtype=dtype),
            "bias": torch.zeros(hidden_size, device=device, dtype=dtype),
        }
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")


def _kaiming_init(
    shape: tuple[int, ...],
    device: str,
    dtype: torch.dtype,
) -> Tensor:
    """Kaiming uniform initialization."""
    tensor = torch.empty(shape, device=device, dtype=dtype)
    torch.nn.init.kaiming_uniform_(tensor, a=5**0.5)
    return tensor


# =============================================================================
# Utility Functions
# =============================================================================


def get_mixer_type(mixer_config: dict) -> str:
    """Get the effective mixer type from config.

    Handles both direct mixer configs and stochastic wrapper configs.
    For stochastic mixers, returns 'stochastic'.
    """
    return mixer_config.get("type", "attention")


def get_main_mixer_config(mixer_config: dict) -> dict:
    """Get the main mixer config, unwrapping stochastic if needed.

    For stochastic mixers, returns the config of the main mixer.
    For regular mixers, returns the config itself.
    """
    if mixer_config.get("type") == "stochastic":
        main_name = mixer_config.get("main_mixer_name", "attention")
        return mixer_config.get("mixers", {}).get(main_name, {})
    return mixer_config


def get_main_mixer_type(mixer_config: dict) -> str:
    """Get the type of the main mixer, unwrapping stochastic if needed."""
    main_config = get_main_mixer_config(mixer_config)
    return main_config.get("type", "attention")
