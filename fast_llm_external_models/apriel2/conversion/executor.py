"""Plan execution with streaming I/O."""

from __future__ import annotations

import hashlib
from typing import Callable, Iterator

import torch
from torch import Tensor

from fast_llm_external_models.apriel2.conversion.expr import ExprPlan, W

MAX_SEED = 2**31 - 1  # torch.Generator.manual_seed limit


class StreamingExecutor:
    """Execute a plan with streaming I/O.

    Sources are loaded on-demand via the source_loader callable.
    With memory-mapped safetensors, repeated loads are free (same data pointer).
    """

    def __init__(
        self,
        plan: ExprPlan,
        source_loader: Callable[[W], Tensor],
    ):
        self.plan = plan
        self.source_loader = source_loader

    def execute(
        self,
        seed: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Iterator[tuple[W, Tensor]]:
        """Execute the plan, yielding (target_key, tensor) pairs.

        Args:
            seed: Base seed for reproducibility. Each target gets a deterministic
                  seed derived from (seed + key_offset) % MAX_SEED.
            device: Device for tensors. If None, inferred from first source tensor.
            dtype: Dtype for tensors. If None, inferred from first source tensor.

        If the plan has no source dependencies (all Init), device/dtype must be provided.
        """
        # Infer device/dtype from first source if not provided
        if device is None or dtype is None:
            for expr in self.plan.mappings.values():
                refs = expr.find_refs()
                if refs:
                    first_tensor = self.source_loader(next(iter(refs)))
                    device, dtype = first_tensor.device, first_tensor.dtype
                    break
            else:
                raise ValueError(
                    "Cannot infer device/dtype: plan has no source references. "
                    "Provide device and dtype explicitly."
                )

        generator = torch.Generator(device=device)

        for target_key, expr in self.plan.mappings.items():
            refs = expr.find_refs()
            sources = {key: self.source_loader(key) for key in refs}

            # Verify device/dtype consistency
            for key, tensor in sources.items():
                if tensor.device != device or tensor.dtype != dtype:
                    raise ValueError(
                        f"Source {key} has {tensor.device}/{tensor.dtype}, "
                        f"expected {device}/{dtype}"
                    )

            # Deterministic per-target seed
            key_offset = int(hashlib.md5(str(target_key).encode()).hexdigest()[:8], 16)
            generator.manual_seed((seed + key_offset) % MAX_SEED)

            result = expr.evaluate(sources, device=device, dtype=dtype, generator=generator)
            yield target_key, result

    def execute_all(
        self,
        seed: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[W, Tensor]:
        """Execute the plan and return all results as a dict."""
        return dict(self.execute(seed, device=device, dtype=dtype))


def execute(
    plan: ExprPlan,
    source_weights: dict[W, Tensor],
    seed: int,
) -> dict[W, Tensor]:
    """Execute a plan with in-memory sources.

    Device and dtype are inferred from source tensors.
    This is a convenience function for when all sources are already loaded.
    For streaming, use StreamingExecutor directly.

    Args:
        plan: The expression plan to execute
        source_weights: Dict mapping source keys to tensors
        seed: Base seed for reproducibility
    """
    executor = StreamingExecutor(plan, lambda key: source_weights[key])
    return executor.execute_all(seed)  # Device/dtype inferred from sources
