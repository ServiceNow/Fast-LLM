"""Streaming I/O for safetensor files.

This module provides memory-efficient reading and writing of sharded safetensor
files, following HuggingFace conventions.

Classes
=======

**SafetensorLoader**
    Context manager for streaming reads from sharded safetensors. Pre-builds a
    key index for O(1) lookups. With memory-mapped files, repeated loads of
    the same key return the same data pointer (no additional memory).

**ShardedSafetensorWriter**
    Context manager for streaming writes to sharded safetensors. Automatically
    flushes to a new shard when the size threshold is reached. Produces
    HuggingFace-compatible output with index.json for sharded models.

Usage
=====

    with SafetensorLoader(source_files) as loader:
        with ShardedSafetensorWriter(output_dir) as writer:
            executor = StreamingExecutor(plan, loader)
            for key, tensor in executor.execute(seed=42):
                writer.add(key, tensor)
    # Output: model-00001-of-NNNNN.safetensors, ..., model.safetensors.index.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

logger = logging.getLogger(__name__)

# Default shard size: 5GB (HuggingFace default)
DEFAULT_MAX_SHARD_SIZE = 5 * 1024 * 1024 * 1024


class SafetensorLoader:
    """Context manager for streaming reads from sharded safetensors.

    Pre-builds a key index for O(1) lookups and manages file handle lifecycle.

    Usage:
        with SafetensorLoader(source_files) as loader:
            executor = StreamingExecutor(plan, loader)
            for key, tensor in executor.execute(seed):
                ...
    """

    def __init__(self, files: list[Path], device: str = "cpu"):
        self.files = [Path(f) for f in files]
        self.device = device
        self._handles: dict[Path, Any] = {}
        self._key_index: dict[str, Path] = {}

    def __enter__(self) -> SafetensorLoader:
        # Pre-build index: key -> file (one-time O(n×m), then O(1) lookups)
        for f in self.files:
            handle = safe_open(f, framework="pt", device=self.device)
            self._handles[f] = handle
            for key in handle.keys():
                self._key_index[key] = f
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._handles.clear()
        self._key_index.clear()

    def __call__(self, key: str) -> Tensor:
        """Load a tensor by key. Raises KeyError if not found."""
        if key not in self._key_index:
            raise KeyError(f"Source key not found in any file: {key}")
        return self._handles[self._key_index[key]].get_tensor(key)

    def keys(self) -> set[str]:
        """Return all available keys across all files."""
        return set(self._key_index.keys())


class ShardedSafetensorWriter:
    """Context manager for streaming writes to sharded safetensors.

    Accumulates tensors until a size threshold is reached, then flushes
    to a shard file. This bounds peak memory to ~max_shard_size instead
    of accumulating all tensors before writing.

    Output follows HuggingFace conventions:
    - model-00001-of-00003.safetensors, model-00002-of-00003.safetensors, etc.
    - model.safetensors.index.json with weight_map and metadata

    Usage:
        with ShardedSafetensorWriter(output_dir) as writer:
            for key, tensor in executor.execute(seed):
                writer.add(key, tensor)
        # Automatically finalizes on exit, cleans up temp files on error
    """

    def __init__(
        self,
        output_dir: Path,
        max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
        base_name: str = "model",
    ):
        self.output_dir = Path(output_dir)
        self.max_shard_size = max_shard_size
        self.base_name = base_name

        # Accumulator state
        self._buffer: dict[str, Tensor] = {}
        self._buffer_bytes: int = 0
        self._shard_index: int = 0
        self._shard_files: list[Path] = []

        # For building the index
        self._weight_map: dict[str, str] = {}
        self._total_bytes: int = 0

        # Context manager state
        self._finalized: bool = False
        self._result_path: Path | None = None

    def __enter__(self) -> ShardedSafetensorWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            # Error occurred - clean up temp files
            self._cleanup_temp_files()
        else:
            # Success - finalize
            self._finalize()
        return False  # Don't suppress exceptions

    def _cleanup_temp_files(self) -> None:
        """Remove any temporary shard files on error."""
        for tmp_file in self._shard_files:
            if tmp_file.exists():
                tmp_file.unlink()
                logger.debug(f"Cleaned up temp file: {tmp_file}")

    def _tensor_bytes(self, tensor: Tensor) -> int:
        """Calculate tensor size in bytes."""
        return tensor.numel() * tensor.element_size()

    def add(self, key: str, tensor: Tensor) -> None:
        """Add a tensor to the current shard buffer.

        If adding this tensor would exceed max_shard_size, the current
        buffer is flushed first.
        """
        if self._finalized:
            raise RuntimeError("Cannot add tensors after finalization")

        tensor_size = self._tensor_bytes(tensor)

        # Flush if this would exceed the threshold (but always allow at least one tensor)
        if self._buffer and self._buffer_bytes + tensor_size > self.max_shard_size:
            self._flush()

        self._buffer[key] = tensor
        self._buffer_bytes += tensor_size
        self._total_bytes += tensor_size

    def _flush(self) -> None:
        """Write the current buffer to a shard file."""
        if not self._buffer:
            return

        self._shard_index += 1
        # Use .tmp extension until we know total shard count
        shard_file = self.output_dir / f"{self.base_name}-{self._shard_index:05d}.safetensors.tmp"

        logger.debug(
            f"Writing shard {self._shard_index}: {len(self._buffer)} tensors, " f"{self._buffer_bytes / 1e9:.2f} GB"
        )
        save_file(self._buffer, shard_file)
        self._shard_files.append(shard_file)

        # Record weight locations (will update names in finalize)
        for key in self._buffer:
            self._weight_map[key] = shard_file.name

        # Clear buffer
        self._buffer.clear()
        self._buffer_bytes = 0

    def _finalize(self) -> Path:
        """Flush remaining tensors and write the index file.

        Returns the path to the index file (or single safetensor file if only one shard).
        """
        if self._finalized:
            return self._result_path

        # Flush any remaining tensors
        self._flush()
        self._finalized = True

        total_shards = len(self._shard_files)

        if total_shards == 0:
            raise ValueError("No tensors were written")

        # Rename temp files to final names with correct shard count
        final_names: dict[str, str] = {}
        for i, tmp_file in enumerate(self._shard_files, 1):
            if total_shards == 1:
                # Single shard: just use model.safetensors
                final_name = f"{self.base_name}.safetensors"
            else:
                final_name = f"{self.base_name}-{i:05d}-of-{total_shards:05d}.safetensors"

            final_path = self.output_dir / final_name
            tmp_file.rename(final_path)
            final_names[tmp_file.name] = final_name
            logger.info(f"Saved {final_path.name}")

        # Update weight_map with final names
        for key in self._weight_map:
            old_name = self._weight_map[key]
            self._weight_map[key] = final_names[old_name]

        # Write index file if sharded
        if total_shards > 1:
            index = {
                "metadata": {"total_size": self._total_bytes},
                "weight_map": self._weight_map,
            }
            index_file = self.output_dir / f"{self.base_name}.safetensors.index.json"
            with open(index_file, "w") as f:
                json.dump(index, f, indent=2, sort_keys=True)
            logger.info(f"Saved index: {index_file.name}")
            self._result_path = index_file
        else:
            self._result_path = self.output_dir / f"{self.base_name}.safetensors"

        return self._result_path

    @property
    def result_path(self) -> Path:
        """Get the path to the result file (available after finalization)."""
        if not self._finalized:
            raise RuntimeError("Result path not available until finalized")
        return self._result_path
