"""Finite, cached whole-document packing plans for exact SFT epochs."""

import bisect
import hashlib
import json
import math
import os
import pathlib
import tempfile
import time

import numpy as np
import torch

from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.config import (
    BlendedDatasetConfig,
    ConcatenatedDatasetConfig,
    SamplingConfig,
)
from fast_llm.data.dataset.epoch_config import EpochDatasetConfig
from fast_llm.data.dataset.gpt.config import GPTDatasetFromFileConfig
from fast_llm.data.dataset.indexed import ConcatenatedDataset
from fast_llm.data.dataset.memmap.config import MemmapDatasetConfig

PLAN_VERSION = 1


def resolve_sources(configs):
    """Flatten prepared shard manifests, preserving registry/path validation."""
    shards = []
    seen = set()

    def visit(config):
        if isinstance(config, GPTDatasetFromFileConfig):
            visit(config._load_config())
        elif isinstance(config, (BlendedDatasetConfig, ConcatenatedDatasetConfig)):
            # Epoch semantics intentionally replace generated shard sampling weights.
            for child in config.datasets:
                visit(child)
        elif isinstance(config, MemmapDatasetConfig):
            path = config.path.resolve()
            if path in seen:
                raise ValueError(f"Duplicate epoch dataset shard: {path}")
            if not path.is_file():
                raise ValueError("Epoch datasets require the prepared single-file memmap format")
            seen.add(path)
            shards.append(config)
        else:
            raise ValueError(f"Unsupported epoch source: {type(config).__name__}; use prepared memmap manifests")

    for config in configs:
        visit(config)
    if not shards:
        raise ValueError("Epoch training requires at least one dataset")
    return shards


def build_plan(
    config: EpochDatasetConfig,
    sampling: SamplingConfig,
    epochs: int,
    global_batch_size: int,
    seed: int,
    cache_directory: pathlib.Path,
):
    from fast_llm.csrc.data import build_epoch_sequence_boundaries

    started = time.perf_counter()
    if sampling.truncate_documents:
        raise ValueError("Epoch training requires truncate_documents: false")
    shards = resolve_sources(config.datasets)
    dataset = ConcatenatedDataset("epoch_training", [shard.build() for shard in shards])
    sizes = dataset.get_document_sizes().cpu().numpy().astype(np.int64, copy=False)
    maximum = sampling.sampling_maximum_document_length
    eligible = np.flatnonzero((sizes > 0) & (sizes <= maximum))
    if eligible.size == 0:
        raise ValueError("No eligible documents for epoch training")
    identity = {
        "version": PLAN_VERSION,
        "shards": [
            {
                "path": str(shard.path.resolve()),
                "size": shard.path.stat().st_size,
                "mtime_ns": shard.path.stat().st_mtime_ns,
            }
            for shard in shards
        ],
        "lengths_sha256": hashlib.sha256(sizes.tobytes()).hexdigest(),
        "capacity": sampling.sample_size,
        "maximum_document_length": maximum,
        "epochs": epochs,
        "seed": seed,
        "shuffle_backend": "torch_cpu",
        "torch_version": str(torch.__version__),
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = pathlib.Path(cache_directory) / fingerprint
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="epoch-plan-", dir=root.parent) as temporary:
            temporary = pathlib.Path(temporary)
            counts = []
            for epoch in range(epochs):
                generator = torch.Generator(device="cpu").manual_seed(seed + epoch * 571)
                order = eligible[torch.randperm(len(eligible), generator=generator).numpy()]
                boundaries = build_epoch_sequence_boundaries(sizes[order], sampling.sample_size)
                np.save(temporary / f"order_{epoch}.npy", order)
                np.save(temporary / f"boundaries_{epoch}.npy", boundaries)
                counts.append(len(boundaries) - 1)
            tokens = int(sizes[eligible].sum())
            summary = {
                "fingerprint": fingerprint,
                "identity": identity,
                "eligible_documents": len(eligible),
                "eligible_tokens": tokens,
                "skipped_documents": len(sizes) - len(eligible),
                "skipped_tokens": int(sizes.sum()) - tokens,
                "packed_sequences": counts,
                "padding_tokens": [count * sampling.sample_size - tokens for count in counts],
                "packing_efficiency": [tokens / (count * sampling.sample_size) for count in counts],
            }
            (temporary / "summary.json").write_text(json.dumps(summary, indent=2))
            # Publish complete directory; tolerate a concurrent identical cache builder.
            try:
                os.rename(temporary, root)
            except OSError:
                if not summary_path.is_file():
                    raise
    summary = json.loads(summary_path.read_text())
    if summary["identity"] != identity:
        raise ValueError("Epoch plan cache identity mismatch")
    steps = [math.ceil(count / global_batch_size) for count in summary["packed_sequences"]]
    summary = dict(
        summary,
        steps_in_epoch=steps,
        epoch_end_steps=np.cumsum(steps).tolist(),
        global_batch_size=global_batch_size,
        padding_sequence_slots=[
            step * global_batch_size - count for step, count in zip(steps, summary["packed_sequences"])
        ],
        planning_seconds=time.perf_counter() - started,
        plan_directory=str(root.resolve()),
    )
    return dataset, summary


class PlannedEpochDataset(SampledDataset):
    def __init__(self, dataset, summary):
        self._dataset = dataset
        self._summary = summary
        self._orders = [
            np.load(pathlib.Path(summary["plan_directory"]) / f"order_{e}.npy", mmap_mode="r")
            for e in range(len(summary["steps_in_epoch"]))
        ]
        self._boundaries = [
            np.load(pathlib.Path(summary["plan_directory"]) / f"boundaries_{e}.npy", mmap_mode="r")
            for e in range(len(self._orders))
        ]
        self._ends = [step * summary["global_batch_size"] for step in summary["epoch_end_steps"]]

    @property
    def name(self):
        return "epoch_training"

    @property
    def requires_broadcast(self):
        return self._dataset.requires_broadcast

    def __len__(self):
        return self._ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        epoch = bisect.bisect_right(self._ends, index)
        local = index - (self._ends[epoch - 1] if epoch else 0)
        boundaries = self._boundaries[epoch]
        if local >= len(boundaries) - 1:
            return []  # Explicit padding slot, never a repeated training document.
        begin, end = boundaries[local : local + 2]
        return [self._dataset.get_document(int(document)) for document in self._orders[epoch][begin:end]]
