from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from fast_llm.config import ValidationError
from fast_llm.csrc.data import build_epoch_sequence_boundaries
from fast_llm.data.dataset.config import SamplingConfig
from fast_llm.data.dataset.epoch import EpochDatasetConfig, PlannedEpochDataset, build_plan, resolve_sources
from fast_llm.data.dataset.memmap.language_model import LanguageModelWriter
from fast_llm.data.dataset.memmap.memmap import MemmapDataset
from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.language_model import LanguageModelBatch, LanguageModelDocument, LanguageModelTargetInput
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.models.gpt.config import GPTTrainerConfig
from fast_llm.models.gpt.epoch import epoch_save_steps, resolve_epoch_config, warmup_steps


def write_source(path, lengths, offset=0):
    docs = [
        LanguageModelDocument(tokens=torch.full((length,), i + offset, dtype=torch.int32))
        for i, length in enumerate(lengths)
    ]
    metadata = MemmapDataset.write_dataset(path, docs, LanguageModelWriter).get_metadata()
    manifest = path.with_suffix(".yaml")
    manifest.write_text(yaml.safe_dump({"config": {"type": "memmap", "path": path.name}, "metadata": metadata}))
    return {"type": "file", "path": str(manifest)}


def source_config(tmp_path, **training):
    return GPTTrainerConfig.from_dict(
        {
            "model": {"distributed": {"use_cuda": False}},
            "run": {"experiment_dir": str(tmp_path)},
            "data": {
                "micro_batch_size": 8,
                "maximum_document_length": 8,
                "truncate_documents": False,
                "datasets": {"training": {"type": "epoch", "datasets": []}},
            },
            "training": {"epochs": 3, "global_batch_size": 4, **training},
        }
    )


@pytest.mark.parametrize(
    "sizes,capacity,expected",
    [
        ([], 8, [0]),
        ([8], 8, [0, 1]),
        ([4, 4], 8, [0, 2]),
        ([5, 4, 2, 8], 8, [0, 1, 3, 4]),
        ([1], 8, [0, 1]),
    ],
)
def test_finite_boundaries(sizes, capacity, expected):
    assert build_epoch_sequence_boundaries(np.array(sizes, dtype=np.int64), capacity).tolist() == expected


@pytest.mark.parametrize("sizes", [[0], [-1], [9]])
def test_invalid_lengths(sizes):
    with pytest.raises(ValueError):
        build_epoch_sequence_boundaries(np.array(sizes, dtype=np.int64), 8)


def test_plan_coverage_cache_and_capacity(tmp_path):
    sources = [
        write_source(tmp_path / "a.fast_llm_dataset", [3, 5, 9], 0),
        write_source(tmp_path / "b.fast_llm_dataset", [4, 8, 2], 3),
    ]
    config = EpochDatasetConfig.from_dict({"datasets": sources})
    sampling = SamplingConfig(
        micro_batch_size=8, predicted_tokens=1, maximum_document_length=8, truncate_documents=False
    )
    dataset, summary = build_plan(config, sampling, 3, 4, 123, tmp_path / "cache")
    assert summary["eligible_documents"] == 5
    assert summary["skipped_documents"] == 1
    planned = PlannedEpochDataset(dataset, summary)
    begin = 0
    for end_step, count in zip(summary["epoch_end_steps"], summary["packed_sequences"]):
        end = end_step * 4
        docs = [doc for i in range(begin, end) for doc in planned[i]]
        assert sorted(int(doc.tokens[0]) for doc in docs) == [0, 1, 3, 4, 5]
        assert sum(bool(planned[i]) for i in range(begin, end)) == count
        assert all(sum(len(doc) for doc in planned[i]) <= 9 for i in range(begin, end))
        begin = end
    _, cached = build_plan(config, sampling, 3, 8, 123, tmp_path / "cache")
    assert cached["fingerprint"] == summary["fingerprint"]
    assert cached["packed_sequences"] == summary["packed_sequences"]
    assert cached["steps_in_epoch"] == [1, 1, 1]
    resolved = config.to_copy({"plan_summary": summary})
    assert len(resolved.build_and_sample(sampling, len(planned), 123)) == len(planned)
    with pytest.raises(ValueError, match="capacity"):
        resolved.build_and_sample(sampling.to_copy({"predicted_tokens": 2}), len(planned), 123)
    with pytest.raises(ValueError, match="Duplicate"):
        resolve_sources(config.datasets + config.datasets)


def test_empty_batch_global_normalization():
    batch = LanguageModelBatch.from_documents([], 9)
    config = LanguageModelBatchPreprocessingConfig(
        distributed=DistributedConfig(use_cuda=False), return_label_counts=True, return_valid_document_count=True
    )
    empty = batch.get_model_inputs(config)[0].targets[0]
    real = (
        LanguageModelBatch.from_documents([LanguageModelDocument(tokens=torch.arange(5))], 9)
        .get_model_inputs(config)[0]
        .targets[0]
    )
    LanguageModelTargetInput.share_batch_data([real, empty], SimpleNamespace(batch_data_group=None))
    assert empty.num_labels == 0 and empty.num_valid_documents == 0
    assert empty.num_labels_in_batch == real.num_labels
    assert empty.num_valid_documents_in_batch == 1


def test_resolved_config_and_saves(tmp_path):
    source = source_config(tmp_path, checkpoint={"every_epochs": 1, "keep": 1})
    summary = {"steps_in_epoch": [2, 3, 2], "epoch_end_steps": [2, 5, 7]}
    resolved = resolve_epoch_config(source, summary)
    assert source.training.train_iters == 0
    assert resolved.training.train_iters == 7
    assert resolved.schedule.depth_first_micro_batches == 4
    assert resolved.training.checkpoint.steps == [2, 5, 7]
    assert resolved.training.checkpoint.enabled(5)
    assert not resolved.training.checkpoint.enabled(4)
    assert resolved.training.checkpoint.get_count(5) == 2
    assert resolved.training.checkpoint.to_delete([2, 5, 7]) == [2, 5]
    assert warmup_steps(1.5, [2, 3, 2]) == 3
    assert warmup_steps(0, [2, 3]) == 0
    assert warmup_steps(3, [2, 3, 2]) == 7


def test_explicit_zero_conflict(tmp_path):
    with pytest.raises(ValueError, match="train_iters"):
        source_config(tmp_path, train_iters=0)


def test_shutdown_rejection():
    with pytest.raises((ValueError, ValidationError), match="shutdown"):
        from fast_llm.engine.training.config import TrainingConfig

        TrainingConfig.from_dict({"checkpoint": {"every_epochs": 1}, "shutdown": {"interval": 2}})


@pytest.mark.parametrize("heads", [1, 2])
def test_planning_capacity_matches_model(tmp_path, heads):
    from fast_llm.engine.distributed.config import PhaseType

    source = source_config(tmp_path).to_copy({("model", "base_model", "head", "prediction_heads"): heads})
    model = source.model.get_model_class()(source.model, verbose=False)
    actual = model.get_preprocessing_config(PhaseType.training).predicted_tokens
    assert actual == source.model.base_model.head.prediction_heads


def test_full_warmup_and_final_save_retention(tmp_path):
    from fast_llm.engine.optimizer.learning_rate import create_schedule_from_config

    source = source_config(tmp_path, checkpoint={"every_epochs": 2, "keep": 1}).to_copy(
        {("optimizer", "learning_rate", "warmup_epochs"): 3}
    )
    resolved = resolve_epoch_config(source, {"steps_in_epoch": [1, 1, 1], "epoch_end_steps": [1, 2, 3]})
    assert resolved.training.checkpoint.steps == [2, 3]
    assert resolved.training.checkpoint.to_delete([2, 3]) == [2]
    schedule = create_schedule_from_config(resolved.optimizer.learning_rate)
    assert schedule(3) == resolved.optimizer.learning_rate.base


def test_globally_empty_epoch_batch_rejected():
    target = LanguageModelTargetInput(num_labels=0, require_supervised_batch=True)
    with pytest.raises(ValueError, match="no supervised"):
        LanguageModelTargetInput.share_batch_data([target], SimpleNamespace(batch_data_group=None))
    # Preserve legacy behavior outside epoch mode.
    legacy = LanguageModelTargetInput(num_labels=0)
    LanguageModelTargetInput.share_batch_data([legacy], SimpleNamespace(batch_data_group=None))
    assert legacy.num_labels_in_batch == 0


@pytest.mark.parametrize(
    "frequency,steps,expected",
    [
        (0.5, [8, 12, 4], [4, 8, 14, 20, 22, 24]),
        (0.25, [8, 12, 4], [2, 4, 6, 8, 11, 14, 17, 20, 21, 22, 23, 24]),
        (0.25, [3, 5], [1, 2, 3, 5, 6, 7, 8]),
        (1.5, [2, 3, 2], [4, 7]),
        (0.1, [10], list(range(1, 11))),
        (1e-12, [2], [1, 2]),
        (4, [2, 3], [5]),
    ],
)
def test_fractional_epoch_save_steps(frequency, steps, expected):
    assert epoch_save_steps(frequency, steps) == expected


def test_fractional_checkpoints_independent_of_exports(tmp_path):
    source = source_config(tmp_path, checkpoint={"every_epochs": 0.25, "keep": 2}, export={"every_epochs": 1})
    resolved = resolve_epoch_config(source, {"steps_in_epoch": [8, 12, 4], "epoch_end_steps": [8, 20, 24]})
    checkpoints = resolved.training.checkpoint
    assert checkpoints.steps == [2, 4, 6, 8, 11, 14, 17, 20, 21, 22, 23, 24]
    assert resolved.training.export.steps == [8, 20, 24]
    assert checkpoints.to_delete(checkpoints.steps) == checkpoints.steps[:-2]
    assert checkpoints.get_count(14) == 6
    with pytest.raises((ValueError, ValidationError), match="finite"):
        source_config(tmp_path, checkpoint={"every_epochs": float("inf")})


@pytest.mark.parametrize(
    "world_size,batch_size,expected_accumulation",
    [
        (8, 96, 96),
        (64, 96, 12),
        (64, 64, 8),
    ],
)
def test_global_batch_size_resolves_accumulation(tmp_path, world_size, batch_size, expected_accumulation):
    source = source_config(tmp_path).to_copy(
        {
            ("model", "distributed", "world_size"): world_size,
            ("model", "distributed", "local_world_size"): 8,
            ("model", "distributed", "tensor_parallel"): 2,
            ("model", "distributed", "sequence_data_parallel"): 4,
            ("training", "global_batch_size"): batch_size,
            ("schedule", "micro_batch_splits"): 2,
        }
    )
    resolved = resolve_epoch_config(source, {"steps_in_epoch": [2, 3, 2], "epoch_end_steps": [2, 5, 7]})
    assert resolved.schedule.depth_first_micro_batches == expected_accumulation
    assert resolved.schedule.breadth_first_micro_batches == 1
    assert resolved.schedule.sequential_micro_batches * resolved.model.distributed.batch_data_parallel == batch_size
    assert resolved.schedule.micro_batch_splits == 2
    assert resolved.data.micro_batch_size == source.data.micro_batch_size
    assert resolved.model.distributed.tensor_parallel == 2
    assert resolved.model.distributed.sequence_data_parallel == 4


def test_incompatible_global_batch_size_rejected(tmp_path):
    with pytest.raises(ValueError, match="global_batch_size=95 must be divisible by batch_data_parallel=8"):
        source_config(tmp_path).to_copy(
            {
                ("model", "distributed", "world_size"): 64,
                ("model", "distributed", "local_world_size"): 8,
                ("model", "distributed", "tensor_parallel"): 2,
                ("model", "distributed", "sequence_data_parallel"): 4,
                ("training", "global_batch_size"): 95,
            }
        )
