"""Resolve exact epoch duration inside the normal training invocation."""

import hashlib
import json
import logging
import math
from fractions import Fraction

import yaml

from fast_llm.config import FieldVerboseLevel, NoAutoValidate
from fast_llm.core.distributed import broadcast_object
from fast_llm.data.dataset.config import SamplingConfig
from fast_llm.data.dataset.epoch import build_plan
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.models.gpt.config import ResolvedGPTTrainerConfig

logger = logging.getLogger(__name__)


def warmup_steps(duration, steps):
    if duration == 0:
        return 0
    whole = math.floor(duration)
    return max(1, sum(steps[:whole]) + (math.floor((duration - whole) * steps[whole]) if whole < len(steps) else 0))


def epoch_save_steps(every_epochs: float, steps_in_epoch: list[int]) -> list[int]:
    """Save after the first update reaching each epoch fraction, always including final.

    Use decimal-rational arithmetic to avoid off-by-one errors at exact boundaries.
    Several boundaries in one optimizer step produce a single save. Iterating steps
    bounds work even when the requested interval is much smaller than one step.
    """
    frequency = Fraction(str(every_epochs))
    if frequency <= 0 or not steps_in_epoch or any(count <= 0 for count in steps_in_epoch):
        raise ValueError("Save frequency and epoch step counts must be positive")
    numerator, denominator = frequency.numerator, frequency.denominator
    saves = []
    offset = 0
    for epoch, count in enumerate(steps_in_epoch):
        previous = epoch * denominator // numerator
        for step in range(1, count + 1):
            reached = (epoch * count + step) * denominator // (count * numerator)
            if reached > previous:
                saves.append(offset + step)
            previous = reached
        offset += count
    if not saves or saves[-1] != offset:
        saves.append(offset)
    return saves


def resolve_epoch_config(source, summary, distributed_config=None):
    batch = source.training.global_batch_size
    updates = {
        "type": None,
        ("training", "train_iters"): summary["epoch_end_steps"][-1],
        ("schedule", "depth_first_micro_batches"): batch // source.model.distributed.batch_data_parallel,
        ("schedule", "breadth_first_micro_batches"): 1,
        ("optimizer", "learning_rate", "decay_iterations"): summary["epoch_end_steps"][-1],
        ("data", "datasets", "training", "plan_summary"): summary,
    }
    duration = source.optimizer.learning_rate.warmup_epochs
    if duration is None and "warmup_iterations" not in source.optimizer.learning_rate._explicit_fields:
        duration = 0.1
    if duration is not None:
        updates[("optimizer", "learning_rate", "warmup_epochs")] = duration
        updates[("optimizer", "learning_rate", "warmup_iterations")] = warmup_steps(
            duration, summary["steps_in_epoch"]
        )
    for name in ("checkpoint", "export"):
        config = getattr(source.training, name)
        every = config.every_epochs
        if every is None and "interval" not in config._explicit_fields:
            every = 1
        if every is not None:
            updates[("training", name, "every_epochs")] = every
            updates[("training", name, "steps")] = epoch_save_steps(every, summary["steps_in_epoch"])
    with NoAutoValidate():
        resolved = ResolvedGPTTrainerConfig.from_dict(source, updates)
        resolved._planning_distributed = distributed_config
    resolved.validate()
    if resolved.optimizer.learning_rate.warmup_iterations > resolved.training.train_iters:
        raise ValueError("Warmup iterations exceed total training steps")
    return resolved


def get_epoch_runnable(source):
    # Same instance/config identity is retained through the execution copy.
    distributed = Distributed(source.model.distributed)
    sampling = SamplingConfig.from_dict(
        source.data,
        {
            "predicted_tokens": source.model.base_model.head.prediction_heads,
        },
        strict=False,
    )
    result = None
    if source.model.distributed.rank == 0:
        try:
            _, summary = build_plan(
                source.data.datasets["training"],
                sampling,
                source.training.epochs,
                source.training.global_batch_size,
                source.data.seed,
                source.run.experiment_dir / "dataset_cache" / "epochs",
            )
            execution = {
                "plan": summary["fingerprint"],
                "global_batch_size": source.training.global_batch_size,
                "world_size": source.model.distributed.world_size,
                "tensor_parallel": source.model.distributed.tensor_parallel,
                "pipeline_parallel": source.model.distributed.pipeline_parallel,
                "sequence_data_parallel": source.model.distributed.sequence_data_parallel,
                "sequence_tensor_parallel": source.model.distributed.sequence_tensor_parallel,
                "micro_batch_splits": source.schedule.micro_batch_splits,
            }
            summary["execution_identity"] = hashlib.sha256(json.dumps(execution, sort_keys=True).encode()).hexdigest()
            state = source.run.experiment_dir / "epoch_plan.json"
            if (
                state.is_file()
                and json.loads(state.read_text())["execution_identity"] != summary["execution_identity"]
            ):
                raise ValueError(
                    "Epoch inputs/batch/topology changed for existing experiment; use a new output directory"
                )
            state.parent.mkdir(parents=True, exist_ok=True)
            temporary = state.with_suffix(".tmp")
            temporary.write_text(json.dumps(summary, indent=2))
            temporary.replace(state)
            (source.run.experiment_dir / "sft_config.yaml").write_text(yaml.safe_dump(source.to_dict()))
            result = {"summary": summary}
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
    if distributed.world_group is not None:
        result = broadcast_object(result, distributed.world_group)
    if "error" in result:
        raise ValueError("Epoch planning failed: " + result["error"])
    summary = result["summary"]
    resolved = resolve_epoch_config(source, summary, source.model.distributed)
    if source.model.distributed.rank == 0:
        logger.info("Epoch planning summary:\n%s", yaml.safe_dump(summary))
        resolved.to_logs(verbose=FieldVerboseLevel.core, title="Resolved epoch training config")
    run = resolved.get_run(distributed)
    trainer = resolved.get_trainer_class()(config=resolved)

    def runnable():
        with run:
            trainer.setup(distributed, run)
            trainer.run()

    return runnable
