import json
import logging
import math
import pathlib
import shutil
import statistics
import typing

from fast_llm.config import Field, FieldHint, config_class
from fast_llm.engine.config_utils.compare_tensor_logs import CompareConfig
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.models.gpt.config import PretrainedGPTModelConfig

# Populate the trainer dynamic-type registry.
import fast_llm.data.auto  # noqa: F401  # isort:skip
import fast_llm.engine.checkpoint.convert  # noqa: F401  # isort:skip
import fast_llm.models.auto  # noqa: F401  # isort:skip

logger = logging.getLogger(__name__)


_REFERENCE_NAME = "reference"
_MODEL_TYPE = "gpt"
# Embedding-weight gradients are row-sparse (only input-token rows non-zero), so a
# uniformly-spaced sample of vocab_size entries usually misses all of them. The pattern
# is applied via `TensorLogsConfig.sample_level_overrides` and picked up inside
# `log_tensor` (samples = 2 ** (level - 3) -> level 23 yields ~1M samples per tensor).
_SPARSE_GRAD_LEVEL = 23
_SPARSE_GRAD_OVERRIDES = {r"Global gradient: embeddings\.": _SPARSE_GRAD_LEVEL}
_CHOSEN_LOGPROB_NAME = "chosen_logprob"
# Seed for the random-token fixed input when no input text file is given.
_INPUT_SEED = 0
# Auto-calibration of the constant gradient scaler. Each variant runs a calibration pass at
# `scale=1` (no overflow risk), then the actual run uses the largest power-of-2 scale that
# keeps logged gradient magnitudes (and a small safety factor for hidden in-kernel
# intermediates like norm partial sums) within fp16's representable range. Per-variant
# unscaling at compare time lets different variants pick different scales without polluting
# the relative metrics.
_HIDDEN_INTERMEDIATE_HEADROOM = 4.0  # safety factor for fused-kernel partial sums we don't log
_CALIBRATION_SUBDIR_PREFIX = ".calibration_"
# Variant-override keys starting with this prefix are interpreted as `torch.backends.<rest>` and
# applied before each run. Used for diagnostics (e.g. enabling bf16 reduced-precision reductions);
# entries are listed in `_TORCH_BACKEND_DEFAULTS` and reset to their defaults before applying.
_TORCH_BACKEND_PREFIX = "_torch_backend."
_TORCH_BACKEND_DEFAULTS = {
    "cuda.matmul.allow_bf16_reduced_precision_reduction": False,
}
_TORCH_MATMUL_PRECISION_KEY = "_torch_matmul_precision"


@config_class()
class EvaluatePrecisionConfig(PretrainedGPTModelConfig, RunnableConfig):
    """Evaluate layer-wise numerical-error propagation against an fp32 reference.

    Inherits `model` and `pretrained` from `PretrainedGPTModelConfig`: either or both
    can be set in the YAML. The tool runs one fp32 reference + one trainer invocation
    per variant, captures per-layer forward activations and input gradients via the
    standard tensor-logs pipeline, and reports per-tensor RMS / max diffs.
    """

    _abstract = False
    variants: dict[str, typing.Any] = Field(
        desc="Named override bundles to evaluate against the fp32 reference."
        " Each value is a flat dict mapping dotted-path keys (same syntax as the Fast-LLM CLI) to values.",
        hint=FieldHint.core,
    )
    output_dir: pathlib.Path = Field(
        desc="Directory for per-run tensor-log artifacts and the final JSON report.",
        hint=FieldHint.core,
    )
    num_samples: int = Field(
        default=8192,
        desc="Number of sampled values stored per logged tensor (rounded up to next power of 2)."
        " Sparse tensors (e.g. embedding-weight gradients) get a higher level via"
        " `TensorLogsConfig.sample_level_overrides`.",
        hint=FieldHint.feature,
    )
    sequence_length: int = Field(
        default=2048,
        desc="Sequence length per micro-batch sample. Drives both `data.micro_batch_size` (the"
        " per-sample token count, despite the name) and `data.maximum_document_length`.",
        hint=FieldHint.feature,
    )
    input_text_file: pathlib.Path | None = Field(
        default=None,
        desc="If set, tokenize this text file (via the pretrained tokenizer) to build the fixed model"
        " input, tiled/truncated to `sequence_length`. If unset, the input is uniform-random token ids."
        " The exact input tensor is saved to `output_dir/input_ids.pt` so the DeepSpeed-side tool"
        " (`tools/evaluate_precision_deepspeed.py`) can consume the identical model input.",
        hint=FieldHint.feature,
    )
    forward_only: bool = Field(
        default=False,
        desc="Run a single forward pass in inference mode (`StageMode.inference`, no optimizer or"
        " gradient buffers) instead of forward+backward. Fits large models (e.g. 7B in fp32) that would"
        " OOM with gradient and optimizer state. Only the per-token log π (chosen_logprob) and forward"
        " activations are captured — no gradient/parameter-gradient tables, and no gradient scaling.",
        hint=FieldHint.feature,
    )

    def _validate(self) -> None:
        super()._validate()
        assert _REFERENCE_NAME not in self.variants, f"'{_REFERENCE_NAME}' is reserved for the fp32 baseline."
        for name, overrides in self.variants.items():
            assert isinstance(overrides, dict) and all(
                isinstance(k, str) for k in overrides
            ), f"Variant {name!r} must be a flat dict of dotted-path string keys."

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        input_ids = self._prepare_input_ids()
        runs: dict[str, dict[str, typing.Any]] = {_REFERENCE_NAME: {}}
        runs.update(self.variants)
        scales: dict[str, float] = {}
        for name, variant_overrides in runs.items():
            if self.forward_only:
                # No backward -> no gradients to scale; run once directly.
                self._run_one(name, variant_overrides, input_ids)
                scales[name] = 1.0
            else:
                scales[name] = self._calibrate_and_run(name, variant_overrides, input_ids)
            self._save_chosen_logprob(name)

        ref_artifacts = self._artifact_path(_REFERENCE_NAME)
        results = {
            name: self._compare(ref_artifacts, self._artifact_path(name), scales[_REFERENCE_NAME], scales[name])
            for name in self.variants
        }

        report_path = self.output_dir / "precision_report.json"
        report_path.write_text(json.dumps({"scales": scales, "variants": results}, indent=2))
        logger.info(f"Wrote report to {report_path}")
        logger.info(f"Per-variant gradient scales: {scales}")

        for name, rows in results.items():
            _print_table(name, rows)
        _print_summary(results)

    def _calibrate_and_run(
        self, name: str, variant_overrides: dict[str, typing.Any], input_ids: "torch.Tensor"
    ) -> float:
        """Pick a power-of-2 gradient scale for this variant via a calibration pass, then run with it.

        Calibration runs with `constant=1.0` so no overflow is possible; scanning logged gradients
        then gives us `max_unscaled`. The largest safe power of 2 keeps `scale * max_unscaled` below
        `fp16_max / hidden_intermediate_budget`, where the budget reserves headroom for partial sums
        inside fused kernels (e.g. norm-weight grads sum over the sequence dimension).
        """
        import torch

        cal_dir = self.output_dir / f"{_CALIBRATION_SUBDIR_PREFIX}{name}"
        self._run_one(name, variant_overrides, input_ids, constant_scale=1.0, experiment_dir=cal_dir)
        max_unscaled = _scan_max_grad(cal_dir / "runs" / "0" / "artifacts")
        shutil.rmtree(cal_dir)
        if max_unscaled <= 0.0:
            scale = 1.0
            logger.warning(f"[{name}] calibration found no nonzero gradient — falling back to scale=1.0")
        else:
            fp16_max = torch.finfo(torch.float16).max
            optimal_unrounded = fp16_max / max_unscaled / _HIDDEN_INTERMEDIATE_HEADROOM
            scale = float(2 ** max(0, math.floor(math.log2(optimal_unrounded))))
        logger.info(f"[{name}] calibration: max_unscaled={max_unscaled:.4e} -> gradient_scaler.constant={scale:g}")
        self._run_one(name, variant_overrides, input_ids, constant_scale=scale)
        return scale

    def _prepare_input_ids(self) -> "torch.Tensor":
        """Build the fixed model input once and save it so the DeepSpeed-side tool feeds the exact
        same tokens. Going through Fast-LLM's data pipeline would re-randomize the model input
        (shuffle/packing), so the input is constructed directly here and fed verbatim to the runner."""
        import torch

        vocab_size = self.model.base_model.embeddings.vocab_size
        if self.input_text_file is not None:
            import transformers

            tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.pretrained.path))
            ids = tokenizer(self.input_text_file.read_text(), return_tensors="pt").input_ids[0]
            if ids.numel() < self.sequence_length:
                ids = ids.repeat((self.sequence_length + ids.numel() - 1) // ids.numel())
            ids = ids[: self.sequence_length].to(torch.int64)
        else:
            generator = torch.Generator().manual_seed(_INPUT_SEED)
            ids = torch.randint(0, vocab_size, (self.sequence_length,), generator=generator, dtype=torch.int64)
        input_ids = ids.unsqueeze(0)
        path = self.output_dir / "input_ids.pt"
        torch.save(input_ids, path)
        logger.info(f"Shared model input: {tuple(input_ids.shape)} saved to {path}")
        return input_ids

    def _artifact_path(self, name: str) -> pathlib.Path:
        return self.output_dir / name / "runs" / "0" / "artifacts"

    def _save_chosen_logprob(self, name: str) -> None:
        """Persist the full per-token log π vector (token order) as a plain tensor for the
        cross-engine comparison. The chosen_logprob loss logs the whole tensor with step=1, so the
        saved samples are the complete ordered vector — aligned 1:1 with vLLM's `prompt_logprobs[1:]`."""
        import torch

        compare_config = CompareConfig()
        errors: list[str] = []
        logs = compare_config._extract_tensor_logs(self._artifact_path(name), errors)
        for step_logs in logs.values():
            for tensor_name, entry in step_logs.items():
                if tensor_name.split(":", 1)[-1].strip() == _CHOSEN_LOGPROB_NAME:
                    torch.save(entry["samples"].float().cpu(), self.output_dir / f"logprobs_{name}.pt")
                    return
        logger.warning(f"[{name}] chosen_logprob not found in tensor logs; cross-engine vector not saved")

    def _run_one(
        self,
        name: str,
        variant_overrides: dict[str, typing.Any],
        input_ids: "torch.Tensor",
        *,
        constant_scale: float | None = None,
        experiment_dir: pathlib.Path | None = None,
    ) -> None:
        # The trainer's Run picks the next `runs/<n>` subdir based on what already exists; wipe
        # any prior contents so each invocation lands in `runs/0` and stale artifacts can't be
        # read by `_artifact_path` below.
        if experiment_dir is None:
            experiment_dir = self.output_dir / name
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        # Base config: hardcoded training/optimizer/data/run skeleton plus the user's model/pretrained.
        # Forced fp32 on the reference baseline lives in here too so a variant can override it.
        optimizer_config: dict[str, typing.Any] = {
            "learning_rate": {"base": 0.0, "decay_style": "constant", "warmup_iterations": 0},
        }
        if constant_scale is not None:
            optimizer_config["gradient_scaler"] = {"constant": float(constant_scale)}
        base_dict: dict[str, typing.Any] = {
            "pretrained": self.pretrained.to_dict(),
            "model": self.model.to_dict(),
            "training": {
                "train_iters": 1,
                "num_workers": 0,
                "logs": {"interval": 1},
            },
            "optimizer": optimizer_config,
            # The lean runner feeds a fixed input directly and ignores this dataset; it's only here so
            # the TrainerConfig validates. Despite the name, `data.micro_batch_size` is the per-sample
            # sequence length, not the batch dimension.
            "data": {
                "datasets": {"training": {"type": "random"}},
                "micro_batch_size": self.sequence_length,
                "maximum_document_length": self.sequence_length,
            },
            "run": {
                "experiment_dir": str(experiment_dir.resolve()),
                "tensor_logs": {
                    "save": True,
                    "show": False,
                    "sample_level_overrides": _SPARSE_GRAD_OVERRIDES,
                },
            },
        }
        # Translate `num_samples` to a `log_tensor` level: 2**(level-3) = samples.
        log_level = math.ceil(math.log2(max(self.num_samples, 1))) + 3
        fp32_dtypes = {
            ("model", "distributed", "compute_dtype"): "float32",
            ("model", "distributed", "optimization_dtype"): "float32",
        }
        # Split off torch-backend overrides before passing the rest to Fast-LLM's config system.
        backend_overrides = {
            key[len(_TORCH_BACKEND_PREFIX) :]: value
            for key, value in variant_overrides.items()
            if key.startswith(_TORCH_BACKEND_PREFIX)
        }
        _apply_torch_backend_overrides(backend_overrides)
        matmul_precision = variant_overrides.get(_TORCH_MATMUL_PRECISION_KEY, "highest")
        _apply_torch_matmul_precision(matmul_precision)
        variant_updates = {
            tuple(key.split(".")): value
            for key, value in variant_overrides.items()
            if not key.startswith(_TORCH_BACKEND_PREFIX) and key != _TORCH_MATMUL_PRECISION_KEY
        }
        # Tool-required overrides win over variants — a variant must not silently disable tensor logging.
        tool_overrides: dict[tuple[str, ...], typing.Any] = {
            ("model", "multi_stage", "debug_layer_outputs"): log_level,
            # Capture the LM-head logits via the `output_hidden_states` mechanism: the head's
            # `_debug(logits, ...)` call matches this pattern and emits to `tensor_logs`.
            ("model", "multi_stage", "debug_hidden_states_log"): [r"head\.logits"],
            # Diagnostic loss that logs log π(label) per position via the tensor-log pipeline.
            # Contributes no gradient (weight=0); the comparison code picks it up by name.
            ("model", "base_model", "head", "losses", _CHOSEN_LOGPROB_NAME): {"type": "chosen_logprob"},
        }
        if not self.forward_only:
            tool_overrides[("model", "multi_stage", "debug_layer_gradients")] = log_level
            tool_overrides[("model", "multi_stage", "debug_all_param_gradients")] = log_level
            # When the user hasn't configured any loss, the head defaults to cross-entropy. Adding a
            # loss explicitly suppresses that default, so re-add it so gradients still flow.
            if not (self.model.base_model.head.losses or {}):
                tool_overrides[("model", "base_model", "head", "losses", "cross_entropy")] = {"type": "label"}
        # In forward-only mode only chosen_logprob runs (no grad-producing loss), so no backward
        # happens and `StageMode.inference` (which allocates no gradient buffers) is sufficient.
        logger.info(f"=== Running {name!r} ===")
        if variant_overrides:
            logger.info(f"Variant overrides: {variant_overrides}")
        trainer_class = TrainerConfig.get_subclass(_MODEL_TYPE)
        trainer_config = trainer_class.from_dict(base_dict, fp32_dtypes, variant_updates, tool_overrides)
        trainer_config.configure_logging()
        _run_fixed_input(trainer_config, input_ids, self.sequence_length, forward_only=self.forward_only)

    def _compare(
        self,
        ref_path: pathlib.Path,
        test_path: pathlib.Path,
        ref_scale: float,
        test_scale: float,
    ) -> list[dict[str, typing.Any]]:
        compare_config = CompareConfig()
        errors: list[str] = []
        ref_logs = compare_config._extract_tensor_logs(ref_path, errors)
        test_logs = compare_config._extract_tensor_logs(test_path, errors)
        for error in errors:
            logger.warning(error)
        # Each variant's gradient logs are scaled by its own `constant` factor (auto-calibrated).
        # Undo per-variant scaling so the relative comparison reflects unscaled gradient diffs.
        _unscale_gradients_in_place(ref_logs, ref_scale)
        _unscale_gradients_in_place(test_logs, test_scale)
        rows: list[dict[str, typing.Any]] = []
        for step_name in sorted(ref_logs):
            if step_name not in test_logs:
                logger.warning(f"Step {step_name!r} missing from test logs")
                continue
            step_ref = ref_logs[step_name]
            step_test = test_logs[step_name]
            for tensor_name, ref in step_ref.items():
                if tensor_name not in step_test:
                    continue
                metrics = compare_config._compute_diff(ref, step_test[tensor_name], step_name, tensor_name)
                if metrics is None:
                    continue
                rows.append(
                    {
                        "step": step_name,
                        "tensor_name": tensor_name,
                        "kind": _classify(tensor_name),
                        "shape": ref["shape"],
                        **metrics,
                    }
                )
        return rows


def _run_fixed_input(config, input_ids, sequence_length: int, *, forward_only: bool = False) -> None:
    """Lean run on a fixed, already-preprocessed input — like `InferenceRunner` but feeding a fixed input
    so the model sees exactly `input_ids` (the data pipeline would re-randomize it) and the tool stops
    paying for training/data-loading infrastructure it doesn't need.

    Default mode is a training-phase schedule + an (lr-0) optimizer so `run_step` runs the backward and
    the chosen-logprob loss / `debug_all_param_gradients` logging captures everything.

    `forward_only=True` runs a single forward in inference mode: `StageMode.inference` (no gradient
    buffers), no optimizer, and a validation-phase schedule (forward-only, but still produces labels —
    `PhaseType.inference` would zero `num_labels`). The head skips all losses in eval mode, so after setup
    the head(s) are forced back into train mode directly; `run_step`'s per-step `multi_stage.train(False)`
    is a guarded no-op once `_training` is False, so the head stays trained and logs chosen_logprob.
    Valid only because no grad-producing loss is configured, so no backward touches the missing buffers."""
    import gc

    import torch

    from fast_llm.data.document.language_model import LanguageModelBatch
    from fast_llm.engine.distributed.config import PhaseType
    from fast_llm.engine.distributed.distributed import Distributed
    from fast_llm.engine.multi_stage.config import StageMode
    from fast_llm.engine.optimizer.config import ParamGroup
    from fast_llm.engine.schedule.runner import ScheduleRunner
    from fast_llm.engine.schedule.schedule import Schedule

    phase = PhaseType.validation if forward_only else PhaseType.training
    distributed = Distributed(config.model.distributed)
    run = config.get_run(distributed)
    optimizer = None
    with run:
        multi_stage = config.model.get_model_class()(
            config.model, optimizer_state_names=() if forward_only else config.optimizer.state_names()
        )
        with torch.no_grad():
            multi_stage.setup(distributed, mode=StageMode.inference if forward_only else StageMode.training)
        if config.pretrained.path is not None and config.pretrained.model_weights:
            multi_stage.load_checkpoint(config.pretrained)
        else:
            multi_stage.initialize_weights()
        if not forward_only:
            param_groups, grads_for_norm = multi_stage.get_param_groups(ParamGroup)
            optimizer = config.optimizer.optimizer_cls(
                config.optimizer, param_groups=param_groups, grads_for_norm=grads_for_norm, distributed=distributed
            )
            optimizer.reset_state()
        runner = ScheduleRunner(
            config=config.schedule, multi_stage=multi_stage, distributed_config=config.model.distributed
        )
        with torch.no_grad():
            runner.setup(distributed, optimizer)
        if forward_only:
            from fast_llm.layers.language_model.head import LanguageModelHead

            multi_stage.train(False)
            for module in multi_stage.base_model.modules():
                if isinstance(module, LanguageModelHead):
                    module.train(True)
        preprocessing_config = multi_stage.get_preprocessing_config(phase, config.schedule.micro_batch_splits)
        # `get_model_inputs` splits off `num_labels` tokens for the shifted next-token labels, so the
        # actual model input is `len(tokens) - num_labels`. The schedule meta must match that length.
        schedule = Schedule(
            config=config.schedule,
            multi_stage=multi_stage,
            batch_meta=preprocessing_config.get_input_meta(sequence_length - preprocessing_config.num_labels),
            distributed_config=config.model.distributed,
            phase=phase,
        )
        tokens = input_ids.flatten().to(device=distributed.device, dtype=torch.int64)
        batch = LanguageModelBatch(tokens=tokens, lengths=[tokens.numel()])
        model_inputs = batch.get_model_inputs(preprocessing_config)
        runner.run_step(iter((tuple(model_inputs),)), schedule, iteration=1)
    # Break the trainer/model/runner reference cycles so each variant's GPU memory is reclaimed.
    del multi_stage, optimizer, runner, schedule, distributed, run
    gc.collect()
    torch.cuda.empty_cache()


def _is_gradient_like(tensor_name: str) -> bool:
    # Anything affected by the loss-scaling multiplier: parameter gradients from `Fsdp.log_shard`,
    # backward activations from layer hooks, and explicit `.grad` debug entries (e.g. logits.grad).
    return ("gradient:" in tensor_name) or (" bw" in tensor_name) or (".grad" in tensor_name)


def _scan_max_grad(artifact_path: pathlib.Path) -> float:
    max_abs = 0.0
    compare_config = CompareConfig()
    errors: list[str] = []
    logs = compare_config._extract_tensor_logs(artifact_path, errors)
    for step_logs in logs.values():
        for tensor_name, entry in step_logs.items():
            if not _is_gradient_like(tensor_name):
                continue
            # Saved stats include min/max; fall back to samples if absent.
            if "max" in entry and "min" in entry:
                value = max(abs(float(entry["max"])), abs(float(entry["min"])))
            else:
                value = float(entry["samples"].abs().max().item())
            if math.isfinite(value) and value > max_abs:
                max_abs = value
    return max_abs


def _unscale_gradients_in_place(logs: dict, scale: float) -> None:
    if scale == 1.0:
        return
    inv = 1.0 / scale
    for step_logs in logs.values():
        for tensor_name, entry in step_logs.items():
            if not _is_gradient_like(tensor_name):
                continue
            entry["samples"] = entry["samples"].float() * inv
            for key in ("min", "max", "mu", "std"):
                if key in entry and entry[key] is not None:
                    entry[key] = float(entry[key]) * inv


def _apply_torch_backend_overrides(overrides: dict[str, typing.Any]) -> None:
    import torch

    unknown = set(overrides) - set(_TORCH_BACKEND_DEFAULTS)
    if unknown:
        logger.warning(f"Unknown torch backend overrides (ignored): {sorted(unknown)}")
    for path, default in _TORCH_BACKEND_DEFAULTS.items():
        value = overrides.get(path, default)
        obj: typing.Any = torch.backends
        parts = path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


def _apply_torch_matmul_precision(precision: str) -> None:
    import torch

    torch.set_float32_matmul_precision(precision)


def _layer_name(tensor_name: str) -> str:
    # Stage hooks name tensors `Global <layer> fw: ...` / `Global <layer> bw: ...`;
    # Fsdp.log_shard names weight gradients `Global gradient: <param-path>`.
    prefix = tensor_name.split(":", 1)[0].strip().split()
    if prefix == ["Global", "gradient"]:
        param = tensor_name.split(":", 1)[1].strip()
        return param.split(".")[0]
    if prefix and prefix[0] == "Global":
        prefix = prefix[1:]
    if prefix and prefix[-1] in ("fw", "bw"):
        prefix = prefix[:-1]
    return " ".join(prefix) if prefix else "?"


def _named_row(rows: list[dict[str, typing.Any]], name: str) -> dict[str, typing.Any] | None:
    return next((r for r in rows if r["tensor_name"].split(":", 1)[-1].strip() == name), None)


_LM_HEAD_NAME = "head.output_weights"
_EMBEDDINGS_NAME = "embeddings.word_embeddings_weight"


def _print_summary(results: dict[str, list[dict[str, typing.Any]]]) -> None:
    sample = next(iter(results.values()))
    has_fw_logits = _named_row(sample, "head.logits") is not None
    has_bw_logits = _named_row(sample, "head.logits.grad") is not None
    has_bias = any(
        r["kind"] == "grad" and r["tensor_name"].split(":", 1)[-1].strip().endswith(".bias") for r in sample
    )
    # Each kind's aggregation columns are listed chronologically (left-to-right matches
    # the order tensors are logged). Logits show up via `output_hidden_states` on the
    # fw/bw boundary; weight gradients have no logits hook.
    fw_aggs = ("first", "median", "max") + (("logits",) if has_fw_logits else ()) + ("last",)
    bw_aggs = ("first",) + (("logits",) if has_bw_logits else ()) + ("median", "max", "last")
    grad_aggs = (
        ("lm_head", "linear_med", "linear_max", "norm_med", "norm_max")
        + (("bias_med", "bias_max") if has_bias else ())
        + ("embeddings",)
    )
    aggs_per_kind = {"fw": fw_aggs, "bw": bw_aggs, "grad": grad_aggs}
    for kind in ("fw", "bw", "grad"):
        _print_summary_table(results, kind, aggs_per_kind[kind])
    if _named_row(sample, _CHOSEN_LOGPROB_NAME) is not None:
        _print_chosen_logprob_summary(results)


def _print_chosen_logprob_summary(results: dict[str, list[dict[str, typing.Any]]]) -> None:
    rows_by_variant = {name: _named_row(rows, _CHOSEN_LOGPROB_NAME) for name, rows in results.items()}
    # log π(label) is the scalar that policy-gradient importance ratios depend on. Bias persists
    # under per-document averaging where RMS shrinks ~1/√T, so for RL stability it's the more
    # informative signal — surface it alongside RMS, slope and residual.
    rms_rel_decimals = _column_decimals((r["rms_rel"] for r in rows_by_variant.values()), default=3, max_decimals=5)
    bias_rel_decimals = _column_decimals((r["bias_rel"] for r in rows_by_variant.values()), default=3, max_decimals=5)
    resid_rel_decimals = _column_decimals(
        (r["residual_rms_rel"] for r in rows_by_variant.values()), default=3, max_decimals=5
    )
    name_width = max((len(name) for name in results), default=7) + 1
    cols = [
        ("RMS rel", lambda r: f"{r['rms_rel'] * 100:.{rms_rel_decimals}f}%"),
        ("Bias rel", lambda r: f"{r['bias_rel'] * 100:+.{bias_rel_decimals}f}%"),
        ("Resid rel", lambda r: f"{r['residual_rms_rel'] * 100:.{resid_rel_decimals}f}%"),
        ("Corr", lambda r: f"{r['correlation']:.5f}"),
        ("Slope", lambda r: f"{r['slope']:+.5f}"),
        ("Max abs", lambda r: f"{r['max_abs']:.4g}"),
        ("Scale", lambda r: f"{r['ref_scale']:.4g}"),
    ]
    widths = [max(len(label), max(len(fn(r)) for r in rows_by_variant.values())) for label, fn in cols]
    print(f"\n=== Summary: chosen_logprob (per-token) ===")
    header = f"{'Variant':<{name_width}}" + " ".join(
        f"{label:<{w}}" for (label, _), w in zip(cols, widths, strict=True)
    )
    print(header)
    print("-" * len(header))
    for name, row in rows_by_variant.items():
        cells = [fn(row) for _, fn in cols]
        print(f"{name:<{name_width}}" + " ".join(f"{c:<{w}}" for c, w in zip(cells, widths, strict=True)))


def _grad_category(tensor_name: str) -> str:
    name = tensor_name.split(":", 1)[-1].strip()
    if name.endswith(".bias"):
        return "bias"
    if ".norm_" in name or name.endswith(".norm.weight"):
        return "norm"
    return "linear"


def _print_summary_table(results: dict[str, list[dict[str, typing.Any]]], kind: str, aggs: tuple[str, ...]) -> None:
    sample = next(iter(results.values()))
    group = [r for r in sample if r["kind"] == kind]
    if not group:
        return
    endpoint_labels = {
        "first": _layer_name(group[0]["tensor_name"]),
        "last": _layer_name(group[-1]["tensor_name"]),
    }
    mid_labels = {
        "median": "mid med",
        "max": "mid max",
        "logits": "logits",
        "lm_head": "lm head",
        "embeddings": "embeddings",
        "linear_med": "linear med",
        "linear_max": "linear max",
        "norm_med": "norm med",
        "norm_max": "norm max",
        "bias_med": "bias med",
        "bias_max": "bias max",
    }

    def _label(agg: str) -> str:
        return endpoint_labels[agg] if agg in endpoint_labels else mid_labels[agg]

    name_width = max((len(name) for name in results), default=7) + 1
    cell_width = max(len(_label(a)) for a in aggs)
    cell_sep = " "
    raw: dict[str, dict[str, float | None]] = {}
    for name, rows in results.items():
        logits_fw = _named_row(rows, "head.logits")
        logits_bw = _named_row(rows, "head.logits.grad")
        logits_value = {
            "fw": logits_fw["rms_rel"] if logits_fw else float("nan"),
            "bw": logits_bw["rms_rel"] if logits_bw else float("nan"),
        }
        kind_rows = [r for r in rows if r["kind"] == kind]
        values = [r["rms_rel"] for r in kind_rows]
        if kind == "grad":
            decoder_rows = [r for r in kind_rows if r["tensor_name"].split(":", 1)[-1].strip().startswith("decoder.")]
            category_values: dict[str, list[float]] = {"linear": [], "norm": [], "bias": []}
            for r in decoder_rows:
                category_values[_grad_category(r["tensor_name"])].append(r["rms_rel"])
            lm_head_row = _named_row(kind_rows, _LM_HEAD_NAME)
            embeddings_row = _named_row(kind_rows, _EMBEDDINGS_NAME)
        else:
            category_values = {}
            lm_head_row = embeddings_row = None
        intermediate = values[1:-1] or values
        cells: dict[str, float | None] = {}
        for agg in aggs:
            if agg == "first":
                cells[agg] = values[0] if values else None
            elif agg == "last":
                cells[agg] = values[-1] if values else None
            elif agg == "logits":
                cells[agg] = logits_value[kind]
            elif agg == "lm_head":
                cells[agg] = lm_head_row["rms_rel"] if lm_head_row else None
            elif agg == "embeddings":
                cells[agg] = embeddings_row["rms_rel"] if embeddings_row else None
            elif "_" in agg and agg.split("_", 1)[0] in category_values:
                cat, stat = agg.split("_", 1)
                cat_values = category_values[cat]
                if not cat_values:
                    cells[agg] = None
                elif stat == "max":
                    cells[agg] = max(cat_values)
                else:
                    cells[agg] = statistics.median(cat_values)
            elif agg == "max":
                cells[agg] = max(intermediate) if intermediate else None
            else:
                cells[agg] = statistics.median(intermediate) if intermediate else None
        raw[name] = cells

    column_decimals = {
        agg: _column_decimals(cells[agg] for cells in raw.values() if cells[agg] is not None) for agg in aggs
    }
    if kind == "grad":
        subtitle = " (Relative %)"
    else:
        subtitle = " (Relative %; mid = excluding first/last)"
    print(f"\n=== Summary: {kind}{subtitle} ===")
    header = f"{'Variant':<{name_width}}" + cell_sep.join(f"{_label(a):<{cell_width}}" for a in aggs)
    print(header)
    print("-" * len(header))
    for name, cells in raw.items():
        formatted = [
            f"{cells[agg] * 100:.{column_decimals[agg]}f}%" if cells[agg] is not None else "n/a" for agg in aggs
        ]
        print(f"{name:<{name_width}}" + cell_sep.join(f"{c:<{cell_width}}" for c in formatted))


def _column_decimals(
    values: typing.Iterable[float], min_sig_figs: int = 2, default: int = 3, max_decimals: int | None = None
) -> int:
    # Keep the default precision, but bump up so the smallest non-zero value carries at least
    # `min_sig_figs` significant digits when formatted as percent. `max_decimals` caps the
    # bump so a single tiny noisy value doesn't widen the whole column.
    smallest = min((abs(v) * 100 for v in values if v != 0), default=None)
    if smallest is None or smallest >= 10 ** -(default - min_sig_figs + 1):
        result = default
    else:
        result = max(default, -math.floor(math.log10(smallest)) + min_sig_figs - 1)
    return min(result, max_decimals) if max_decimals is not None else result


def _display_group(row: dict[str, typing.Any]) -> str:
    # Map each row to one of "fw"/"bw"/"grad" for the per-variant table, independent
    # of `kind`: head.logits is a forward activation, head.logits.grad is a backward
    # quantity, parameter gradients are their own group.
    if row["kind"] == "grad":
        return "grad"
    if row["kind"] == "bw" or row["tensor_name"].endswith(".grad"):
        return "bw"
    return "fw"


def _classify(tensor_name: str) -> str:
    # Stage._log_layer_forward / _log_layer_backward produce "<module_name> fw[, mb=…]"
    # and "<module_name> bw[, mb=…]"; log_distributed_tensor may prefix the name
    # with "Global " and append a ": <description>" suffix when reconstructing a
    # tensor-parallel-global tensor. Per-parameter gradient logs come from
    # `Fsdp.log_shard(name="gradient", ...)` and are tagged "grad" so they appear
    # in the per-variant table but stay out of the fw/bw summary aggregation.
    # Other entries (e.g. `Global : head.logits`, `Global : head.logits.grad`) come
    # from the `_debug` / `output_hidden_states` path and are surfaced via dedicated
    # logits columns in the summary.
    if "gradient:" in tensor_name:
        return "grad"
    for kind in ("fw", "bw"):
        if f" {kind}:" in tensor_name or f" {kind}," in tensor_name or tensor_name.endswith(f" {kind}"):
            return kind
    return "other"


def _print_table(name: str, rows: list[dict[str, typing.Any]]) -> None:
    print(f"\n=== Variant: {name} ===")
    if not rows:
        print("(no matching tensors)")
        return
    name_fn = lambda r: f"{r['tensor_name'].split(':', 1)[-1].strip()} ({r['kind']})"
    name_width = max(len("Tensor"), max(len(name_fn(r)) for r in rows))
    # Adaptive precision for the relative column: bump decimals so small but real values
    # (typical for weight gradients) stay legible, capped at 5 to bound column width.
    relative_decimals = _column_decimals((r["rms_rel"] for r in rows), default=2, max_decimals=5)
    relative_fn = lambda r: f"{r['rms_rel'] * 100:.{relative_decimals}f}%"
    bias_decimals = _column_decimals((r["bias_rel"] for r in rows), default=2, max_decimals=5)
    bias_fn = lambda r: f"{r['bias_rel'] * 100:+.{bias_decimals}f}%"
    relative_width = max(len("Relative"), max(len(relative_fn(r)) for r in rows))
    bias_width = max(len("Bias"), max(len(bias_fn(r)) for r in rows))
    columns: list[tuple[str, int, typing.Callable[[dict[str, typing.Any]], str]]] = [
        ("Tensor", name_width, name_fn),
        ("Relative", relative_width, relative_fn),
        ("Bias", bias_width, bias_fn),
        ("Absolute", 10, lambda r: f"{r['rms_abs']:.4g}"),
        ("Max", 10, lambda r: f"{r['max_abs']:.4g}"),
        ("Scale", 10, lambda r: f"{r['ref_scale']:.4g}"),
    ]
    header = "  ".join(f"{title:<{width}}" for title, width, _ in columns)
    print(header)
    print("-" * len(header))
    # Display grouping (fw / bw / grad) separates the chronologically-interleaved
    # backward and reduce_gradients hooks. Independent of `kind` so the summary
    # aggregation isn't affected.
    groups = ("fw", "bw", "grad")
    grouped: dict[str, list[dict[str, typing.Any]]] = {g: [] for g in groups}
    for row in rows:
        grouped[_display_group(row)].append(row)
    first = True
    for group in groups:
        if not grouped[group]:
            continue
        if not first:
            print()
        first = False
        for row in grouped[group]:
            print("  ".join(f"{format_fn(row):<{width}}" for _, width, format_fn in columns))


if __name__ == "__main__":
    EvaluatePrecisionConfig.parse_and_run()
