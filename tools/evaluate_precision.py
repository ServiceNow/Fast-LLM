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


# Tensor-log verbosity level. 13 gives 2**(13-3)=1024 sampled values per tensor,
# matching the convention in the existing layer-comparison tests.
_LOG_LEVEL = 13
_REFERENCE_NAME = "reference"
_MODEL_TYPE = "gpt"


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
        default=1024,
        desc="Number of sampled values stored per logged tensor.",
        hint=FieldHint.feature,
    )
    micro_batch_size: int = Field(
        default=1,
        desc="Micro-batch size for the single forward+backward pass.",
        hint=FieldHint.feature,
    )
    sequence_length: int = Field(
        default=2048,
        desc="Sequence length (maximum document length) for the random input.",
        hint=FieldHint.feature,
    )
    data_path: pathlib.Path | None = Field(
        default=None,
        desc="If set, prepare a tokenized memmap dataset with advantages and `old_log_probabilities`"
        " at this path (using the test helper `_get_test_dataset`) and use it as the training"
        " input — required for policy-gradient losses like GSPO/GRPO. If unset, uses random tokens.",
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
        self._prepare_data()
        runs: dict[str, dict[str, typing.Any]] = {_REFERENCE_NAME: {}}
        runs.update(self.variants)
        for name, variant_overrides in runs.items():
            self._run_one(name, variant_overrides)

        ref_artifacts = self._artifact_path(_REFERENCE_NAME)
        results = {name: self._compare(ref_artifacts, self._artifact_path(name)) for name in self.variants}

        report_path = self.output_dir / "precision_report.json"
        report_path.write_text(json.dumps(results, indent=2))
        logger.info(f"Wrote report to {report_path}")

        for name, rows in results.items():
            _print_table(name, rows)
        _print_summary(results)

    def _prepare_data(self) -> None:
        if self.data_path is None:
            return
        if (self.data_path / "fast_llm_config.yaml").is_file():
            return
        # Couples `tools/` to `tests/utils/` for now — extract later if it sticks.
        from tests.utils.dataset import _get_test_dataset

        self.data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Preparing memmap dataset at {self.data_path}")
        _get_test_dataset(
            self.data_path,
            seed=42,
            has_grpo_data=True,
            max_vocab_size=self.model.base_model.embeddings.vocab_size,
        )

    def _artifact_path(self, name: str) -> pathlib.Path:
        return self.output_dir / name / "runs" / "0" / "artifacts"

    def _run_one(self, name: str, variant_overrides: dict[str, typing.Any]) -> None:
        # The trainer's Run picks the next `runs/<n>` subdir based on what already exists; wipe
        # any prior contents so each invocation lands in `runs/0` and stale artifacts can't be
        # read by `_artifact_path` below.
        experiment_dir = self.output_dir / name
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        # Base config: hardcoded training/optimizer/data/run skeleton plus the user's model/pretrained.
        # Forced fp32 on the reference baseline lives in here too so a variant can override it.
        base_dict: dict[str, typing.Any] = {
            "pretrained": self.pretrained.to_dict(),
            "model": self.model.to_dict(),
            "training": {
                "train_iters": 1,
                "num_workers": 0,
                "logs": {"interval": 1},
            },
            "optimizer": {
                "learning_rate": {"base": 0.0, "decay_style": "constant", "warmup_iterations": 0},
            },
            "data": {
                "datasets": {
                    "training": (
                        {"type": "file", "path": str(self.data_path / "fast_llm_config.yaml")}
                        if self.data_path is not None
                        else {"type": "random"}
                    )
                },
                "micro_batch_size": self.micro_batch_size,
                "maximum_document_length": self.sequence_length,
            },
            "run": {
                "experiment_dir": str((self.output_dir / name).resolve()),
                "tensor_logs": {"save": True, "show": False, "max_elements": self.num_samples},
            },
        }
        fp32_dtypes = {
            ("model", "distributed", "compute_dtype"): "float32",
            ("model", "distributed", "optimization_dtype"): "float32",
        }
        variant_updates = {tuple(key.split(".")): value for key, value in variant_overrides.items()}
        # Tool-required overrides win over variants — a variant must not silently disable tensor logging.
        tool_overrides = {
            ("model", "multi_stage", "debug_layer_outputs"): _LOG_LEVEL,
            ("model", "multi_stage", "debug_layer_gradients"): _LOG_LEVEL,
            # Capture the LM-head logits via the `output_hidden_states` mechanism: the head's
            # `_debug(logits, ...)` call matches this pattern and emits to `tensor_logs`.
            ("model", "multi_stage", "debug_hidden_states_log"): [r"head\.logits"],
        }
        logger.info(f"=== Running {name!r} ===")
        if variant_overrides:
            logger.info(f"Variant overrides: {variant_overrides}")
        trainer_class = TrainerConfig.get_subclass(_MODEL_TYPE)
        trainer_config = trainer_class.from_dict(base_dict, fp32_dtypes, variant_updates, tool_overrides)
        trainer_config.configure_logging()
        trainer_config._get_runnable()()

    def _compare(self, ref_path: pathlib.Path, test_path: pathlib.Path) -> list[dict[str, typing.Any]]:
        compare_config = CompareConfig()
        errors: list[str] = []
        ref_logs = compare_config._extract_tensor_logs(ref_path, errors)
        test_logs = compare_config._extract_tensor_logs(test_path, errors)
        for error in errors:
            logger.warning(error)
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


def _layer_name(tensor_name: str) -> str:
    # Stage hooks name tensors `Global <layer> fw: ...` / `Global <layer> bw: ...`;
    # extract the layer to use as a meaningful column label.
    prefix = tensor_name.split(":", 1)[0].strip().split()
    if prefix and prefix[0] == "Global":
        prefix = prefix[1:]
    if prefix and prefix[-1] in ("fw", "bw"):
        prefix = prefix[:-1]
    return " ".join(prefix) if prefix else "?"


def _named_row(rows: list[dict[str, typing.Any]], name: str) -> dict[str, typing.Any] | None:
    return next((r for r in rows if r["tensor_name"].split(":", 1)[-1].strip() == name), None)


def _print_summary(results: dict[str, list[dict[str, typing.Any]]]) -> None:
    # Per-pass labels for `first`/`last` come from the actual layer name on the matching row.
    sample = next(iter(results.values()))
    endpoint_labels: dict[tuple[str, str], str] = {
        ("fw", "first"): "first",
        ("fw", "last"): "last",
        ("bw", "first"): "first",
        ("bw", "last"): "last",
    }
    for kind in ("fw", "bw"):
        group = [r for r in sample if r["kind"] == kind]
        if group:
            endpoint_labels[(kind, "first")] = _layer_name(group[0]["tensor_name"])
            endpoint_labels[(kind, "last")] = _layer_name(group[-1]["tensor_name"])
    mid_labels = {"median": "mid med", "max": "mid max", "logits": "logits"}
    # Logits show up via `output_hidden_states` (`Global : head.logits` on the fw side and
    # `Global : head.logits.grad` on the bw side once the loss has computed dL/dlogits).
    # Each gets a dedicated column placed chronologically: end-of-fw and start-of-bw.
    has_fw_logits = _named_row(sample, "head.logits") is not None
    has_bw_logits = _named_row(sample, "head.logits.grad") is not None
    fw_aggs = ("first", "median", "max") + (("logits",) if has_fw_logits else ()) + ("last",)
    bw_aggs = ("first",) + (("logits",) if has_bw_logits else ()) + ("median", "max", "last")
    aggs_per_kind = {"fw": fw_aggs, "bw": bw_aggs}
    kinds = ("fw", "bw")

    def _label(kind: str, agg: str) -> str:
        return endpoint_labels[(kind, agg)] if agg in ("first", "last") else mid_labels[agg]

    name_width = max((len(name) for name in results), default=7) + 1
    cell_width = max(len(_label(k, a)) for k in kinds for a in aggs_per_kind[k])
    cell_sep = " "
    group_sep = "   "
    group_widths = {
        kind: len(cell_sep.join(f"{_label(kind, a):<{cell_width}}" for a in aggs_per_kind[kind])) for kind in kinds
    }
    print("\n=== Summary (Relative %; mid = excluding first/last) ===")
    top = f"{'':<{name_width}}" + group_sep.join(f"{kind:^{group_widths[kind]}}" for kind in kinds)
    bottom = f"{'Variant':<{name_width}}" + group_sep.join(
        cell_sep.join(f"{_label(kind, a):<{cell_width}}" for a in aggs_per_kind[kind]) for kind in kinds
    )
    print(top)
    print(bottom)
    print("-" * len(bottom))
    # Collect raw values first so we can pick a per-column decimal count: keep the previous
    # .3f% default, but bump up just enough to give every cell in a column ≥ 2 sig figs.
    raw: dict[str, dict[tuple[str, str], float | None]] = {}
    for name, rows in results.items():
        logits_fw = _named_row(rows, "head.logits")
        logits_bw = _named_row(rows, "head.logits.grad")
        logits_value = {
            "fw": logits_fw["rms_rel"] if logits_fw else float("nan"),
            "bw": logits_bw["rms_rel"] if logits_bw else float("nan"),
        }
        cells: dict[tuple[str, str], float | None] = {}
        for kind in kinds:
            values = [r["rms_rel"] for r in rows if r["kind"] == kind]
            intermediate = values[1:-1] or values
            for agg in aggs_per_kind[kind]:
                if not values:
                    cells[(kind, agg)] = None
                    continue
                if agg == "first":
                    cells[(kind, agg)] = values[0]
                elif agg == "last":
                    cells[(kind, agg)] = values[-1]
                elif agg == "logits":
                    cells[(kind, agg)] = logits_value[kind]
                elif agg == "max":
                    cells[(kind, agg)] = max(intermediate)
                else:
                    cells[(kind, agg)] = statistics.median(intermediate)
        raw[name] = cells

    column_decimals: dict[tuple[str, str], int] = {}
    for kind in kinds:
        for agg in aggs_per_kind[kind]:
            column_decimals[(kind, agg)] = _column_decimals(
                cells[(kind, agg)] for cells in raw.values() if cells[(kind, agg)] is not None
            )
    for name, cells in raw.items():
        groups = []
        for kind in kinds:
            formatted = []
            for agg in aggs_per_kind[kind]:
                value = cells[(kind, agg)]
                if value is None:
                    formatted.append("n/a")
                else:
                    formatted.append(f"{value * 100:.{column_decimals[(kind, agg)]}f}%")
            groups.append(cell_sep.join(f"{c:<{cell_width}}" for c in formatted))
        print(f"{name:<{name_width}}" + group_sep.join(groups))


def _column_decimals(values: typing.Iterable[float], min_sig_figs: int = 2, default: int = 3) -> int:
    # Keep the previous default precision, but bump up so the smallest non-zero value
    # carries at least `min_sig_figs` significant digits when formatted as percent.
    smallest = min((abs(v) * 100 for v in values if v != 0), default=None)
    if smallest is None or smallest >= 10 ** -(default - min_sig_figs + 1):
        return default
    return max(default, -math.floor(math.log10(smallest)) + min_sig_figs - 1)


def _classify(tensor_name: str) -> str:
    # Stage._log_layer_forward / _log_layer_backward produce "<module_name> fw[, mb=…]"
    # and "<module_name> bw[, mb=…]"; log_distributed_tensor may prefix the name
    # with "Global " and append a ": <description>" suffix when reconstructing a
    # tensor-parallel-global tensor. Other entries (e.g. `Global : head.logits`,
    # `Global : head.logits.grad`) come from the `_debug` / `output_hidden_states` path
    # and are surfaced via dedicated logits columns in the summary.
    for kind in ("fw", "bw"):
        if f" {kind}:" in tensor_name or f" {kind}," in tensor_name or tensor_name.endswith(f" {kind}"):
            return kind
    return "other"


def _print_table(name: str, rows: list[dict[str, typing.Any]]) -> None:
    print(f"\n=== Variant: {name} ===")
    if not rows:
        print("(no matching tensors)")
        return
    columns: list[tuple[str, int, typing.Callable[[dict[str, typing.Any]], str]]] = [
        ("Tensor", 26, lambda r: f"{r['tensor_name'].split(':', 1)[-1].strip()} ({r['kind']})"),
        ("Relative", 8, lambda r: f"{r['rms_rel'] * 100:.2f}%"),
        ("Absolute", 10, lambda r: f"{r['rms_abs']:.4g}"),
        ("Max", 10, lambda r: f"{r['max_abs']:.4g}"),
        ("Scale", 10, lambda r: f"{r['ref_scale']:.4g}"),
    ]
    header = "  ".join(f"{title:<{width}}" for title, width, _ in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(f"{format_fn(row):<{width}}" for _, width, format_fn in columns))


if __name__ == "__main__":
    EvaluatePrecisionConfig.parse_and_run()
