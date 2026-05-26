import json
import logging
import pathlib
import typing

import yaml

from fast_llm.config import Field, FieldHint, config_class
from fast_llm.engine.config_utils.compare_tensor_logs import CompareConfig
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.training.config import TrainerConfig

# Populate the trainer dynamic-type registry.
import fast_llm.data.auto  # noqa: F401  # isort:skip
import fast_llm.engine.checkpoint.convert  # noqa: F401  # isort:skip
import fast_llm.models.auto  # noqa: F401  # isort:skip

logger = logging.getLogger(__name__)


# Tensor-log verbosity level. 13 gives 2**(13-3)=1024 sampled values per tensor,
# matching the convention in the existing layer-comparison tests.
_LOG_LEVEL = 13
_REFERENCE_NAME = "reference"


@config_class()
class EvaluatePrecisionConfig(RunnableConfig):
    training_config: pathlib.Path = Field(
        desc="Path to a Fast-LLM training YAML serving as the fp32 reference configuration.",
        hint=FieldHint.core,
    )
    model_type: str = Field(
        desc="Trainer dynamic-type name (e.g. 'gpt') used to dispatch to the right TrainerConfig subclass.",
        hint=FieldHint.core,
    )
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

    def _validate(self) -> None:
        super()._validate()
        assert self.training_config.is_file(), f"Training config not found: {self.training_config}"
        assert _REFERENCE_NAME not in self.variants, f"'{_REFERENCE_NAME}' is reserved for the fp32 baseline."
        for name, overrides in self.variants.items():
            assert isinstance(overrides, dict) and all(
                isinstance(k, str) for k in overrides
            ), f"Variant {name!r} must be a flat dict of dotted-path string keys."

    def run(self) -> None:
        base_dict = yaml.safe_load(self.training_config.read_text())
        for field_name in ("compute_dtype", "optimization_dtype"):
            current = _get_nested(base_dict, ("model", "distributed", field_name))
            if current is not None and DataType(current) is not DataType.float32:
                logger.warning(
                    f"Base config sets model.distributed.{field_name}={current!r};"
                    f" overriding to float32 for the reference run."
                )

        runs: dict[str, dict[str, typing.Any]] = {_REFERENCE_NAME: {}}
        runs.update(self.variants)
        for name, variant_overrides in runs.items():
            self._run_one(name, variant_overrides)

        ref_artifacts = self._artifact_path(_REFERENCE_NAME)
        results = {name: self._compare(ref_artifacts, self._artifact_path(name)) for name in self.variants}

        report_path = self.output_dir / "precision_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(results, indent=2))
        logger.info(f"Wrote report to {report_path}")

        for name, rows in results.items():
            _print_table(name, rows)

    def _artifact_path(self, name: str) -> pathlib.Path:
        return self.output_dir / name / "runs" / "0" / "artifacts"

    def _run_one(self, name: str, variant_overrides: dict[str, typing.Any]) -> None:
        experiment_dir = (self.output_dir / name).resolve()
        forced_fp32 = {
            "model.distributed.compute_dtype": "float32",
            "model.distributed.optimization_dtype": "float32",
        }
        tool_overrides = {
            "training.train_iters": 1,
            "training.checkpoint.interval": None,
            "run.tensor_logs.save": True,
            "run.tensor_logs.show": False,
            "run.tensor_logs.max_elements": self.num_samples,
            "run.experiment_dir": str(experiment_dir),
            "model.multi_stage.debug_layer_outputs": _LOG_LEVEL,
            "model.multi_stage.debug_layer_gradients": _LOG_LEVEL,
        }
        # Compose: forced fp32 first so a variant can override it (e.g. compute_dtype=bfloat16);
        # tool overrides last so logging and single-iteration mode always win.
        combined = {**forced_fp32, **variant_overrides, **tool_overrides}
        cli_overrides = [f"{key}={yaml.safe_dump(value).strip()}" for key, value in combined.items()]
        logger.info(f"=== Running {name!r} ===")
        if variant_overrides:
            logger.info(f"Variant overrides: {variant_overrides}")
        TrainerConfig.parse_and_run([self.model_type, "-c", str(self.training_config), *cli_overrides])

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


def _classify(tensor_name: str) -> str:
    # Stage._log_layer_forward / _log_layer_backward produce "<module_name> fw[, mb=…]"
    # and "<module_name> bw[, mb=…]"; log_distributed_tensor may prefix the name
    # with "Global " and append a ": <description>" suffix when reconstructing a
    # tensor-parallel-global tensor.
    for kind in ("fw", "bw"):
        if f" {kind}:" in tensor_name or f" {kind}," in tensor_name or tensor_name.endswith(f" {kind}"):
            return kind
    return "other"


def _get_nested(d: typing.Any, keys: tuple[str, ...]) -> typing.Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _print_table(name: str, rows: list[dict[str, typing.Any]]) -> None:
    print(f"\n=== Variant: {name} ===")
    if not rows:
        print("(no matching tensors)")
        return
    columns = [
        ("step", "step", 6),
        ("kind", "kind", 6),
        ("tensor_name", "tensor", 48),
        ("shape", "shape", 22),
        ("ref_scale", "ref_scale", 12),
        ("rms_abs", "rms_abs", 12),
        ("rms_rel", "rms_rel", 12),
        ("max_abs", "max_abs", 12),
        ("max_rel", "max_rel", 12),
    ]
    header = "  ".join(f"{title:<{width}}" for _, title, width in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        parts = []
        for key, _, width in columns:
            value = row[key]
            if isinstance(value, float):
                cell = f"{value:.4e}"
            elif isinstance(value, list):
                cell = "x".join(str(x) for x in value)
            else:
                cell = str(value)
            parts.append(f"{cell:<{width}}")
        print("  ".join(parts))


if __name__ == "__main__":
    EvaluatePrecisionConfig.parse_and_run()
