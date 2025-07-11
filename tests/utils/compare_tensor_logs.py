import argparse
import dataclasses
import pathlib
import typing
import warnings

import torch

_TENSOR_LOG_PREFIX = "tensor_logs_"


@dataclasses.dataclass()
class CompareConfig:
    rms_eps: float = 1e-3
    rms_rel_tolerance: float = 3e-2
    rms_abs_tolerance: float = 5e-3
    max_rel_tolerance: float = 1.5e-1
    max_abs_tolerance: float = 5e-2
    show_samples: int = 10
    ignore_tensors: list[str] = dataclasses.field(default_factory=list)
    ignore_duplicates: list[str] = dataclasses.field(default_factory=list)


def extract_tensor_logs(
    artifact_path: pathlib.Path, errors, config: CompareConfig, artifacts: typing.Sequence[str] | None = None
):
    tensor_logs = {}
    ignore_keys = set()
    for rank_path in sorted(artifact_path.iterdir()):
        for p in rank_path.iterdir():
            if p.name.startswith(_TENSOR_LOG_PREFIX) and p.suffix == ".pt":
                step_name = p.stem[len(_TENSOR_LOG_PREFIX) :]
                if artifacts is not None and step_name not in artifacts:
                    continue
                step_logs = torch.load(p)
                if step_name not in tensor_logs:
                    tensor_logs[step_name] = {}
                for step_log in step_logs:
                    name = step_log["name"]
                    if any(ignore_name in name for ignore_name in config.ignore_tensors):
                        ignore_keys.add(name)
                    else:
                        if name in tensor_logs[step_name] and not any(
                            ignore_name in name for ignore_name in config.ignore_duplicates
                        ):
                            errors.append(f"Duplicate tensor log in step {step_name}: {name}")
                        tensor_logs[step_name][name] = step_log
    if ignore_keys:
        warnings.warn(f"Ignoring keys in {artifact_path}: {ignore_keys}")
    return tensor_logs


def compare_dict_keys(dict_ref, dict_test, errors, name):
    keys_ref = set(dict_ref)
    keys_test = set(dict_test)
    if keys_ref != keys_test:
        errors.append(f">>>> {name} do not match. Missing = {keys_ref-keys_test}, extra = {keys_test-keys_ref}.")

    # Avoid set to preserve ordering.
    return [key for key in dict_test if key in dict_ref]


def compare_logged_tensor(tensor_ref, tensor_test, errors, step, name, config: CompareConfig):
    if tensor_ref["shape"] != tensor_test["shape"]:
        errors.append(
            "\n".join(
                [f">>>> [{step}] Incompatible shape for tensor {name}: {tensor_test['shape']}!={tensor_ref['shape']}"]
            )
        )
        return
    if tensor_ref["step"] != tensor_test["step"]:
        errors.append(
            "\n".join(
                [
                    f">>>> [{step}] Incompatible sampling rate for tensor {name}: {tensor_test['step']}!={tensor_ref['step']}"
                ]
            )
        )
        return

    samples_ref = tensor_ref["samples"].flatten().float()
    samples_test = tensor_test["samples"].flatten().float()
    scale_unreg = (samples_ref**2).mean() ** 0.5
    rms_scale = (scale_unreg**2 + config.rms_eps**2) ** 0.5
    rms = ((samples_ref - samples_test) ** 2).mean() ** 0.5
    max_diff = (samples_ref - samples_test).abs().max()

    tensor_errors = []

    if rms > config.rms_abs_tolerance:
        tensor_errors.append(f"  * RMS diff absolute = {rms} > {config.rms_abs_tolerance}")

    if rms / rms_scale > config.rms_rel_tolerance:
        tensor_errors.append(
            f"  * RMS diff scaled = {rms/rms_scale} > {config.rms_rel_tolerance} (scale={rms_scale}, unregularized={scale_unreg})"
        )

    if max_diff > config.max_abs_tolerance:
        tensor_errors.append(f"  * Max diff absolute = {max_diff} > {config.max_abs_tolerance}")

    if max_diff / rms_scale > config.max_rel_tolerance:
        tensor_errors.append(
            f"  * Max diff scaled = {max_diff/rms_scale} > {config.max_rel_tolerance} (scale={rms_scale}, unregularized={scale_unreg})"
        )

    if tensor_errors:
        tensor_errors.extend(
            [
                f"  Test samples: " + "".join(f"{x:12.4e}" for x in samples_test[: config.show_samples].tolist()),
                f"  Ref samples:  " + "".join(f"{x:12.4e}" for x in samples_ref[: config.show_samples].tolist()),
            ]
        )
        errors.append("\n".join([f">>>> [{step}] Excessive diff for tensor {name}:"] + tensor_errors))


def compare_tensor_logs_base(
    artifact_path_ref: pathlib.Path,
    artifact_path_test: pathlib.Path,
    config: CompareConfig | None = None,
    artifacts: typing.Sequence[str] | None = None,
):
    errors = []

    if config is None:
        config = CompareConfig()

    logs_ref = extract_tensor_logs(artifact_path_ref, errors, config=config, artifacts=artifacts)
    logs_test = extract_tensor_logs(artifact_path_test, errors, config=config, artifacts=artifacts)

    for step_key in sorted(compare_dict_keys(logs_ref, logs_test, errors, "Logged steps")):
        step_logs_ref = logs_ref[step_key]
        step_logs_test = logs_test[step_key]

        for tensor_key in compare_dict_keys(
            step_logs_ref, step_logs_test, errors=errors, name=f"[{step_key}] Tensor keys"
        ):
            compare_logged_tensor(
                step_logs_ref[tensor_key],
                step_logs_test[tensor_key],
                errors,
                step_key,
                tensor_key,
                config,
            )

    return errors


def compare_tensor_logs(
    artifact_path_ref: pathlib.Path,
    artifact_path_test: pathlib.Path,
    config: CompareConfig | None = None,
    artifacts: typing.Sequence[str] | None = None,
):
    print(f'Comparing tensor logs in "{artifact_path_test}" with reference logs "{artifact_path_ref}"')
    errors = compare_tensor_logs_base(artifact_path_ref, artifact_path_test, config, artifacts)
    if errors:
        for error in errors:
            print(error)
        raise ValueError(f"Comparison failed ({len(errors)} errors)")
    else:
        print("Comparison succeeded!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_ref", type=pathlib.Path)
    parser.add_argument("path_test", type=pathlib.Path)
    args = parser.parse_args()
    compare_tensor_logs(args.path_ref, args.path_test)
