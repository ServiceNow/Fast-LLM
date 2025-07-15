import argparse
import dataclasses
import pathlib
import typing

import torch

_TENSOR_LOG_PREFIX = "tensor_logs_"


def _compare_pattern(pattern: typing.Iterable[str] | str | None, name: str):
    # TODO: Regex?
    return (
        True
        if pattern is None
        else pattern in name if isinstance(pattern, str) else any(pattern_ in name for pattern_ in pattern)
    )


@dataclasses.dataclass()
class CompareConfig:
    rms_eps: float = 1e-4
    rms_rel_tolerance: float = 3e-3
    rms_abs_tolerance: float = 5e-4
    max_rel_tolerance: float = 1.5e-2
    max_abs_tolerance: float = 5e-3
    # Test tensors are scaled by this amount (ex. gradient scaling). Unscale (divide) them before comparison.
    scale: float = 1.0
    show_samples: int = 10
    ignore_tensors: bool = False
    ignore_duplicates: bool = False
    # Use a different config for specific step and/or tensor names. First match is used.
    sub_configs: dict[tuple[typing.Iterable[str] | str | None, typing.Iterable[str] | str | None], "CompareConfig"] = (
        dataclasses.field(default_factory=dict)
    )

    def rescale(self, factor: float) -> typing.Self:
        # Scale all tolerances by this factor.
        if factor == 1.0:
            return self
        return dataclasses.replace(
            self,
            rms_eps=self.rms_eps * factor,
            rms_rel_tolerance=self.rms_rel_tolerance * factor,
            rms_abs_tolerance=self.rms_abs_tolerance * factor,
            max_rel_tolerance=self.max_rel_tolerance * factor,
            max_abs_tolerance=self.max_abs_tolerance * factor,
            sub_configs={key: sub_config.rescale(factor) for key, sub_config in self.sub_configs.items()},
        )

    def _get_sub_config(self, step_name: str, tensor_name: str) -> typing.Self:
        for (step_key, name_key), sub_config in self.sub_configs.items():
            if _compare_pattern(step_key, step_name) and _compare_pattern(name_key, tensor_name):
                return sub_config._get_sub_config(step_name, tensor_name)
        return self

    def _extract_tensor_logs(self, artifact_path: pathlib.Path, errors):
        tensor_logs = {}
        for rank_path in sorted(artifact_path.iterdir()):
            for p in rank_path.iterdir():
                if p.name.startswith(_TENSOR_LOG_PREFIX) and p.suffix == ".pt":
                    step_name = p.stem[len(_TENSOR_LOG_PREFIX) :]
                    for step_log in torch.load(p):
                        tensor_name = step_log["name"]
                        sub_config = self._get_sub_config(step_name, tensor_name)
                        if not sub_config.ignore_tensors:
                            if step_name not in tensor_logs:
                                tensor_logs[step_name] = {}
                            if (
                                tensor_name in (tensor_step_logs := tensor_logs[step_name])
                                and not sub_config.ignore_duplicates
                            ):
                                errors.append(f"Duplicate tensor log in step {step_name}: {tensor_name}")
                            tensor_step_logs[tensor_name] = step_log
        return tensor_logs

    def _compare_dict_keys(self, dict_ref, dict_test, errors, name):
        keys_ref = set(dict_ref)
        keys_test = set(dict_test)
        if keys_ref != keys_test:
            errors.append(
                f">>>> {name} do not match. Missing = {keys_ref - keys_test}, extra = {keys_test - keys_ref}."
            )

        # Avoid set to preserve ordering.
        return [key for key in dict_test if key in dict_ref]

    def compare_tensors(self, tensor_ref, tensor_test, errors, step_name, tensor_name):
        sub_config = self._get_sub_config(step_name, tensor_name)
        if tensor_ref["shape"] != tensor_test["shape"]:
            errors.append(
                "\n".join(
                    [
                        f">>>> [{step_name}] Incompatible shape for tensor {tensor_name}: {tensor_test['shape']}!={tensor_ref['shape']}"
                    ]
                )
            )
            return
        if tensor_ref["step"] != tensor_test["step"]:
            errors.append(
                "\n".join(
                    [
                        f">>>> [{step_name}] Incompatible sampling rate for tensor {tensor_name}: {tensor_test['step']}!={tensor_ref['step']}"
                    ]
                )
            )
            return

        samples_ref = tensor_ref["samples"].flatten().float()
        samples_test = tensor_test["samples"].flatten().float()
        if sub_config.scale != 1.0:
            samples_test = samples_test / sub_config.scale
        scale_unreg = (samples_ref**2).mean() ** 0.5
        rms_scale = (scale_unreg**2 + sub_config.rms_eps**2) ** 0.5
        rms = ((samples_ref - samples_test) ** 2).mean() ** 0.5
        max_diff = (samples_ref - samples_test).abs().max()

        tensor_errors = []

        if rms > sub_config.rms_abs_tolerance:
            tensor_errors.append(f"  * RMS diff absolute = {rms} > {sub_config.rms_abs_tolerance}")

        if rms / rms_scale > sub_config.rms_rel_tolerance:
            tensor_errors.append(
                f"  * RMS diff scaled = {rms / rms_scale} > {sub_config.rms_rel_tolerance} (scale={rms_scale}, unregularized={scale_unreg})"
            )

        if max_diff > sub_config.max_abs_tolerance:
            tensor_errors.append(f"  * Max diff absolute = {max_diff} > {sub_config.max_abs_tolerance}")

        if max_diff / rms_scale > sub_config.max_rel_tolerance:
            tensor_errors.append(
                f"  * Max diff scaled = {max_diff / rms_scale} > {sub_config.max_rel_tolerance} (scale={rms_scale}, unregularized={scale_unreg})"
            )

        if tensor_errors:
            tensor_errors.extend(
                [
                    f"  Test samples: " + "".join(f"{x:12.4e}" for x in samples_test[: self.show_samples].tolist()),
                    f"  Ref samples:  " + "".join(f"{x:12.4e}" for x in samples_ref[: self.show_samples].tolist()),
                ]
            )
            errors.append("\n".join([f">>>> [{step_name}] Excessive diff for tensor {tensor_name}:"] + tensor_errors))

    def _compare_tensor_logs(
        self,
        artifact_path_ref: pathlib.Path,
        artifact_path_test: pathlib.Path,
    ):
        errors = []

        logs_ref = self._extract_tensor_logs(artifact_path_ref, errors)
        logs_test = self._extract_tensor_logs(artifact_path_test, errors)

        for step_key in sorted(self._compare_dict_keys(logs_ref, logs_test, errors, "Logged steps")):
            step_logs_ref = logs_ref[step_key]
            step_logs_test = logs_test[step_key]

            for tensor_key in self._compare_dict_keys(
                step_logs_ref, step_logs_test, errors=errors, name=f"[{step_key}] Tensor keys"
            ):
                self.compare_tensors(
                    step_logs_ref[tensor_key],
                    step_logs_test[tensor_key],
                    errors,
                    step_key,
                    tensor_key,
                )

        return errors

    def compare_tensor_logs(
        self,
        artifact_path_ref: pathlib.Path,
        artifact_path_test: pathlib.Path,
    ):
        print(f'Comparing tensor logs in "{artifact_path_test}" with reference logs "{artifact_path_ref}"')
        errors = self._compare_tensor_logs(artifact_path_ref, artifact_path_test)
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
    CompareConfig().compare_tensor_logs(args.path_ref, args.path_test)
