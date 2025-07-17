import dataclasses
import functools
import pathlib
import typing

import pytest

from fast_llm.engine.checkpoint.config import CheckpointFormat, DistributedCheckpointFormat, FastLLMCheckpointFormat
from tests.utils.model_configs import ModelTestingConfig


@dataclasses.dataclass(kw_only=True)
class DistributedSaveLoadConfig:
    load_path: pathlib.Path | str
    load_format: str
    save_path: pathlib.Path | str
    distributed: dict[str, typing.Any]
    num_gpus: int = 2

    def resolve(self, base_path: pathlib.Path, model_testing_config: ModelTestingConfig) -> typing.Self:
        return dataclasses.replace(
            self,
            load_path=base_path
            / str(self.load_path).format(checkpoint_format=model_testing_config.checkpoint_format.name),
            load_format=self.load_format.format(checkpoint_format=model_testing_config.checkpoint_format.name),
            save_path=base_path
            / str(self.save_path).format(checkpoint_format=model_testing_config.checkpoint_format.name),
        )

    @property
    def name(self) -> str:
        return pathlib.Path(self.save_path).name


def do_get_convert_path(
    to: type[CheckpointFormat] | str | None = None,
    from_: type[CheckpointFormat] | str | None = None,
    *,
    base_path: pathlib.Path,
) -> pathlib.Path:
    if to is None or from_ is None:
        return base_path / "checkpoint_and_eval" / "checkpoint" / "2"
    return (
        base_path
        / "convert_model"
        / f"{to.name if isinstance(to,type) else to}_from_{from_.name if isinstance(from_,type) else from_}"
    )


@pytest.fixture(scope="module")
def get_convert_path(run_test_script_base_path):
    return functools.partial(do_get_convert_path, base_path=run_test_script_base_path)


_DISTRIBUTED_SAVE_LOAD_CONFIGS = []


for pretrained_format, pretrained_path in (
    (
        DistributedCheckpointFormat.name,
        do_get_convert_path(DistributedCheckpointFormat.name, "{checkpoint_format}", base_path=pathlib.Path()),
    ),
    (
        FastLLMCheckpointFormat.name,
        do_get_convert_path(FastLLMCheckpointFormat.name, "{checkpoint_format}", base_path=pathlib.Path()),
    ),
    (
        "{checkpoint_format}",
        do_get_convert_path("{checkpoint_format}", DistributedCheckpointFormat.name, base_path=pathlib.Path()),
    ),
):
    _DISTRIBUTED_SAVE_LOAD_CONFIGS.extend(
        [
            DistributedSaveLoadConfig(
                load_path=pretrained_path,
                load_format=pretrained_format,
                save_path=f"load_{pretrained_format}_in_dp2",
                distributed={},
            ),
            DistributedSaveLoadConfig(
                load_path=pretrained_path,
                load_format=pretrained_format,
                save_path=f"load_{pretrained_format}_in_tp2",
                distributed={"tensor_parallel": 2},
            ),
            DistributedSaveLoadConfig(
                load_path=pretrained_path,
                load_format=pretrained_format,
                save_path=f"load_{pretrained_format}_in_stp2",
                distributed={"tensor_parallel": 2, "sequence_tensor_parallel": True},
            ),
            DistributedSaveLoadConfig(
                load_path=pretrained_path,
                load_format=pretrained_format,
                save_path=f"load_{pretrained_format}_in_pp2",
                distributed={"pipeline_parallel": 2},
            ),
        ]
    )

_DISTRIBUTED_SAVE_LOAD_CONFIGS.extend(
    [
        DistributedSaveLoadConfig(
            load_path=f"load_{DistributedCheckpointFormat.name}_in_dp2/{DistributedCheckpointFormat.name}",
            load_format=DistributedCheckpointFormat.name,
            save_path="load_dp2_in_stp2",
            distributed={"tensor_parallel": 2, "sequence_tensor_parallel": True},
        ),
        DistributedSaveLoadConfig(
            load_path=f"load_{DistributedCheckpointFormat.name}_in_stp2/{DistributedCheckpointFormat.name}",
            load_format=DistributedCheckpointFormat.name,
            save_path="load_stp2_in_dp2",
            distributed={},
        ),
        DistributedSaveLoadConfig(
            load_path=f"load_{DistributedCheckpointFormat.name}_in_tp2/{DistributedCheckpointFormat.name}",
            load_format=DistributedCheckpointFormat.name,
            save_path="load_tp2_in_pp2",
            distributed={"pipeline_parallel": 2},
        ),
        DistributedSaveLoadConfig(
            load_path=f"load_{DistributedCheckpointFormat.name}_in_pp2/{DistributedCheckpointFormat.name}",
            load_format=DistributedCheckpointFormat.name,
            save_path="load_pp2_in_tp2",
            distributed={"tensor_parallel": 2},
        ),
    ]
)

# TODO: Name isn't formated.
DISTRIBUTED_SAVE_LOAD_CONFIGS: dict[str, DistributedSaveLoadConfig] = {
    config.name: config for config in _DISTRIBUTED_SAVE_LOAD_CONFIGS
}


@pytest.fixture(scope="module", params=DISTRIBUTED_SAVE_LOAD_CONFIGS)
def distributed_save_load_config(request):
    return DISTRIBUTED_SAVE_LOAD_CONFIGS[request.param]


@pytest.fixture(scope="module", params=[name for name in DISTRIBUTED_SAVE_LOAD_CONFIGS if "pp2" not in name])
def distributed_save_load_config_non_pp(request):
    return DISTRIBUTED_SAVE_LOAD_CONFIGS[request.param]
