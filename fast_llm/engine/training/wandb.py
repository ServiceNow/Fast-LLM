import os
import pathlib

import yaml

from fast_llm.config import Config
from fast_llm.engine.config_utils.run import Run
from fast_llm.engine.training.config import WandbConfig


class Wandb:
    def __init__(self, config: WandbConfig, run: Run, experiment_config: Config):
        self._config = config
        self._is_setup = True
        self._run = run
        if self._config.entity_name is not None and self._run.is_main_rank:
            import wandb.sdk.lib.runid

            # Wandb login from file
            api_key_path = os.environ.get("WANDB_API_KEY_PATH")
            if api_key_path:
                os.environ["WANDB_API_KEY"] = pathlib.Path(api_key_path).open("r").read().strip()
            wandb_path = (
                None
                if self._run.experiment_directory is None
                else self._run.experiment_directory / "wandb_config.yaml"
            )
            if wandb_path is not None and wandb_path.is_file():
                wandb_config = yaml.safe_load(wandb_path.open("r"))
            else:
                wandb_config = {
                    "id": wandb.sdk.lib.runid.generate_id(16),
                    "project": self._config.project_name,
                    "name": self._run.experiment_name,
                    "entity": self._config.entity_name,
                    "group": self._config.group_name,
                    "save_code": False,
                    "resume": "allow",
                }
                if wandb_path is not None:
                    yaml.safe_dump(wandb_config, wandb_path.open("w"))
            # TODO: Does wandb work with nested configs?
            self._wandb = wandb.init(config=experiment_config.to_dict(), **wandb_config)
        else:
            self._wandb = None

    def log_metrics(self, completed_steps: int, metrics: dict[str, dict[str, float | int]]) -> None:
        # Note: metrics modified in-place
        if self._wandb is not None:
            import wandb

            wandb.log(metrics, step=completed_steps)  # noqa

    def alert(self, title, text, level="INFO", wait=0.001) -> None:
        if self._wandb is not None and self._config.alert.post_alerts:
            pass

            self._wandb.alert(  # noqa
                title=title() if callable(title) else title,
                text=f"[{self._config.project_name}/{self._run.experiment_name}, run {self._run.index}]"
                f" {text() if callable(text) else text}",
                level=level,
                wait_duration=wait,
            )

    def __enter__(self) -> "Wandb":
        self.alert(f"Run started!", "", "ERROR")
        return self

    def __exit__(self, exc_type, exc_val: OSError, exc_tb):
        if exc_val:
            self.alert(f"Run crashed!", (lambda: ", ".join(exc_val.args)), "ERROR")
        else:
            self.alert(f"Run ended!", "", "INFO")
