import argparse
import pathlib
import urllib.parse

import requests
import yaml

from fast_llm.config import NoAutoValidate
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.models.auto import trainer_registry
from fast_llm.utils import Assert


def load_url(config_url: str, config_auth_token_file: pathlib.Path | None = None):
    """
    Read a config from a URL, typically a config file hosted on GitHub.
    """

    headers = {"Accept": "application/vnd.github.v3.raw"}
    if config_auth_token_file is not None:
        config_auth_token = config_auth_token_file.open("r").read().strip()
        with open(config_auth_token_file) as f:
            headers["Authorization"] = f"token {config_auth_token}"
    response = requests.get(config_url, headers=headers)
    if response.status_code == 200:
        return response
    else:
        if isinstance(response.reason, bytes):
            try:
                reason = response.reason.decode("utf-8")
            except UnicodeDecodeError:
                reason = response.reason.decode("iso-8859-1")
        else:
            reason = response.reason
        raise ValueError(f"Failed to fetch config from {config_url}: {response.status_code}, {reason}")


class ConfigSource:
    file = "file"
    url = "url"
    args = "args"
    auto = "auto"


def train(args=None):
    # TODO: Generalize this for use in other tools, eg. convert?
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "model_type",
        choices=trainer_registry.keys(),
        help="The Fast-LLM model type to use. Must be defined in the trainer registry in `fast_llm.models.auto`.",
    )
    parser.add_argument(
        "-v",
        "--validate",
        dest="do_run",
        action="store_false",
        help="Validate the config without running the actual experiment.",
    )
    parser.add_argument(
        "-s",
        "--source",
        choices=(ConfigSource.file, ConfigSource.url, ConfigSource.args, ConfigSource.auto),
        default=ConfigSource.auto,
        help="Validate the config without running the actual experiment.",
    )
    parser.add_argument(
        # TODO: remove --config_url once changed in fml-ops
        "-c",
        "--config",
        "--config_url",
        help="The configuration file or url.",
    )
    parser.add_argument(
        "--config_auth_token_file",
        type=pathlib.Path,
        help="Path to a file containing a (Github) authentication token.",
    )
    parsed, unparsed = parser.parse_known_args(args)
    trainer_config_class = trainer_registry[parsed.model_type]
    with NoAutoValidate():
        if parsed.source == ConfigSource.auto:
            if parsed.config is None:
                parsed.source = ConfigSource.args
            elif urllib.parse.urlparse(parsed.config).scheme == "https":
                parsed.source = ConfigSource.url
            elif pathlib.Path(parsed.config).is_file():
                parsed.source = ConfigSource.file
            else:
                raise ValueError(f"Cannot deduce source from config `{parsed.config}`")
        config: TrainerConfig
        if parsed.source == ConfigSource.args:
            config = trainer_config_class.from_flat_args(unparsed)
        else:
            Assert.empty(unparsed)
            if parsed.source == ConfigSource.url:
                config_file = load_url(parsed.config, parsed.config_auth_token_file)
            elif parsed.source == ConfigSource.file:
                config_file = pathlib.Path(parsed.config).open("r").read()
            config = trainer_config_class.from_dict(yaml.safe_load(config_file))
    try:
        config.validate()
        if not parsed.do_run:
            return
        from fast_llm.engine.distributed.distributed import Distributed

        distributed = Distributed(config.distributed)
        run = config.get_run(distributed)
        trainer = trainer_config_class.get_trainer_class()(config=config)
    finally:
        # We always want to show the config for debugging.
        config.show_main_rank(config.model.distributed)
    with run:
        trainer.setup(distributed, run)
        trainer.run()


if __name__ == "__main__":
    train()
