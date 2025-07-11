import argparse
import logging
import pathlib
import shlex
import sys
import typing
import urllib.parse
import warnings

import requests
import yaml

from fast_llm.config import Config, FieldVerboseLevel, NoAutoValidate, config_class
from fast_llm.engine.config_utils.logging import configure_logging

logger = logging.getLogger(__name__)


@config_class(registry=True)
class RunnableConfig(Config):
    @classmethod
    def parse_and_run(cls, args: list[str] | None = None) -> None:
        if args is None:
            args = sys.argv[1:]
        if cls._first_arg_is_dynamic_type(args):
            # Allow chained dynamic type selection without the `type=`, ex. `train gpt`.
            return cls.get_subclass(args[0]).parse_and_run(args[1:])
        parsed, unparsed = cls._get_parser().parse_known_args(args)
        with NoAutoValidate():
            config: "RunnableConfig" = cls._from_parsed_args(parsed, unparsed)
        try:
            # Configure logging so validation errors are logged properly.
            config.configure_logging()
            config.validate()
            if not parsed.do_run:
                return
            # Do here so we have a chance to finalize logging configuration before logging the config.
            runnable = config._get_runnable()
        finally:
            # We always want to show the config for debugging.
            config._show(parsed.verbose)
        runnable()

    @classmethod
    def _first_arg_is_dynamic_type(cls, args: list[str]) -> bool:
        return len(args) >= 1 and "=" not in args[0] and not args[0].startswith("-")

    @classmethod
    def _get_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-v",
            "--validate",
            dest="do_run",
            action="store_false",
            help="Validate the config without running the command.",
        )
        parser.add_argument(
            "-c",
            "--config",
            help="The path or url to the base configuration yaml file.",
        )
        parser.add_argument(
            "--verbose", type=int, help="Verbose level for logging the config.", default=FieldVerboseLevel.core
        )
        parser.add_argument(
            "--config_auth_token_file",
            type=pathlib.Path,
            help="Path to a file containing a (Github) authentication token.",
        )
        parser.add_argument(
            "--hydra",
            action="store_true",
            help="Enable the hydra syntax for configuration updates."
            " Note: this will only enable the update syntax for the command line arguments."
            " See the other Hydra options for enabling more optional features.",
        )
        parser.add_argument(
            "--hydra-path",
            type=pathlib.Path,
            help="The hydra configuration path in which to loop for updates relative to the current working directory."
            " Setting this will implicitly enable --hydra.",
        )
        parser.add_argument(
            "--hydra-config",
            help="The name of the hydra base configuration as would be provided in a typical Hydra application."
            " Mutually exclusive with `--config`, which it replaces."
            " Requires --hydra-path to be set.",
        )
        return parser

    @classmethod
    def _from_parsed_args(cls, parsed: argparse.Namespace, unparsed: list[str]) -> typing.Self:
        if parsed.hydra or parsed.hydra_path is not None or parsed.hydra_config is not None:
            default = cls._load_hydra(parsed, unparsed)
            updates = {}
        else:
            default = cls._load_default_config_dict(parsed)
            updates = cls._parse_updates(unparsed)
        return cls.from_dict(default, updates)

    def configure_logging(self) -> None:
        configure_logging()

    def _get_runnable(self) -> typing.Callable[[], None]:
        return self.run

    def run(self) -> None:
        raise NotImplementedError()

    def _show[
        T
    ](
        self,
        verbose: int = FieldVerboseLevel.core,
        *,
        log_fn: typing.Callable[[str], T] = logger.info,
        title: str | None = None,
        width: int = 60,
        fill_char: str = "-",
    ) -> T:
        log_fn(f"Command run:\n{shlex.join(sys.argv)}")
        return self.to_logs(verbose=verbose, log_fn=log_fn, title=title, width=width, fill_char=fill_char)

    @classmethod
    def _load_hydra(cls, parsed: argparse.Namespace, unparsed: list[str]) -> typing.Any:
        warnings.warn("Hydra support is experimental and may be modified or removed in the future")
        import hydra.core.config_store
        import omegaconf

        default = cls._load_default_config_dict(parsed)
        # Hydra expects a path relative to the application path but that doesn't make sense here.
        config_path = (
            None
            if parsed.hydra_path is None
            else str(parsed.hydra_path.resolve().relative_to(pathlib.Path(__file__).parent.resolve(), walk_up=True))
        )
        with hydra.initialize(version_base=None, config_path=config_path):
            if default:
                assert parsed.hydra_config is None
                cs = hydra.core.config_store.ConfigStore.instance()
                cs.store(name="__fast_llm__default__", node=omegaconf.DictConfig(default))
                default = "__fast_llm__default__"
            else:
                default = parsed.hydra_config
                if default is not None:
                    assert config_path is not None
            cfg = hydra.compose(config_name=default, overrides=unparsed)
            return omegaconf.OmegaConf.to_object(cfg)

    @classmethod
    def _load_default_config_dict(cls, parsed: argparse.Namespace) -> typing.Any:
        if parsed.config is None:
            return {}
        elif urllib.parse.urlparse(parsed.config).scheme == "https":
            return yaml.safe_load(cls._load_url(parsed.config, parsed.config_auth_token_file))
        elif pathlib.Path(parsed.config).is_file():
            return yaml.safe_load(pathlib.Path(parsed.config).open("r").read())
        else:
            raise FileNotFoundError(parsed.config)

    @classmethod
    def _load_url(cls, config_url: str, config_auth_token_file: pathlib.Path | None = None) -> typing.Any:
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
            return response.text
        else:
            if isinstance(response.reason, bytes):
                try:
                    reason = response.reason.decode("utf-8")
                except UnicodeDecodeError:
                    reason = response.reason.decode("iso-8859-1")
            else:
                reason = response.reason
            raise ValueError(f"Failed to fetch config from {config_url}: {response.status_code}, {reason}")

    @classmethod
    def _parse_updates(cls, unparsed: list[str]) -> dict[str | tuple[str, ...], typing.Any]:
        updates: dict[str | tuple[str, ...], typing.Any] = {}
        errors = []
        for arg in unparsed:
            try:
                key, value = arg.split("=", 1)
                updates[tuple(key.split("."))] = yaml.safe_load(value)
            except (ValueError, yaml.YAMLError):
                errors.append(arg)
        if errors:
            raise ValueError(f"Cannot parse arguments {' '.join(errors)}")
        return updates
