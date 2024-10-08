import argparse
import logging
import pathlib
import shlex
import sys
import typing
import urllib.parse

import requests
import yaml

from fast_llm.config import Config, FieldVerboseLevel, NoAutoValidate, config_class
from fast_llm.engine.config_utils.logging import configure_logging

logger = logging.getLogger(__name__)


@config_class()
class RunnableConfig(Config):
    @classmethod
    def parse_and_run(cls, args=None):
        parsed, unparsed = cls._get_parser().parse_known_args(args)
        with NoAutoValidate():
            config: "RunnableConfig" = cls._from_parsed_args(parsed, unparsed)
        try:
            config.configure_logging()
            config.validate()
            if not parsed.do_run:
                return
            # Do here so we have a chance to finalize logging configuration before logging the config.
            runnable = config._get_runnable(parsed)
        finally:
            # We always want to show the config for debugging.
            config._show(parsed.verbose)
        runnable()

    @classmethod
    def _get_parser(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-v",
            "--validate",
            dest="do_run",
            action="store_false",
            help="Validate the config without running the command.",
        )
        parser.add_argument(
            # TODO: remove --config_url once changed in fml-ops
            "-c",
            "--config",
            "--config_url",
            help="The configuration file or url.",
        )
        parser.add_argument(
            "--verbose", type=int, help="Verbose level for logging the config.", default=FieldVerboseLevel.core
        )
        parser.add_argument(
            "--config_auth_token_file",
            type=pathlib.Path,
            help="Path to a file containing a (Github) authentication token.",
        )
        return parser

    @classmethod
    def _from_parsed_args(cls, parsed: argparse.Namespace, unparsed: list[str]):
        default = cls._load_default_config_dict(parsed)
        updates = cls._parse_updates(unparsed)
        return cls.from_dict(default, updates)

    def configure_logging(self):
        configure_logging()

    def _get_runnable(self, parsed: argparse.Namespace) -> typing.Callable[[], None]:
        return self.run

    def run(self):
        raise NotImplementedError()

    def _show(
        self,
        verbose: int = FieldVerboseLevel.core,
        *,
        log_fn=logger.info,
        title: str | None = None,
        width: int = 60,
        fill_char: str = "-",
    ):
        log_fn(f"Command run:\n{shlex.join(sys.argv)}")
        self.to_logs(verbose=verbose, log_fn=log_fn, title=title, width=width, fill_char=fill_char)

    @classmethod
    def _load_default_config_dict(cls, parsed: argparse.Namespace):
        if parsed.config is None:
            return {}
        elif urllib.parse.urlparse(parsed.config).scheme == "https":
            return yaml.safe_load(cls._load_url(parsed.config, parsed.config_auth_token_file))
        elif pathlib.Path(parsed.config).is_file():
            return yaml.safe_load(pathlib.Path(parsed.config).open("r").read())
        else:
            raise FileNotFoundError(parsed.config)

    @classmethod
    def _load_url(cls, config_url: str, config_auth_token_file: pathlib.Path | None = None):
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
    def _parse_updates(cls, unparsed: list[str]):
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
