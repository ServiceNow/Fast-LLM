import argparse

from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.models.auto import trainer_registry


class CliTrainingConfig(RunnableConfig):
    @classmethod
    def _get_parser(cls):
        parser = super()._get_parser()
        parser.add_argument(
            "model_type",
            choices=trainer_registry.keys(),
            help="The Fast-LLM model type to use. Must be defined in the trainer registry in `fast_llm.models.auto`.",
        )
        return parser

    @classmethod
    def _from_parsed_args(cls, parsed: argparse.Namespace, unparsed: list[str]):
        return trainer_registry[parsed.model_type]._from_parsed_args(parsed, unparsed)


if __name__ == "__main__":
    CliTrainingConfig.parse_and_run()
