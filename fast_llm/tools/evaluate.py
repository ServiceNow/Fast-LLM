import argparse

from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.models.auto import trainer_registry


class CliEvaluationConfig(RunnableConfig):
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
        unparsed += ['training.train_iters=0']
        return trainer_registry[parsed.model_type]._from_parsed_args(parsed, unparsed)


if __name__ == "__main__":
    CliEvaluationConfig.parse_and_run()
