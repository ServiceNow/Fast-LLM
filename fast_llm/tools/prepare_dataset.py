import argparse

from fast_llm.data.auto import dataset_preparator_registry
from fast_llm.engine.config_utils.runnable import RunnableConfig


class PrepareDatasetConfig(RunnableConfig):
    @classmethod
    def _get_parser(cls):
        parser = super()._get_parser()
        parser.add_argument(
            "model_type",
            choices=dataset_preparator_registry.keys(),
            help="The Fast-LLM model type to use. Must be defined in the model registry in `fast_llm.models.auto`.",
        )
        return parser

    @classmethod
    def _from_parsed_args(cls, parsed: argparse.Namespace, unparsed: list[str]):
        return dataset_preparator_registry[parsed.model_type]._from_parsed_args(parsed, unparsed)


if __name__ == "__main__":
    PrepareDatasetConfig.parse_and_run()
