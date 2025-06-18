from fast_llm.config import config_class
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.training.config import TrainerConfig


@config_class(dynamic_type={RunnableConfig: "evaluate"})
class EvaluatorsConfig(RunnableConfig):

    @classmethod
    def parse_and_run(cls, args: list[str] | None = None) -> None:
        args.append("training.train_iters=0")
        return TrainerConfig.parse_and_run(args)
