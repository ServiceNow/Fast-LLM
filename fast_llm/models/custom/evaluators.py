from fast_llm.config import config_class
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.evaluation.evaluators import EvaluatorsConfig
from fast_llm.models.custom.config import CustomTrainerConfig


@config_class(dynamic_type={RunnableConfig: "evaluate_custom", EvaluatorsConfig: "custom"})
class CustomEvaluatorsConfig(EvaluatorsConfig, CustomTrainerConfig):
    _abstract = False
