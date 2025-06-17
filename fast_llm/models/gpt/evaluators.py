from fast_llm.config import config_class
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.evaluation.evaluators import EvaluatorsConfig
from fast_llm.models.gpt.config import GPTTrainerConfig


@config_class(dynamic_type={RunnableConfig: "evaluate_gpt", EvaluatorsConfig: "gpt"})
class GPTEvaluatorsConfig(EvaluatorsConfig, GPTTrainerConfig):
    _abstract = False
