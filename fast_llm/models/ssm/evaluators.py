from fast_llm.config import config_class
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.evaluation.evaluators import EvaluatorsConfig
from fast_llm.models.ssm.config import HybridSSMTrainerConfig


@config_class(dynamic_type={RunnableConfig: "evaluate_hybrid_ssm", EvaluatorsConfig: "hybrid_ssm"})
class HybridSSMEvaluatorsConfig(EvaluatorsConfig, HybridSSMTrainerConfig):
    _abstract = False
