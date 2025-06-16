import typing

from fast_llm.models.gpt.trainer import GPTTrainer
from fast_llm.models.ssm.config import HybridSSMTrainerConfig
from fast_llm.models.ssm.model import HybridSSMModel


class HybridSSMTrainer[ConfigType: HybridSSMTrainerConfig](GPTTrainer[ConfigType]):
    config_class: typing.ClassVar[type[HybridSSMTrainerConfig]] = HybridSSMTrainerConfig
    model_class: typing.ClassVar[type[HybridSSMModel]] = HybridSSMModel
