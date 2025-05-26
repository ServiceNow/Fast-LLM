import typing

from fast_llm.models.gpt.trainer import GPTTrainer
from fast_llm.models.hybrid.config import HybridTrainerConfig
from fast_llm.models.hybrid.model import HybridSSMModel


class SSMTrainer[ConfigType: HybridTrainerConfig](GPTTrainer[ConfigType]):
    config_class: typing.ClassVar[type[HybridTrainerConfig]] = HybridTrainerConfig
    model_class: typing.ClassVar[type[HybridSSMModel]] = HybridSSMModel
