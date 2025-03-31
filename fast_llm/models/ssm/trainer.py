import typing

from fast_llm.models.gpt.trainer import GPTTrainer
from fast_llm.models.ssm.config import HybridTrainerConfig
from fast_llm.models.ssm.model import HybridModel


class SSMTrainer[ConfigType: HybridTrainerConfig](GPTTrainer[ConfigType]):
    config_class: typing.ClassVar[type[HybridTrainerConfig]] = HybridTrainerConfig
    model_class: typing.ClassVar[type[HybridModel]] = HybridModel
