import logging

from fast_llm.models.gpt.trainer import GPTTrainer
from fast_llm.models.multimodal.config import MultiModalTrainerConfig

logger = logging.getLogger(__name__)


class MultiModalTrainer[ConfigType: MultiModalTrainerConfig](GPTTrainer[ConfigType]):
    pass
