import logging

from fast_llm.data.data.gpt.data import GPTData
from fast_llm.engine.training.trainer import Trainer
from fast_llm.models.gpt.config import GPTTrainerConfig

logger = logging.getLogger(__name__)


class GPTTrainer[ConfigType: GPTTrainerConfig](Trainer[ConfigType]):
    def _get_data(self) -> GPTData:
        return GPTData(
            config=self._config.data,
            distributed_config=self._config.model.distributed,
        )
