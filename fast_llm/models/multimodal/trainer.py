import logging

from fast_llm.models.gpt.trainer import GPTTrainer
from fast_llm.models.multimodal.config import MultiModalTrainerConfig

logger = logging.getLogger(__name__)


class MultiModalTrainer[ConfigType: MultiModalTrainerConfig](GPTTrainer[ConfigType]):
    def _get_data(self) -> MultiModalData:
        return MultiModalData(
            config=self._config.data,
            distributed_config=self._config.model.distributed,
        )
