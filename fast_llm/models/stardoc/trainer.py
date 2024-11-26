from fast_llm.models.stardoc.config import StarDocTrainerConfig
from fast_llm.models.stardoc.data import StarDocData
from fast_llm.models.stardoc.model import StarDocModel
from fast_llm.models.gpt.trainer import GPTTrainer


class StarDocTrainer(GPTTrainer):
    _abstract = False
    _config: StarDocTrainerConfig
    config_class = StarDocTrainerConfig
    model_class = StarDocModel

    def _get_data(self):
        return StarDocData(
            config=self._config.data,
            distributed_config=self._config.distributed,
            vocab_size=self._config.base_model.vocab_size,
            max_sequence_length=self._config.batch.sequence_length,
        )
