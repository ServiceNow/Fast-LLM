from fast_llm.models.custom.config import CustomTrainerConfig
from fast_llm.models.custom.data import CustomData
from fast_llm.models.custom.model import CustomModel
from fast_llm.models.gpt.trainer import GPTTrainer


class CustomTrainer(GPTTrainer):
    # TODO: Implement changes in the training loop (or tflops computation), if any (typically none).
    _abstract = False
    _config: CustomTrainerConfig
    config_class = CustomTrainerConfig
    model_class = CustomModel

    def _get_data(self):
        # TODO: Adjust signature if needed.
        return CustomData(
            config=self._config.data,
            distributed_config=self._config.distributed,
            vocab_size=self._config.base_model.vocab_size,
            max_sequence_length=self._config.batch.sequence_length,
        )
