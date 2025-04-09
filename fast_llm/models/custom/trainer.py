import typing

from fast_llm.models.custom.config import CustomTrainerConfig
from fast_llm.models.custom.data import CustomData
from fast_llm.models.gpt.trainer import GPTTrainer


class CustomTrainer[ConfigType: CustomTrainerConfig](GPTTrainer[ConfigType]):
    # TODO: Implement changes in the training loop (or tflops computation), if any (typically none).
    config_class: typing.ClassVar[type[CustomTrainerConfig]] = CustomTrainerConfig

    def _get_data(self):
        # TODO: Adjust signature if needed.
        return CustomData(
            config=self._config.data,
            distributed_config=self._config.model.distributed,
            vocab_size=self._config.model.base_model.vocab_size,
            max_sequence_length=self._config.batch.sequence_length,
        )
