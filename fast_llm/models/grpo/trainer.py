from fast_llm.models.grpo.config import GRPOTrainerConfig
from fast_llm.models.grpo.data import GRPOData
from fast_llm.models.grpo.model import GRPOModel
from fast_llm.models.gpt.trainer import GPTTrainer


class GRPOTrainer(GPTTrainer):
    # TODO: Implement changes in the training loop (or tflops computation), if any (typically none).
    _abstract = False
    _config: GRPOTrainerConfig
    config_class = GRPOTrainerConfig
    model_class = GRPOModel

    def _get_data(self):
        # TODO: Adjust signature if needed.
        return GRPOData(
            config=self._config.data,
            distributed_config=self._config.distributed,
            vocab_size=self._config.base_model.vocab_size,
            max_sequence_length=self._config.batch.sequence_length,
        )
