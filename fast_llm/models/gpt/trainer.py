import logging
import typing

from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.gpt.config import GPTSamplingParameters
from fast_llm.engine.training.trainer import Trainer
from fast_llm.models.gpt.config import GPTTrainerConfig

logger = logging.getLogger(__name__)


class GPTTrainer[ConfigType: GPTTrainerConfig](Trainer[ConfigType]):
    def _get_data(self) -> GPTData:
        return GPTData(
            config=self._config.data,
            distributed_config=self._config.model.distributed,
        )

    def _get_sampling_parameters(
        self, parameters: dict[str, typing.Any], _return_dict: bool = False
    ) -> GPTSamplingParameters | dict[str, typing.Any]:
        parameters = super()._get_sampling_parameters(parameters, _return_dict=True)
        parameters.update(
            {
                "vocab_size": self._config.model.base_model.embeddings_layer.vocab_size,
                "sequence_length": self._config.batch.sequence_length,
                "use_loss_masking_spans": self._config.batch.use_loss_masking_spans,
                "use_preference_loss_spans": self._config.model.base_model.output_layer.enable_dpo,
                "cross_document_attention": self._config.batch.cross_document_attention,
                "truncate_documents": self._config.batch.truncate_documents,
                "extra_tokens": self._config.model.base_model.output_layer.prediction_heads,
            }
        )
        return parameters if _return_dict else GPTSamplingParameters(**parameters)
