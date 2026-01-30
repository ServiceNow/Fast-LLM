import logging
import typing

from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.config import SamplingParameters
from fast_llm.data.preprocessing.language_model import LanguageModelPreprocessingConfig
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
        self, parameters: dict[str, typing.Any], *, _return_dict: bool = False
    ) -> SamplingParameters | dict[str, typing.Any]:
        parameters = super()._get_sampling_parameters(parameters, _return_dict=True)
        parameters.update(
            {
                "sequence_length": self._config.batch.sequence_length,
                "truncate_documents": self._config.batch.truncate_documents,
                "extra_tokens": self._config.model.base_model.head.max_prediction_distance,
            }
        )
        return parameters if _return_dict else SamplingParameters(**parameters)

    def _get_preprocessing_config(
        self, *, _return_dict: bool = False
    ) -> LanguageModelPreprocessingConfig | dict[str, typing.Any]:

        out = {
            "type": "language_model",
            "vocab_size": self._config.model.base_model.embeddings.vocab_size,
            "use_loss_masking_spans": self._config.batch.use_loss_masking_spans,
            "use_preference_spans": self._config.batch.use_preference_spans,
            "use_grpo_data": self._config.batch.use_grpo_data,
        }
        return out if _return_dict else LanguageModelPreprocessingConfig.from_dict(out)
