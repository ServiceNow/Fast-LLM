import logging
import typing

from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.gpt.config import GPTSamplingParameters
from fast_llm.engine.base_model.config import Preprocessor
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.training.trainer import Trainer
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.models.gpt.config import GPTTrainerConfig
from fast_llm.models.gpt.model import GPTInferenceRunner

logger = logging.getLogger(__name__)


class GPTReferenceModelPreprocessor(Preprocessor):
    def __init__(self, name: str, inference_runner: GPTInferenceRunner):
        self._name = name
        self._inference_runner = inference_runner

    def preprocess_meta(self, kwargs: dict[str, typing.Any]) -> None:
        pass

    def preprocess(self, batch, kwargs: dict[str, typing.Any]) -> None:
        # TODO: Fix random state/iteration.
        preprocess_kwargs = kwargs.copy()
        del preprocess_kwargs[LanguageModelKwargs.labels]
        self._inference_runner.forward(batch, preprocess_kwargs, iteration=1)
        # TODO: Improve.
        kwargs[f"{self._name}_logits"] = preprocess_kwargs["logits"]


class GPTTrainer[ConfigType: GPTTrainerConfig](Trainer[ConfigType]):
    config_class: typing.ClassVar[type[GPTTrainerConfig]] = GPTTrainerConfig

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
                "vocab_size": self._config.model.base_model.vocab_size,
                "sequence_length": self._config.batch.sequence_length,
                "use_loss_masking_spans": self._config.batch.use_loss_masking_spans,
                "cross_document_attention": self._config.batch.cross_document_attention,
                "extra_tokens": self._config.model.base_model.prediction_heads,
            }
        )
        return parameters if _return_dict else GPTSamplingParameters(**parameters)

    def _get_reference_model_preprocessor(self, name: str, inference_runner: GPTInferenceRunner) -> Preprocessor:
        return GPTReferenceModelPreprocessor(name, inference_runner)
