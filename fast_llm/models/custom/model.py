import typing

import torch

from fast_llm.data.data.gpt.data import GPTBatch
from fast_llm.engine.base_model.base_model import LossDef
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.models.custom.config import CustomBaseModelConfig
from fast_llm.models.custom.head import CustomHead
from fast_llm.models.gpt.model import GPTBaseModel, GPTModel
from fast_llm.tensor import TensorMeta


class CustomBaseModel[ConfigType: CustomBaseModelConfig](GPTBaseModel[ConfigType]):
    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
    ):
        # TODO: Implement / update.
        super().__init__(config, distributed_config)

    def _get_head(self, prediction_distance):
        return CustomHead(
            self._config,
            self._distributed_config,
            self._hidden_dim,
            max(self._config.transformer.num_layers + prediction_distance, 1),
            f"Language model head {prediction_distance}",
            prediction_distance=prediction_distance,
        )

    def preprocess_meta(
        self, batch_meta: BatchConfig | torch.Tensor, phase: PhaseType
    ) -> list[tuple[TensorMeta, dict]]:
        # TODO: Adjust or reimplement.
        return super().preprocess_meta(batch_meta, phase)

    def preprocess(
        self,
        batch: GPTBatch,
        preprocessed_meta: list[tuple[TensorMeta, dict]] | None = None,
        *,
        phase: PhaseType,
        iteration: int,
        metrics: dict | None = None,
    ) -> list[tuple[torch.Tensor, dict]]:
        # TODO: Adjust or reimplement.
        return super().preprocess(batch, preprocessed_meta, phase=phase, iteration=iteration, metrics=metrics)

    @property
    def loss_defs(self) -> list[LossDef]:
        # TODO: Adjust or reimplement.
        return super().loss_defs


class CustomModel[ConfigType: CustomBaseModelConfig](GPTModel[ConfigType]):
    base_model_class: typing.ClassVar[type[CustomBaseModel]] = CustomBaseModel
