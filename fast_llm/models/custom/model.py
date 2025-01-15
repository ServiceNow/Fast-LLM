import typing

import torch

from fast_llm.engine.base_model.base_model import Layer, LossDef
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.models.custom.config import CustomBaseModelConfig, CustomModelConfig
from fast_llm.models.custom.head import CustomHead
from fast_llm.models.gpt.config import GPTBaseModelConfig
from fast_llm.models.gpt.model import GPTBaseModel, GPTModel
from fast_llm.tensor import TensorMeta


class CustomBaseModel[ConfigType: CustomBaseModelConfig](GPTBaseModel[ConfigType]):
    config_class: typing.ClassVar[type[GPTBaseModelConfig]] = GPTBaseModelConfig

    def __init__(
        self,
        config: CustomBaseModelConfig,
        distributed_config: DistributedConfig,
    ):
        # TODO: Implement / update.
        super().__init__(config, distributed_config)

    def get_layers(self) -> list[Layer]:
        # TODO: Adjust as needed.
        return [
            LanguageModelEmbedding(self._config, self._tensor_space),
            *[
                TransformerLayer(
                    self._config.transformer,
                    self._tensor_space,
                    layer_index=i + 1,
                )
                for i in range(self._config.transformer.num_layers)
            ],
            CustomHead(self._config, self._tensor_space),
        ]

    def preprocess_meta(
        self, batch_meta: BatchConfig | torch.Tensor, phase: PhaseType
    ) -> list[tuple[TensorMeta, dict]]:
        # TODO: Adjust or reimplement.
        return super().preprocess_meta(batch_meta, phase)

    def preprocess(
        self,
        batch: torch.Tensor,
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
    config_class: typing.ClassVar[type[CustomModelConfig]] = CustomModelConfig
    base_model_class: typing.ClassVar[type[CustomBaseModel]] = CustomBaseModel
