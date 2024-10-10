from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.models.custom.config import CustomBaseModelConfig, CustomModelConfig
from fast_llm.models.custom.head import CustomHead
from fast_llm.models.gpt.model import GPTBaseModel, GPTModel


class CustomBaseModel(GPTBaseModel):
    _config: CustomBaseModelConfig
    config_cls = CustomBaseModelConfig

    def __init__(
        self,
        config: CustomBaseModelConfig,
        distributed_config: DistributedConfig,
    ):
        # TODO: Implement / update.
        super().__init__(config, distributed_config)

    def get_layers(self):
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

    def preprocess_meta(self, input_, phase):
        # TODO: Adjust or reimplement.
        return super().preprocess_meta(input_, phase)

    def preprocess(self, batch, preprocessed_meta=None, *, phase, iteration, metrics=None):
        # TODO: Adjust or reimplement.
        return super().preprocess(batch, preprocessed_meta, phase=phase, iteration=iteration, metrics=metrics)

    @property
    def loss_defs(self):
        # TODO: Adjust or reimplement.
        return super().loss_defs


class CustomModel(GPTModel):
    config_class = CustomModelConfig
    base_model_class = CustomBaseModel
