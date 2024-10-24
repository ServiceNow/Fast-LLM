from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
from fast_llm.layers.language_model.head import LanguageModelHead
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.models.grpo.config import GRPOBaseModelConfig, GRPOModelConfig
from fast_llm.models.gpt.model import GPTBaseModel, GPTModel


class GRPOBaseModel(GPTBaseModel):
    _config: GRPOBaseModelConfig
    config_cls = GRPOBaseModelConfig

    def __init__(
        self,
        config: GRPOModelConfig,
        distributed_config: DistributedConfig,
    ):
        super().__init__(config, distributed_config)
        assert self._config.transformer.use_rotary_position_embeddings
        assert not self._config.use_absolute_position_embeddings

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
            LanguageModelHead(self._config, self._tensor_space),
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


class GRPOModel(GPTModel):
    config_class = GRPOModelConfig
    base_model_class = GRPOBaseModel
