import logging
import typing

from fast_llm.config import Configurable
from fast_llm.engine.base_model.base_model import Layer, LayerBase
from fast_llm.engine.base_model.config import LossDef
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.language_model.config import LanguageModelConfig
from fast_llm.layers.language_model.embedding import LanguageModelEmbedding

logger = logging.getLogger(__name__)


class LanguageModel[ConfigType: LanguageModelConfig](Configurable[ConfigType], LayerBase):
    _config: ConfigType

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
    ):
        super().__init__(config, distributed_config)

        self._hidden_dim = TensorDim("hidden", config.embeddings.hidden_size)
        self.embeddings: LanguageModelEmbedding = self._config.embeddings.get_layer(
            distributed_config,
            hidden_dim=self._hidden_dim,
            lr_scale=None,
            peft=self._config.peft,
        )
        self.decoder = self._config.decoder.get_layer(
            distributed_config,
            self._hidden_dim,
            lr_scale=None,
            peft=self._config.peft,
        )
        self.head = self._config.head.get_layer(
            distributed_config,
            self._config.embeddings,
            hidden_dim=self._hidden_dim,
            lr_scale=None,
            peft=self._config.peft,
        )

    def get_layers(self) -> list["Layer"]:
        return self.embeddings.get_layers() + self.decoder.get_layers() + self.head.get_layers()

    def preprocess(self, batch: "torch.Tensor", kwargs: dict[str, typing.Any]) -> None:
        # Needed because the base class uses `get_layers` which may bypass the decoder and head. TODO: Avoidable?
        self.embeddings.preprocess(batch, kwargs)
        self.decoder.preprocess(batch, kwargs)
        self.head.preprocess(batch, kwargs)

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        # Needed because the base class uses `get_layers` which may bypass the decoder and head. TODO: Avoidable?
        return (
            self.embeddings.get_loss_definitions(count)
            + self.decoder.get_loss_definitions(count)
            + self.head.get_loss_definitions(count)
        )
