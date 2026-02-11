import logging
import typing

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.base_model.config import LossDef
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.block.block import BlockBase
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.language_model.config import LanguageModelConfig
from fast_llm.layers.language_model.embedding import LanguageModelEmbedding

logger = logging.getLogger(__name__)


class LanguageModel[ConfigType: LanguageModelConfig](BlockBase[ConfigType]):
    _config: ConfigType

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        # TODO: Unused, but required by the `BlockBase` interface.
        hidden_dim: TensorDim | None = None,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ):
        super().__init__(
            config,
            distributed_config,
            hidden_dim=TensorDim("hidden", config.hidden_size),
            lr_scale=lr_scale,
            peft=peft,
        )
        self.embeddings: LanguageModelEmbedding = self._config.embeddings.get_layer(
            distributed_config,
            hidden_dim=self._hidden_dim,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.decoder = self._config.decoder.get_layer(
            distributed_config,
            self._hidden_dim,
            lr_scale=self._lr_scale,
            peft=self._peft,
            **({"return_last_layer_input": True} if self._config.head.prediction_heads > 1 else {}),
        )
        self.head, self.multi_token_prediction = self._config.head.get_layer(
            distributed_config,
            self._config.embeddings,
            hidden_dim=self._hidden_dim,
            lr_scale=self._lr_scale,
            peft=self._peft,
            **(
                {"block_config": self._config.decoder.last_block_config}
                if self._config.head.prediction_heads > 1
                else {}
            ),
        )

    def get_layers(self) -> list[Layer]:
        layers = self.embeddings.get_layers() + self.decoder.get_layers() + self.head.get_layers()
        if self.multi_token_prediction is not None:
            layers += self.multi_token_prediction.get_layers()
        return layers

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        # Needed because the base class uses `get_layers` which may bypass the decoder and head. TODO: Avoidable?
        self.embeddings.preprocess(kwargs)
        self.decoder.preprocess(kwargs)
        self.head.preprocess(kwargs)
        if self.multi_token_prediction is not None:
            self.multi_token_prediction.preprocess(kwargs)

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        # Needed because the base class uses `get_layers` which may bypass the decoder and head. TODO: Avoidable?
        losses = (
            self.embeddings.get_loss_definitions(count)
            + self.decoder.get_loss_definitions(count)
            + self.head.get_loss_definitions(count)
        )
        if self.multi_token_prediction is not None:
            losses += self.multi_token_prediction.get_loss_definitions(count)
        return losses
