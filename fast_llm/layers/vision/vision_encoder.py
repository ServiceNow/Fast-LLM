import functools
import logging
import typing

from fast_llm.engine.base_model.base_model import Layer, LayerBaseWithNamespace
from fast_llm.engine.base_model.config import LossDef
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.block.block import BlockBase
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.language_model.language_model import LanguageModel
from fast_llm.layers.vision.config import VisionEncoderConfig, VisionMultiModalModelConfig

logger = logging.getLogger(__name__)


class VisionEncoder[ConfigType: VisionEncoderConfig](BlockBase[ConfigType]):
    _config: ConfigType

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ):
        super().__init__(config, distributed_config, hidden_dim=hidden_dim, lr_scale=lr_scale, peft=peft)
        # Internal hidden dimension for embeddings and encoder (may differ from output hidden_dim for adapter)
        self._vision_hidden_dim = TensorDim("hidden", self._config.hidden_size)
        vision_hidden_dim = self._vision_hidden_dim
        self.embeddings = self._config.embeddings.get_layer(
            distributed_config,
            vision_hidden_dim,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.encoder = self._config.encoder.get_layer(
            distributed_config,
            vision_hidden_dim,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.adapter = self._config.adapter.get_layer(
            distributed_config,
            vision_hidden_dim,
            output_dim=self._hidden_dim,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

    def get_layers(self) -> list["Layer"]:
        return self.embeddings.get_layers() + self.encoder.get_layers() + self.adapter.get_layers()

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        # Needed because the base class uses `get_layers` which may bypass the decoder. TODO: Avoidable?
        self.embeddings.preprocess(kwargs)
        self.encoder.preprocess(kwargs)
        self.adapter.preprocess(kwargs)

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        # Needed because the base class uses `get_layers` which may bypass the decoder. TODO: Avoidable?
        return (
            self.embeddings.get_loss_definitions(count)
            + self.encoder.get_loss_definitions(count)
            + self.adapter.get_loss_definitions(count)
        )


class VisionMultiModalModel[ConfigType: VisionMultiModalModelConfig](LanguageModel[ConfigType]):
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
        self.vision_encoder = self._config.vision_encoder.get_layer(
            distributed_config,
            hidden_dim=self._hidden_dim,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

    def get_layers(self) -> list[Layer]:
        return self._vision_encoder_with_namespace.get_layers() + super().get_layers()

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        self._vision_encoder_with_namespace.preprocess(kwargs)
        super().preprocess(kwargs)

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        return self.vision_encoder.get_loss_definitions(count) + super().get_loss_definitions(count)

    @functools.cached_property
    def _vision_encoder_namespace(self) -> str:
        return self.vision_encoder.module_name

    @functools.cached_property
    def _vision_encoder_with_namespace(self) -> LayerBaseWithNamespace:
        return LayerBaseWithNamespace(self.vision_encoder, self._vision_encoder_namespace)
