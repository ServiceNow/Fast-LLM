import logging
import typing

import torch

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.base_model.config import LossDef
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.block.block import BlockBase
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.vision.config import VisionEncoderConfig

logger = logging.getLogger(__name__)


class VisionEncoder[ConfigType: VisionEncoderConfig](BlockBase[VisionEncoderConfig]):
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
        vision_hidden_dim = TensorDim("hidden", self._config.hidden_size)
        super().__init__(config, distributed_config, hidden_dim=hidden_dim, lr_scale=lr_scale, peft=peft)
        self.patch_convolution = self._config.patch_convolution.get_layer(
            distributed_config,
            vision_hidden_dim,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        # TODO: ====== Appropriate name?? ======
        self.decoder = self._config.decoder.get_layer(
            distributed_config,
            vision_hidden_dim,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        # TODO: ====== Hidden dim ======
        self.adapter = self._config.adapter.get_layer(
            distributed_config,
            vision_hidden_dim,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

    def get_layers(self) -> list["Layer"]:
        return self.patch_convolution.get_layers() + self.decoder.get_layers() + self.adapter.get_layers()

    def preprocess(self, batch: torch.Tensor, kwargs: dict[str, typing.Any]) -> None:
        # Needed because the base class uses `get_layers` which may bypass the decoder and head. TODO: Avoidable?
        self.patch_convolution.preprocess(batch, kwargs)
        self.decoder.preprocess(batch, kwargs)
        self.adapter.preprocess(batch, kwargs)

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        # Needed because the base class uses `get_layers` which may bypass the decoder and head. TODO: Avoidable?
        return (
            self.patch_convolution.get_loss_definitions(count)
            + self.decoder.get_loss_definitions(count)
            + self.adapter.get_loss_definitions(count)
        )
