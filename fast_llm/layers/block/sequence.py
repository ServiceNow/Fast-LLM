import collections

import torch.nn

from fast_llm.engine.base_model.base_model import Layer, LayerWithNamespace
from fast_llm.engine.base_model.config import LossDef
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.block.block import BlockBase
from fast_llm.layers.block.config import FixedBlockSequenceConfig, PatternBlockSequenceConfig
from fast_llm.layers.common.peft.config import PeftConfig


class FixedBlockSequence[ConfigType: FixedBlockSequenceConfig](BlockBase[ConfigType], torch.nn.ModuleList):
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
        super().__init__(
            config,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=lr_scale,
            peft=peft,
        )
        self.extend(
            layers := [
                self._config.block.get_layer(
                    distributed_config,
                    hidden_dim,
                    lr_scale=self._lr_scale,
                    peft=self._peft,
                )
                for _ in range(self._config.num_blocks)
            ]
        )
        # Wrap all blocks in a namespace using the unique module name of the first one.
        namespace = layers[0].module_name if self._config.num_blocks > 0 else ""
        # Note: Pytorch won't redundantly register modules  because it doesn't look into lists.
        self._layers_with_namespace = [
            LayerWithNamespace(sublayer, namespace) for layer in layers for sublayer in layer.get_layers()
        ]

    def get_layers(self) -> list["Layer"]:
        return self._layers_with_namespace

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        return self[0].get_loss_definitions(count=count * self.num_blocks) if self._config.num_blocks > 0 else []


class PatternBlockSequence[ConfigType: PatternBlockSequenceConfig](BlockBase[ConfigType], torch.nn.ModuleList):
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
        super().__init__(
            config,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=lr_scale,
            peft=peft,
        )
        self.extend(
            layers := [
                self._config.blocks[name].get_layer(
                    distributed_config,
                    hidden_dim,
                    lr_scale=self._lr_scale,
                    peft=self._peft,
                )
                for name in self._config.expanded_pattern
            ]
        )
        # Wrap each set of blocks with identical config in a namespace
        # using the unique module name of the first such block.
        # Note: Pytorch won't redundantly register modules  because it doesn't look into lists.
        self._layers_with_namespace = [
            LayerWithNamespace(sublayer, layers[self._config.preprocessing_layers[name]].module_name)
            for name, layer in zip(self._config.expanded_pattern, layers)
            for sublayer in layer.get_layers()
        ]

    def get_layers(self) -> list["Layer"]:
        return self._layers_with_namespace

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        # TODO: Prevent name conflicts.
        return sum(
            (
                self[self._config.preprocessing_layers[name]].get_loss_definitions(count=count * count_)
                for name, count_ in collections.Counter(self.expanded_pattern).items()
            ),
            [],
        )
