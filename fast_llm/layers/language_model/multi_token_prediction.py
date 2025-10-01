import torch

from fast_llm.engine.base_model.base_model import Layer, LayerWithNamespace
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.block.block import BlockBase
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.language_model.config import LanguageModelEmbeddingsConfig, MultiTokenPredictionConfig


class MultiTokenPrediction[ConfigType: MultiTokenPredictionConfig](BlockBase[ConfigType]):
    _config: ConfigType

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        embeddings_config: LanguageModelEmbeddingsConfig,
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
        self.blocks = torch.nn.ModuleList(
            [
                self._config.block.get_layer(
                    self._distributed_config,
                    self._hidden_dim,
                    lr_scale=self._lr_scale,
                    peft=self._peft,
                    # The last block only returns the model output.
                    # The previous blocks return a stack of shared_hidden and transformer_output.
                    return_input=index < self._config.prediction_heads - 1,
                )
                for index in range(self._config.prediction_heads)
            ]
        )
        self.heads = torch.nn.ModuleList(
            [
                self._config.head.get_layer(
                    distributed_config,
                    embeddings_config,
                    hidden_dim=hidden_dim,
                    lr_scale=lr_scale,
                    peft=peft,
                    prediction_distance=index,
                    prediction_heads=self._config.prediction_heads,
                    loss_coefficient=(
                        1.0
                        if self._config.prediction_loss_coefficient is None
                        else self._config.prediction_loss_coefficient[index]
                    ),
                )
                for index in range(self._config.prediction_heads)
            ]
        )

        # Wrap all blocks in a namespace using the unique module name of the first one.
        namespace = self.blocks[0].module_name
        # Note: Pytorch won't redundantly register modules  because it doesn't look into lists.
        self._blocks_with_namespace = [
            LayerWithNamespace(sublayer, namespace) for layer in self.blocks for sublayer in layer.get_layers()
        ]

    def get_layers(self) -> list[Layer]:
        return [
            module
            for block, head in zip(self._blocks_with_namespace, self.heads, strict=True)
            for module in (block, head)
        ]
