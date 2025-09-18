import typing

import torch

from fast_llm.engine.config_utils.initialization import init_normal_, init_zeros_
from fast_llm.engine.config_utils.tensor_dim import ConcatenatedTensorDim, TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.mlp import mlp_autograd, torch_mlp_activation, triton_mlp_activation_autograd
from fast_llm.layers.block.block import BlockLayer
from fast_llm.layers.block.config import BlockConfig
from fast_llm.layers.block.mlp.config import MLPConfig
from fast_llm.layers.block.peft import TransformerSubLayerName
from fast_llm.layers.common.linear import LinearBase
from fast_llm.utils import Assert, combine_lr_scales


class MLPBase[ConfigType: MLPConfig](BlockLayer[ConfigType]):
    def __init__(
        self,
        config: ConfigType,
        block_config: BlockConfig,
        distributed_config: DistributedConfig,
        hidden_dim: TensorDim,
        block_index: int,
        name: str,
        lr_scale: float | None,
    ):
        super().__init__(config, block_config, distributed_config, hidden_dim, block_index, name, lr_scale)
        self._parallel_dim = self._distributed_config.get_distributed_dim(DistributedDimNames.tensor)
        intermediate_1_dim, intermediate_2_dim = self._get_intermediate_dims()

        init_method_1 = init_normal_(
            std=self._config.init_method_std_mlp_1,
            min_val=self._config.init_method_min_mlp_1,
            max_val=self._config.init_method_max_mlp_1,
        )
        init_method_2 = init_normal_(
            std=self._config.init_method_std_mlp_2,
            min_val=self._config.init_method_min_mlp_2,
            max_val=self._config.init_method_max_mlp_2,
        )

        self._activation_fn = triton_mlp_activation_autograd if TritonConfig.TRITON_ENABLED else torch_mlp_activation

        lr_scale = combine_lr_scales(self._lr_scale, self._config.mlp_lr_scale)

        # So both layers' weights have shape (num_experts [* gate_up] * ffn, hidden_size)
        self.layer_1 = LinearBase(
            hidden_dim,
            intermediate_1_dim,
            bias=self._config.add_mlp_bias,
            weight_init_method=init_method_1,
            bias_init_method=init_zeros_,
            lr_scale=lr_scale,
        )
        self.layer_2 = LinearBase(
            intermediate_2_dim,
            hidden_dim,
            bias=self._config.add_mlp_bias,
            weight_init_method=init_method_2,
            bias_init_method=init_zeros_,
            auto_bias_grad_accumulation=self._distributed_config.tensor_parallel > 1,
            transposed_weight=True,
            lr_scale=lr_scale,
        )

        # PEFT.
        self.layer_1 = self._block_config.peft.apply_linear(self.layer_1, TransformerSubLayerName.mlp_1)
        self.layer_2 = self._block_config.peft.apply_linear(self.layer_2, TransformerSubLayerName.mlp_2)

    def _get_intermediate_dims(self):
        intermediate_2_dim = TensorDim("intermediate", self._config.ffn_hidden_size, self._parallel_dim)
        if self._config.gated:
            TensorDim("gate_and_up", 2)
            intermediate_1_dim = ConcatenatedTensorDim("gate_and_up", (intermediate_2_dim, intermediate_2_dim))
        else:
            intermediate_1_dim = intermediate_2_dim
        return intermediate_1_dim, intermediate_2_dim


class MLP[ConfigType: BlockConfig](MLPBase[ConfigType]):
    def __init__(
        self,
        config: ConfigType,
        block_config: BlockConfig,
        distributed_config: DistributedConfig,
        hidden_dim: TensorDim,
        block_index: int,
        name: str,
        lr_scale: float | None,
    ):
        Assert.eq(config.num_experts, 1)
        super().__init__(config, block_config, distributed_config, hidden_dim, block_index, name, lr_scale)

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return (
            mlp_autograd(
                input_,
                None,
                self.layer_1.weight,
                self.layer_1.bias,
                self.layer_2.weight,
                None if self._parallel_dim.group else self.layer_2.bias,
                gated=self._config.gated,
                activation_type=self._config.activation_type,
                group=self._parallel_dim.group,
                sequence_parallel=self._sequence_parallel,
                training=self.training,
                recompute_level=self._config.mlp_recompute_level,
                transposed_layer_2_weight=self.layer_2.transposed_weight,
            ),
            self.layer_2.bias if self._parallel_dim.group else None,
        )
