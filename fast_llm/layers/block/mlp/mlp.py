import typing

import torch

from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.mlp import mlp_autograd, torch_mlp_activation, triton_mlp_activation_autograd
from fast_llm.layers.block.block import BlockLayer
from fast_llm.layers.block.config import BlockDimNames
from fast_llm.layers.block.mlp.config import MLPConfig, MLPDimNames
from fast_llm.layers.block.peft import TransformerSubLayerName
from fast_llm.layers.common.linear import LinearBase
from fast_llm.utils import get_lr_scale


class MLPBase[ConfigType: MLPConfig](BlockLayer[ConfigType]):
    def __init__(self, config: ConfigType, tensor_space: TensorSpace, block_index: int, name: str):
        super().__init__(config, tensor_space, block_index, name)

        hidden_dim = self._tensor_space[BlockDimNames.hidden]
        self._intermediate_dim = self._tensor_space[MLPDimNames.composite_expert_mlp]
        self._activation_fn = triton_mlp_activation_autograd if TritonConfig.TRITON_ENABLED else torch_mlp_activation

        layer_lr_scale = (
            self._config.block.block_sequence.per_layer_lr_scale[self._block_index]
            if self._config.block.block_sequence.per_layer_lr_scale
            else None
        )
        lr_scale = (
            tuple(self._config.mlp_lr_scale)
            if isinstance(self._config.mlp_lr_scale, list)
            else self._config.mlp_lr_scale
        )
        lr_scale = get_lr_scale(lr_scale, layer_lr_scale)

        # So both layers' weights have shape (num_experts [* gate_up] * ffn, hidden_size)
        self.layer_1 = LinearBase(
            hidden_dim,
            self._tensor_space[MLPDimNames.composite_gated_expert_mlp],
            bias=self._config.add_bias,
            weight_init_method=self._config.layer_1_weight_initialization_method,
            bias_init_method=self._config.layer_1_bias_initialization_method,
            lr_scale=lr_scale,
        )
        self.layer_2 = LinearBase(
            self._intermediate_dim,
            hidden_dim,
            bias=self._config.add_bias,
            weight_init_method=self._config.layer_2_weight_initialization_method,
            bias_init_method=self._config.layer_2_bias_initialization_method,
            auto_bias_grad_accumulation=self._tensor_space.distributed_config.tensor_parallel > 1,
            transposed_weight=True,
            lr_scale=lr_scale,
        )

        # PEFT.
        self.layer_1 = self._config.block.peft.apply_linear(self.layer_1, TransformerSubLayerName.mlp_1)
        self.layer_2 = self._config.block.peft.apply_linear(self.layer_2, TransformerSubLayerName.mlp_2)


class MLP[ConfigType: MLPConfig](MLPBase[ConfigType]):
    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        parallel_group = self._intermediate_dim.parallel_group
        return (
            mlp_autograd(
                input_,
                None,
                self.layer_1.weight,
                self.layer_1.bias,
                self.layer_2.weight,
                None if parallel_group else self.layer_2.bias,
                gated=self._config.gated,
                activation_type=self._config.activation_type,
                group=parallel_group,
                sequence_parallel=self._sequence_parallel,
                training=self.training,
                recompute_level=self._config.mlp_recompute_level,
                transposed_layer_2_weight=self.layer_2.transposed_weight,
            ),
            self.layer_2.bias if parallel_group else None,
        )
