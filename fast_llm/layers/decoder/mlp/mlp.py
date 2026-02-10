import dataclasses
import typing

import torch

from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.engine.config_utils.initialization import init_normal_
from fast_llm.engine.config_utils.tensor_dim import ConcatenatedTensorDim, TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.mlp import mlp_autograd, torch_mlp_activation, triton_mlp_activation_autograd
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.block import BlockWithBias
from fast_llm.layers.decoder.mlp.config import MLPConfig
from fast_llm.tensor import TensorMeta


class MLPBase[ConfigType: MLPConfig](BlockWithBias[ConfigType]):
    _config: ConfigType

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        # TODO: Review `hidden_dim` and `block_index`
        hidden_dim: TensorDim,
        output_dim: TensorDim | None = None,
        lr_scale: float | None,
        peft: PeftConfig | None,
        return_bias: bool = True,
    ):
        super().__init__(
            config,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=lr_scale,
            peft=peft,
            return_bias=return_bias,
        )
        self._output_dim = self._hidden_dim if output_dim is None else output_dim
        self._parallel_dim = self._distributed_config.get_distributed_dim(DistributedDimNames.tensor)
        intermediate_1_dim, self._intermediate_2_dim = self._get_intermediate_dims()

        self._activation_fn = (
            triton_mlp_activation_autograd if TritonConfig.enabled(torch.device("cuda")) else torch_mlp_activation
        )

        # So both layers' weights have shape (num_experts [* gate_up] * ffn, hidden_size)
        self.layer_1 = self._config.layer_1.get_layer(
            hidden_dim,
            intermediate_1_dim,
            default_weight_initialization=init_normal_(std=self._hidden_size**-0.5),
            default_add_bias=self._config.add_linear_biases,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.layer_2 = self._config.layer_2.get_layer(
            self._intermediate_2_dim,
            self._output_dim,
            default_weight_initialization=init_normal_(std=self._hidden_size**-0.5),
            default_add_bias=self._config.add_linear_biases,
            sequence_parallel=self._sequence_parallel,
            transposed_weight=True,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

    def _get_intermediate_dims(self):
        intermediate_2_dim = TensorDim("intermediate", self._config.intermediate_size, self._parallel_dim)
        if self._config.gated:
            TensorDim("gate_and_up", 2)
            intermediate_1_dim = ConcatenatedTensorDim("gate_and_up", (intermediate_2_dim, intermediate_2_dim))
        else:
            intermediate_1_dim = intermediate_2_dim
        return intermediate_1_dim, intermediate_2_dim

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        # TODO: Generalize?
        layer_1_config = (
            dataclasses.replace(config, forward=config.forward + config.backward)
            if config.hardware and self._config.recompute_level.recompute_layer_1
            else config
        )
        # Also adjust the dtype in case of full-precision residual
        layer_2_input = TensorMeta.from_dims(
            (input_.dims[0], self._intermediate_2_dim),
            tensor_name="intermediate_1",
            dtype=self._distributed_config.compute_dtype.torch,
        )

        # TODO: Add marginal compute? (ex. activation, gate + up)
        return sum(
            (
                self.layer_1.get_compute_usage(input_, layer_1_config),
                self.layer_2.get_compute_usage(layer_2_input, config),
            )
        )


class MLP[ConfigType: MLPConfig](MLPBase[ConfigType]):
    _config: MLPConfig

    def _forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(input_, TensorMeta):
            return (
                TensorMeta.from_dims(
                    input_.dims[:-1] + (self._output_dim,), tensor_name="MLP output", dtype=input_.dtype
                ),
                None,
            )
        out = mlp_autograd(
            input_,
            None,
            self.layer_1.weight,
            self.layer_1.bias,
            self.layer_2.weight,
            None if self._parallel_dim.group else self.layer_2.bias,
            gated=self._config.gated,
            activation_type=self._config.activation,
            group=self._parallel_dim.group,
            sequence_parallel=self._sequence_parallel,
            training=self.training,
            recompute_level=self._config.recompute_level,
            transposed_layer_2_weight=self.layer_2.transposed_weight,
        )
        bias = self.layer_2.bias if self._parallel_dim.group else None
        # Use None for dims when output_dim differs from hidden_dim (e.g., adapter projections)
        # to let _debug infer dims from actual tensor shape
        self._debug(
            out, None, (kwargs[BlockKwargs.batch_config].hidden_token_dim, self._hidden_dim), kwargs, bias=bias
        )
        return out, bias
