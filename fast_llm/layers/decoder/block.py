import abc
import logging
import typing

import torch

from fast_llm.config import Config
from fast_llm.core.distributed import set_generator
from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.block.block import BaseBlock, Block
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.tensor import TensorMeta

logger = logging.getLogger(__name__)


class BlockWithBias[ConfigType: Config](BaseBlock[ConfigType]):
    """
    Base class for mixer and MLP modules.
    """

    @abc.abstractmethod
    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pass


class DecoderBlock[ConfigType: DecoderBlockConfig](Block[ConfigType]):
    """
    A transformer-like decoder base block with abstract mixer.
    """

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
        return_input: bool = False,
    ):
        super().__init__(
            config,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=lr_scale,
            peft=peft,
        )
        # For multi-token prediction, return a stack of shared_hidden and transformer_output.
        self._return_input: bool = return_input
        # Note, layer_lr_scale does not impact the norms
        # TODO: add a separate norm_lr_scale
        self.norm_1 = self._config.normalization.get_layer(self._hidden_dim, lr_scale=self._lr_scale, peft=self._peft)
        self.norm_2 = self._config.normalization.get_layer(self._hidden_dim, lr_scale=self._lr_scale, peft=self._peft)

        # Attribute should be mixer, but Attention uses a different name for backward compatibility. TODO: Fix.
        self.mixer = self._config.mixer.get_layer(
            self._distributed_config,
            self._hidden_dim,
            self._lr_scale,
            peft=peft,
        )

        self.mlp = self._config.mlp.get_layer(
            self._distributed_config,
            self._hidden_dim,
            self._lr_scale,
            peft=peft,
        )

    def setup(self, distributed: Distributed) -> None:
        super().setup(distributed)
        self.mixer.setup(distributed)
        self.mlp.setup(distributed)

    @torch.compile
    def _bias_dropout_add(
        self, input_: torch.Tensor, bias: torch.Tensor | None, residual: torch.Tensor
    ) -> torch.Tensor:
        if bias is not None:
            input_ = input_ + bias
        return residual + torch.dropout(input_, self._config.dropout, self.training)

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> torch.Tensor:
        if isinstance(input_, TensorMeta):
            dims = kwargs[BlockKwargs.hidden_dims]
            if self._return_input:
                dims = (TensorDim("stacked_input_output", 2),) + dims
            return TensorMeta.from_dims(dims, tensor_name=f"{self.module_name} output", dtype=input_.dtype)
        generator = self._distributed.tp_generator if self._sequence_parallel else self._distributed.pp_generator
        if self._debug.enabled:
            self._debug(None, "begin", kwargs[BlockKwargs.hidden_dims], kwargs)
        fw_input = input_
        hidden_states = self.norm_1(input_)
        if self._debug.enabled:
            self._debug(hidden_states, "norm 1", kwargs[BlockKwargs.hidden_dims], kwargs)
        hidden_states, bias = self.mixer(hidden_states, kwargs)
        if self._debug.enabled:
            self._debug(
                hidden_states if bias is None else hidden_states + bias,
                "mixer output",
                kwargs[BlockKwargs.hidden_dims],
                kwargs,
            )
        with set_generator(generator):
            input_ = self._bias_dropout_add(hidden_states, bias, input_)
        if self._debug.enabled:
            self._debug(input_, "mixer residual", kwargs[BlockKwargs.hidden_dims], kwargs)
        hidden_states = self.norm_2(input_)
        if self._debug.enabled:
            self._debug(hidden_states, "norm 2", kwargs[BlockKwargs.hidden_dims], kwargs)
        hidden_states, bias = self.mlp(hidden_states, kwargs, losses, metrics)
        if self._debug.enabled:
            self._debug(
                hidden_states if bias is None else hidden_states + bias,
                "MLP output",
                kwargs[BlockKwargs.hidden_dims],
                kwargs,
            )
        with set_generator(generator):
            hidden_states = self._bias_dropout_add(hidden_states, bias, input_)
        if self._debug.enabled:
            self._debug(None, "MLP residual", kwargs[BlockKwargs.hidden_dims], kwargs)
        if self._return_input:
            hidden_states = torch.stack((fw_input, hidden_states), dim=0)
        return hidden_states

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        # TODO: Add marginal compute? (normalization, bias_dropout_add)
        return sum(
            (
                self.mixer.get_compute_usage(input_, kwargs, config),
                self.mlp.get_compute_usage(input_, kwargs, config),
            )
        )
