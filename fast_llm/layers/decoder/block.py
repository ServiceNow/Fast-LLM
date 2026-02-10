import abc
import logging
import typing

import torch

from fast_llm.core.distributed import ReduceOp, all_reduce, set_generator
from fast_llm.engine.base_model.config import LossDef, ResourceUsageConfig
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.functional.autograd import AuxiliaryLoss
from fast_llm.layers.block.block import Block
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.config import BlockWithBiasConfig, DecoderBlockConfig
from fast_llm.tensor import TensorMeta

logger = logging.getLogger(__name__)


class BlockWithBias[ConfigType: BlockWithBiasConfig](Block[ConfigType]):
    """
    Base class for mixer and MLP modules.
    """

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
        return_bias: bool = True,
    ):
        super().__init__(config, distributed_config, hidden_dim=hidden_dim, lr_scale=lr_scale, peft=peft)
        self._return_bias = return_bias

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor:
        output, bias = self._forward(input_, kwargs, losses, metrics)
        if self._return_bias:
            return output, bias
        else:
            return output if bias is None else output + bias

    @abc.abstractmethod
    def _forward(
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
        self._return_input = return_input
        self.norm_1 = self._config.normalization.get_layer(self._hidden_dim, lr_scale=self._lr_scale, peft=self._peft)
        self.norm_2 = self._config.normalization.get_layer(self._hidden_dim, lr_scale=self._lr_scale, peft=self._peft)

        self.mixer = self._config.mixer.get_layer(
            self._distributed_config,
            self._hidden_dim,
            lr_scale=self._lr_scale,
            peft=peft,
            return_bias=True,
        )

        self.mlp = self._config.mlp.get_layer(
            self._distributed_config,
            self._hidden_dim,
            lr_scale=self._lr_scale,
            peft=peft,
            return_bias=True,
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
            dims = input_.dims
            if self._return_input:
                dims = (TensorDim("stacked_input_output", 2),) + dims
            return TensorMeta.from_dims(dims, tensor_name=f"{self.module_name} output", dtype=input_.dtype)
        generator = self._distributed.tp_generator if self._sequence_parallel else self._distributed.pp_generator
        hidden_dims = (kwargs[BlockKwargs.batch_config].hidden_token_dim, self._hidden_dim)
        self._debug(None, "begin", hidden_dims, kwargs)
        fw_input = input_
        hidden_states = self.norm_1(input_)
        self._debug(hidden_states, "norm_1", hidden_dims, kwargs)
        hidden_states, bias = self.mixer(hidden_states, kwargs, metrics=metrics)

        if self._config.distillation_model is not None and self.training:
            if bias is not None:
                hidden_states = hidden_states + bias
                bias = None
            self._debug(hidden_states.detach(), "mixer_output", hidden_dims, kwargs)
            hidden_states = self._activation_distillation_loss(hidden_states, kwargs, losses, metrics)

        with set_generator(generator):
            input_ = self._bias_dropout_add(hidden_states, bias, input_)
        self._debug(input_, "mixer_residual", hidden_dims, kwargs)
        hidden_states = self.norm_2(input_)
        self._debug(hidden_states, "norm_2", hidden_dims, kwargs)
        hidden_states, bias = self.mlp(hidden_states, kwargs, losses, metrics)
        with set_generator(generator):
            hidden_states = self._bias_dropout_add(hidden_states, bias, input_)
        self._debug(hidden_states, None, hidden_dims, kwargs)

        if self._return_input:
            hidden_states = torch.stack((fw_input, hidden_states), dim=0)
        return hidden_states

    def _activation_distillation_loss(self, hidden_states, kwargs, losses, metrics):
        teacher_hidden_states = kwargs[f"reference_{self._config.distillation_model}_hidden_states"].pop(
            f"{self.module_name}.mixer_output"
        )

        # L2 loss
        per_token_loss = torch.norm(hidden_states - teacher_hidden_states, dim=-1, dtype=torch.float32)
        if (activation_mask := kwargs.get(BlockKwargs.activation_mask)) is not None:
            per_token_loss = per_token_loss * activation_mask
        loss = torch.mean(per_token_loss)

        # All-reduce across tensor-parallel group if sequence-parallel is enabled
        if self._sequence_parallel and self._distributed.tensor_group is not None:
            all_reduce(loss, group=self._distributed.tensor_group, op=ReduceOp.AVG)

        scaled_activation_loss = self._config.distillation_loss_weight * loss

        # Backward hook
        hidden_states = AuxiliaryLoss.apply(hidden_states, scaled_activation_loss, kwargs.get(BlockKwargs.grad_output))

        # Logging
        if losses is not None and self._distillation_loss_name in losses:
            losses[self._distillation_loss_name].append(loss.detach())

        if metrics is not None:
            metrics[f"{self.module_name}/activation_distillation_loss"] = loss.detach()

            # If using stochastic mixer, also log per-mixer-type activation distillation loss
            from fast_llm.layers.decoder.stochastic_mixer import StochasticMixer

            if isinstance(self.mixer, StochasticMixer):
                metrics[f"{self.module_name}/activation_distillation_loss/{self.mixer._last_selected_mixer}"] = (
                    loss.detach()
                )
        return hidden_states

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        # TODO: Add marginal compute? (normalization, bias_dropout_add)
        return sum(
            (
                self.mixer.get_compute_usage(input_, kwargs, config),
                self.mlp.get_compute_usage(input_, kwargs, config),
            )
        )

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        self.mixer.preprocess(kwargs)
        self.mlp.preprocess(kwargs)

    # TODO: add layer_index
    _distillation_loss_name = "activation_distillation_loss"

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        loss_definitions = []
        if self._config.distillation_model is not None:
            loss_definitions.append(
                LossDef(
                    name=self._distillation_loss_name,
                    formatted_name=self._distillation_loss_name,
                    count=count,
                )
            )
        return (
            loss_definitions
            + self.mixer.get_loss_definitions(count=count)
            + self.mlp.get_loss_definitions(count=count)
        )
