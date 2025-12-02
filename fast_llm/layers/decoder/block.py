import abc
import logging
import typing

import torch

from fast_llm.core.distributed import ReduceOp, all_reduce, set_generator
from fast_llm.engine.base_model.config import LossDef, ResourceUsageConfig
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.block.block import Block
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.common.auxiliary_loss import AuxiliaryLoss
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.config import BlockWithBiasConfig, DecoderBlockConfig
from fast_llm.layers.language_model.head import _format_name
from fast_llm.tensor import TensorMeta
from fast_llm.utils import Assert

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
            dims = kwargs[BlockKwargs.hidden_dims]
            if self._return_input:
                dims = (TensorDim("stacked_input_output", 2),) + dims
            return TensorMeta.from_dims(dims, tensor_name=f"{self.module_name} output", dtype=input_.dtype)
        generator = self._distributed.tp_generator if self._sequence_parallel else self._distributed.pp_generator
        self._debug(None, "begin", kwargs.get(BlockKwargs.hidden_dims), kwargs)
        fw_input = input_
        hidden_states = self.norm_1(input_)
        self._debug(hidden_states, "norm_1", kwargs.get(BlockKwargs.hidden_dims), kwargs)
        hidden_states, bias = self.mixer(hidden_states, kwargs)

        hidden_states, bias = self.activation_distillation_loss(hidden_states, bias, kwargs, losses)

        with set_generator(generator):
            input_ = self._bias_dropout_add(hidden_states, bias, input_)
        self._debug(input_, "mixer_residual", kwargs.get(BlockKwargs.hidden_dims), kwargs)
        hidden_states = self.norm_2(input_)
        self._debug(hidden_states, "norm_2", kwargs.get(BlockKwargs.hidden_dims), kwargs)
        hidden_states, bias = self.mlp(hidden_states, kwargs, losses, metrics)
        with set_generator(generator):
            hidden_states = self._bias_dropout_add(hidden_states, bias, input_)
        self._debug(hidden_states, None, kwargs.get(BlockKwargs.hidden_dims), kwargs)

        if self._return_input:
            hidden_states = torch.stack((fw_input, hidden_states), dim=0)
        return hidden_states

    def activation_distillation_loss(self, hidden_states, bias, kwargs, losses):
        """
        Maybe apply activation distillation loss and setup backward hooks
        """
        mixer_output = hidden_states if bias is None else hidden_states + bias
        # Teacher populates mixer activations for distillation.
        activation_storage = kwargs.get(BlockKwargs.activation_distillation_storage)
        if activation_storage is not None:
            activation_storage[self.module_name] = mixer_output.detach()
        # Student gets teacher activations and computes the activation-level loss.
        activation_targets = kwargs.get(BlockKwargs.activation_distillation_targets)
        if (
            activation_targets is not None
            and self.training
            and (teacher_output := activation_targets.pop(self.module_name, None)) is not None
        ):
            # Compare student mixer output with the teacher's stored activation and accumulate the loss.
            teacher_tensor = teacher_output.detach().to(device=mixer_output.device, dtype=mixer_output.dtype)
            Assert.eq(teacher_tensor.shape, mixer_output.shape)
            # TODO: un-scaled loss for reporting? Average loss over layers?
            # L2 loss
            activation_loss_factor = self._config.activation_distillation_factor
            # (batch, sequence, hidden) or (sequence, batch, hidden). Take the norm over hidden dim.
            # TODO: handle possible padding?
            local_loss_sum = torch.sum(torch.norm(mixer_output - teacher_tensor, p=2, dim=(2)))
            # mixer_output.shape is (batch, sequence, hidden) or (sequence, batch, hidden)
            # In either case, dims 0 and 1 are batch and sequence
            total_count = mixer_output.shape[0] * mixer_output.shape[1]

            # All-reduce across tensor-parallel group if sequence-parallel is enabled
            if self._sequence_parallel and self._distributed.tensor_group is not None:
                all_reduce(local_loss_sum, group=self._distributed.tensor_group, op=ReduceOp.SUM)
                # Assume all ranks contribute the same count (not the case if padding)
                total_count *= self._distributed.tensor_group.size()

            activation_loss = activation_loss_factor * (local_loss_sum / total_count)

            # Backward hooks
            hidden_states = AuxiliaryLoss.apply(hidden_states, activation_loss, 1.0)
            bias = AuxiliaryLoss.apply(bias, activation_loss, 1.0) if bias is not None else None
            # Logging
            if losses is not None and self._activation_distillation_loss_name in losses:
                losses[self._activation_distillation_loss_name].append(activation_loss.detach())
        return hidden_states, bias

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
    _activation_distillation_loss_name = "activation_distillation_loss"

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        loss_definitions = []
        if self._config.activation_distillation_factor > 0.0 and self._config.distillation_model is not None:
            loss_definitions.append(
                LossDef(
                    name=self._activation_distillation_loss_name,
                    formatted_name=_format_name(self._activation_distillation_loss_name),
                    count=count,
                )
            )
        return (
            loss_definitions
            + self.mixer.get_loss_definitions(count=count)
            + self.mlp.get_loss_definitions(count=count)
        )
