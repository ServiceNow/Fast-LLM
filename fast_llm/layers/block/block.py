import abc
import functools
import logging
import typing

import torch

from fast_llm.config import Configurable
from fast_llm.core.distributed import set_generator
from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.config_utils.run import log_pipeline_parallel_main_rank
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.layers.block.config import BlockConfig, BlockDimNames, BlockKwargs, BlockLayerConfig
from fast_llm.logging import log_distributed_grad, log_distributed_tensor, log_memory_usage
from fast_llm.tensor import TensorMeta

logger = logging.getLogger(__name__)


class DebugLayer:
    # TODO: Move elsewhere?
    def __init__(self, tensor_space: TensorSpace, name: str, debug_level: int = 0, debug_memory: bool = False):
        self._tensor_space = tensor_space
        self._name = name
        self._debug_level = debug_level
        self._debug_memory = debug_memory

    def _get_meta(
        self, tensor: torch.Tensor, name: str, dims: tuple[TensorDim | str, ...], kwargs: dict[str, typing.Any]
    ) -> TensorMeta:
        hidden_dims = {
            dim.name: dim for dim in kwargs[BlockKwargs.hidden_dims] + (kwargs[BlockKwargs.sequence_q_dim],)
        }
        return TensorMeta.from_dims(
            tuple(
                (
                    dim
                    if isinstance(dim, TensorDim)
                    else hidden_dims[dim] if dim in hidden_dims else self._tensor_space[dim]
                )
                for dim in dims
            ),
            tensor_name=f"{self._name} {name}",
            dtype=tensor.dtype,
        )

    @functools.cached_property
    def enabled(self) -> bool:
        return self._debug_level > 0 or self._debug_memory

    def __call__[
        T
    ](
        self,
        tensor: torch.Tensor | None,
        name: str,
        dims: tuple[TensorDim | str, ...],
        kwargs: dict[str, typing.Any],
        scale: float = 1.0,
        global_: bool = True,
        log_fn: type[BaseException] | typing.Callable[[str], T] | None = logger.info,
    ) -> None:
        # TODO: Local vs global?
        if self._debug_memory:
            log_pipeline_parallel_main_rank(lambda: log_memory_usage(f"{self._name} {name}", str))
        if self._debug_level > 0 and tensor is not None:
            log_distributed_tensor(
                "",
                tensor,
                level=self._debug_level,
                meta=self._get_meta(tensor, name, dims, kwargs),
                distributed=self._tensor_space.distributed,
                global_=global_,
                log_fn=log_fn,
                scale=scale,
            )
            if tensor.requires_grad:
                log_distributed_grad(
                    "",
                    tensor,
                    level=self._debug_level,
                    meta=self._get_meta(tensor, name + " grad", dims, kwargs),
                    distributed=self._tensor_space.distributed,
                    global_=global_,
                    log_fn=log_fn,
                    scale=scale,
                )


class BlockLayerBase[ConfigType: BaseModelConfig](Configurable[ConfigType], torch.nn.Module, abc.ABC):
    """
    Base class for blocks, mixer and MLP modules.
    """

    def __init__(
        self, config: ConfigType, tensor_space: TensorSpace, block_index: int, name: str, block_config: BlockConfig
    ):
        super().__init__(config)
        self._tensor_space = tensor_space
        self._block_index = block_index
        self._name = name
        self._sequence_parallel: bool = self._tensor_space.distributed_config.sequence_tensor_parallel
        self._debug = DebugLayer(
            tensor_space,
            self._name,
            block_config.debug_transformer,
            block_config.debug_transformer_memory,
        )

    # @property
    # def name(self) -> str:
    #   return self._name


class BlockLayer[ConfigType: BlockLayerConfig](BlockLayerBase[ConfigType], torch.nn.Module):
    """
    Base class for mixer and MLP modules.
    """

    def __init__(self, config: ConfigType, tensor_space: TensorSpace, block_index: int, name: str):
        super().__init__(config, tensor_space, block_index, name, config.block)

    @abc.abstractmethod
    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pass


class Block[ConfigType: BlockConfig](BlockLayerBase[ConfigType], Layer):
    """
    A transformer-like decoder base block with abstract mixer.
    """

    def __init__(
        self, config: ConfigType, tensor_space: TensorSpace, block_index: int, name: str, return_input: bool = False
    ):
        super().__init__(config, tensor_space, block_index, name, config)
        # For multi-token prediction, return a stack of shared_hidden and transformer_output.
        self._return_input: bool = return_input
        hidden_dim = self._tensor_space[BlockDimNames.hidden]
        # Note, layer_lr_scale does not impact the norms
        # TODO: add a separate norm_lr_scale
        self.norm_1 = self._config.peft.apply_other(self._config.normalization.get_layer(hidden_dim))
        self.norm_2 = self._config.peft.apply_other(self._config.normalization.get_layer(hidden_dim))

        # Attribute should be mixer, but Attention uses a different name for backward compatibility. TODO: Fix.
        setattr(
            self,
            self._config.mixer.module_name,
            self._config.mixer.get_layer(self._tensor_space, self._block_index, f"{self._name} mixer"),
        )
        self.mlp = self._config.mlp.get_layer(self._tensor_space, self._block_index, f"{self._name} mlp")

    @torch.compile
    def _bias_dropout_add(
        self, input_: torch.Tensor, bias: torch.Tensor | None, residual: torch.Tensor
    ) -> torch.Tensor:
        if bias is not None:
            input_ = input_ + bias
        return residual + torch.dropout(input_, self._config.hidden_dropout, self.training)

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
            return TensorMeta.from_dims(dims, tensor_name=f"{self._name} output", dtype=input_.dtype)
        generator = (
            self._tensor_space.distributed.tp_generator
            if self._tensor_space.distributed_config.sequence_tensor_parallel
            else self._tensor_space.distributed.pp_generator
        )
        if self._debug.enabled:
            self._debug(None, "begin", kwargs[BlockKwargs.hidden_dims], kwargs)
        fw_input = input_
        hidden_states = self.norm_1(input_)
        if self._debug.enabled:
            self._debug(hidden_states, "norm 1", kwargs[BlockKwargs.hidden_dims], kwargs)
        hidden_states, bias = getattr(self, self._config.mixer.module_name)(hidden_states, kwargs)
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
