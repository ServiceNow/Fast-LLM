import abc
import functools
import logging
import typing

import torch

from fast_llm.config import Config, Configurable
from fast_llm.core.distributed import set_generator
from fast_llm.engine.base_model.base_model import Layer, Module
from fast_llm.engine.config_utils.run import log_pipeline_parallel_main_rank
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.block.config import BlockConfig, BlockKwargs, MixerConfig
from fast_llm.logging import log_distributed_grad, log_distributed_tensor, log_memory_usage
from fast_llm.tensor import TensorMeta

logger = logging.getLogger(__name__)


class DebugLayer:
    # TODO: Move elsewhere?
    def __init__(self, name: str, debug_level: int = 0, debug_memory: bool = False):
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
                    else hidden_dims[dim] if dim in hidden_dims else TensorDim(dim, tensor.size(i))
                )
                for i, dim in enumerate(dims)
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
                    global_=global_,
                    log_fn=log_fn,
                    scale=scale,
                )


class BlockLayerBase[ConfigType: Config](Configurable[ConfigType], Module):
    """
    Base class for blocks, mixer and MLP modules.
    """

    def __init__(
        self,
        config: ConfigType,
        block_config: BlockConfig,
        distributed_config: DistributedConfig,
        # TODO: Review `hidden_dim` and `block_index`
        hidden_dim: TensorDim,
        block_index: int,
        name: str,
        lr_scale: float | None,
    ):
        super().__init__(config, distributed_config)
        self._block_config = block_config
        self._hidden_dim = hidden_dim
        self._block_index = block_index
        self._name = name
        self._sequence_parallel: bool = self._distributed_config.sequence_tensor_parallel
        self._debug = DebugLayer(
            self._name,
            self._block_config.debug_transformer,
            self._block_config.debug_transformer_memory,
        )
        self._lr_scale = lr_scale


class BlockLayer[ConfigType: Config](BlockLayerBase[ConfigType]):
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


class Block[ConfigType: BlockConfig](BlockLayerBase[ConfigType], Layer):
    """
    A transformer-like decoder base block with abstract mixer.
    """

    # TODO: Standardize to `mixer`
    _mixer_module_name: typing.ClassVar[str] = "mixer"

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        hidden_dim: TensorDim,
        block_index: int,
        name: str,
        lr_scale: float | None,
        return_input: bool = False,
    ):
        super().__init__(
            config,
            config,
            distributed_config,
            hidden_dim,
            block_index,
            name,
            lr_scale,
        )
        # For multi-token prediction, return a stack of shared_hidden and transformer_output.
        self._return_input: bool = return_input
        # Note, layer_lr_scale does not impact the norms
        # TODO: add a separate norm_lr_scale
        self.norm_1 = self._config.peft.apply_other(
            self._config.normalization.get_layer(self._hidden_dim, self._lr_scale)
        )
        self.norm_2 = self._config.peft.apply_other(
            self._config.normalization.get_layer(self._hidden_dim, self._lr_scale)
        )

        # Attribute should be mixer, but Attention uses a different name for backward compatibility. TODO: Fix.
        setattr(
            self,
            self._mixer_module_name,
            self._mixer_config.get_layer(
                self._config,
                self._distributed_config,
                self._hidden_dim,
                self._block_index,
                f"{self._name} mixer",
                self._lr_scale,
            ),
        )

        self.mlp = self._config.mlp.get_layer(
            self._config,
            self._distributed_config,
            self._hidden_dim,
            self._block_index,
            f"{self._name} MLP",
            self._lr_scale,
        )

    @property
    @abc.abstractmethod
    def _mixer_config(self) -> MixerConfig:
        pass

    def setup(self, distributed: Distributed) -> None:
        super().setup(distributed)
        getattr(self, self._mixer_module_name).setup(distributed)
        self.mlp.setup(distributed)

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
        generator = self._distributed.tp_generator if self._sequence_parallel else self._distributed.pp_generator
        if self._debug.enabled:
            self._debug(None, "begin", kwargs[BlockKwargs.hidden_dims], kwargs)
        fw_input = input_
        hidden_states = self.norm_1(input_)
        if self._debug.enabled:
            self._debug(hidden_states, "norm 1", kwargs[BlockKwargs.hidden_dims], kwargs)
        hidden_states, bias = getattr(self, self._mixer_module_name)(hidden_states, kwargs)
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
