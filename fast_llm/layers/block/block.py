import functools
import logging
import typing

import torch

from fast_llm.config import Config, Configurable
from fast_llm.engine.base_model.base_model import Layer, Module
from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.engine.config_utils.run import log_pipeline_parallel_main_rank
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.logging import get_model_debug_level, log_distributed_grad, log_distributed_tensor, log_memory_usage
from fast_llm.tensor import TensorMeta

logger = logging.getLogger(__name__)


class DebugLayer:
    """
    A debugging utility for blocks.
    """

    # TODO: Move elsewhere?
    def __init__(self, module: torch.nn.Module):
        self._module = module

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
    def _name(self):
        # Should be called after `module_name` is set in `BaseModel`
        return getattr(self._module, "module_name", "unknown")

    @property
    def enabled(self) -> bool:
        return get_model_debug_level() > 0

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
        if (level := get_model_debug_level()) == 0:
            return
        if level > 1:
            log_pipeline_parallel_main_rank(lambda: log_memory_usage(f"{self._name} {name}", str))
        if tensor is not None:
            log_distributed_tensor(
                "",
                tensor,
                level=level,
                meta=self._get_meta(tensor, name, dims, kwargs),
                global_=global_,
                log_fn=log_fn,
                scale=scale,
            )
            if tensor.requires_grad:
                log_distributed_grad(
                    "",
                    tensor,
                    level=level,
                    meta=self._get_meta(tensor, name + " grad", dims, kwargs),
                    global_=global_,
                    log_fn=log_fn,
                    scale=scale,
                )


class BaseBlock[ConfigType: Config](Configurable[ConfigType], Module):
    """
    Base class for blocks and block-like layers (mlp, mixers, etc.).
    """

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ):
        super().__init__(config, distributed_config)
        self._hidden_dim = hidden_dim
        self._hidden_size = self._hidden_dim.global_size
        self._sequence_parallel: bool = self._distributed_config.sequence_tensor_parallel
        self._debug = DebugLayer(self)
        self._lr_scale = lr_scale
        self._peft = peft

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        raise NotImplementedError()


class Block[ConfigType: Config](BaseBlock[ConfigType], Layer):
    """
    Base class for actual blocks, i.e., base blocks that are also `Layers`.
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
        super().__init__(config, distributed_config, hidden_dim=hidden_dim, lr_scale=lr_scale, peft=peft)
        self._return_input = return_input
