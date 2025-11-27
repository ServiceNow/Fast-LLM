import functools
import logging
import typing

import torch

from fast_llm.config import Configurable
from fast_llm.engine.base_model.base_model import Layer, LayerBase
from fast_llm.engine.base_model.config import ModuleConfig
from fast_llm.engine.config_utils.run import log_pipeline_parallel_main_rank
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.block.config import BlockConfig, BlockKwargs
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

    def __call__(
        self,
        tensor: torch.Tensor | None,
        suffix: str | None,
        dims: tuple[TensorDim | str, ...],
        kwargs: dict[str, typing.Any],
        bias: torch.Tensor | None = None,
        **logging_kwargs,
    ):
        name = self._name if suffix is None else f"{self._name}.{suffix}"
        output_hidden_state = (
            BlockKwargs.output_hidden_states in kwargs
            and any(pattern.match(name) for pattern in kwargs[BlockKwargs.output_hidden_states])
            and tensor is not None
        )
        if (level := get_model_debug_level()) == 0 and not output_hidden_state:
            return
        if bias is not None:
            assert tensor is not None
            tensor = tensor + bias
        meta = self._get_meta(tensor, name, dims, kwargs)

        if output_hidden_state:
            kwargs[BlockKwargs.hidden_states][name] = (meta, tensor)

        if level > 1:
            log_pipeline_parallel_main_rank(lambda: log_memory_usage(name, str))

        if level > 0 and tensor is not None:
            log_distributed_tensor(
                "",
                tensor,
                level=level,
                meta=meta,
                **logging_kwargs,
            )
            if tensor.requires_grad:
                log_distributed_grad(
                    "",
                    tensor,
                    level=level,
                    meta=self._get_meta(tensor, name + f"{name}.grad", dims, kwargs),
                    **logging_kwargs,
                )

    def _get_meta(
        self, tensor: torch.Tensor | None, name: str, dims: tuple[TensorDim | str, ...], kwargs: dict[str, typing.Any]
    ) -> TensorMeta | None:
        if tensor is None:
            return None
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
            tensor_name=name,
            dtype=tensor.dtype,
        )

    @functools.cached_property
    def _name(self):
        # Should be called after `module_name` is set in `BaseModel`
        return getattr(self._module, "module_name", "unknown")


class BlockBase[ConfigType: ModuleConfig](Configurable[ConfigType], LayerBase):
    """
    Base class for blocks and block-like layers (mlp, mixers, block sequences, etc.).
    """

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        # TODO: Review. Use `input_dim(s)` and `output_dim(s)` instead?
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


class Block[ConfigType: BlockConfig](BlockBase[ConfigType], Layer):
    pass
