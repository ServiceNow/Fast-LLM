import abc
import typing

import torch

from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.logging import log_distributed_grad, log_distributed_tensor
from fast_llm.tensor import TensorMeta
from fast_llm.utils import Assert


class Mixer(torch.nn.Module, abc.ABC):
    """
    Base class for mixer modules.
    """

    _mixer_name: typing.ClassVar[str]

    def __init__(self, tensor_space: TensorSpace, block_index: int, debug_level: int = 0):
        super().__init__()
        self._tensor_space = tensor_space
        self._sequence_parallel = self._tensor_space.distributed_config.sequence_tensor_parallel
        self._block_index = block_index
        self._debug_level = debug_level

    @abc.abstractmethod
    def forward(self, input_: torch.Tensor, kwargs: dict[str, typing.Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Mixer module forward. Returns the output hidden states and an optional bias,
         in case its addition can be made more efficient in `_bias_dropout_add`.
        """

    def _get_meta(
        self, input_: torch.Tensor, name: str, dim_names: tuple[str, ...], kwargs: dict[str, typing.Any]
    ) -> TensorMeta:
        hidden_dims = {
            dim.name: dim for dim in kwargs[BlockKwargs.hidden_dims] + (kwargs[BlockKwargs.sequence_q_dim],)
        }
        return TensorMeta.from_dims(
            tuple(
                hidden_dims[dim_name] if dim_name in hidden_dims else self._tensor_space[dim_name]
                for dim_name in dim_names
            ),
            tensor_name=f"Block {self._block_index} {self._mixer_name} {name}",
            dtype=input_.dtype,
        )

    def _debug_log(
        self, tensor: torch.Tensor, name: str, dim_names: tuple[str, ...], kwargs: dict[str, typing.Any]
    ) -> None:
        # TODO: Local vs global
        Assert.gt(self._debug_level, 0)
        log_distributed_tensor(
            "",
            tensor,
            level=self._debug_level,
            meta=self._get_meta(tensor, name, dim_names, kwargs),
            distributed=self._tensor_space.distributed,
        )
        if tensor.requires_grad:
            log_distributed_grad(
                "",
                tensor,
                level=self._debug_level,
                meta=self._get_meta(tensor, name + " grad", dim_names, kwargs),
                distributed=self._tensor_space.distributed,
            )
