import logging
import typing

import torch

from fast_llm.core.distributed import all_gather_scalar
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedDim
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.layers.vision.vision_encoder import VisionMultiModalModel
from fast_llm.models.gpt.model import GPTBaseModel, GPTModel
from fast_llm.models.multimodal.config import MultiModalBaseModelConfig, MultiModalModelConfig

logger = logging.getLogger(__name__)


class PatchSequenceTensorDim(TensorDim):
    """
    A custom `TensorDim` class to handle the combined batch/sequence dimension in image patches.

    A simple gather `TensorDim.local_to_global` yields inconsistent results between distributed configuration,
    (because of the padding of image patches) which makes direct comparison in tests impossible.
    This class solves the problem removing the padding in the tensor returned by `local_to_global`,
    allowing for consistent results.
    Note that `local_unpadded_size` must be set manually before any call to `local_to_global`.
    """

    local_unpadded_size: typing.ClassVar[int]

    def __init__(self, name: str, global_size: int, parallel_dim: DistributedDim, batch_parallel_dim: DistributedDim):
        super().__init__(name, global_size * batch_parallel_dim.size, parallel_dim, variable_size=True)
        self._batch_parallel_dim = batch_parallel_dim

    @property
    def is_parallel(self) -> bool:
        # Ensure `local_to_global` is called in non-parallel setting.
        return True

    def replace_parallel_dim(self, distributed_dim: DistributedDim) -> typing.Self:
        raise NotImplementedError()

    def local_to_global(self, tensor: "torch.Tensor", dim: int = 0) -> "torch.Tensor":
        assert hasattr(self, "local_unpadded_size")
        batch_parallel_group = self._batch_parallel_dim.group
        global_padded_tensor = super().local_to_global(tensor, dim)

        if batch_parallel_group is None:
            return global_padded_tensor[*(slice(None) for _ in range(dim)), : self.local_unpadded_size]
        else:
            unpadded_sequence_lengths = all_gather_scalar(self.local_unpadded_size, torch.int32, batch_parallel_group)
            return torch.cat(
                [
                    tensor[*(slice(None) for _ in range(dim)), :unpadded_sequence_length]
                    for tensor, unpadded_sequence_length in zip(
                        global_padded_tensor.chunk(batch_parallel_group.size(), dim=dim),
                        unpadded_sequence_lengths,
                        strict=True,
                    )
                ],
                dim=dim,
            )

    def local_to_global_partial(
        self, tensor: "torch.Tensor", dim: int = 0, fill_value: float | int = -1
    ) -> "torch.Tensor":
        # Not needed.
        raise NotImplementedError()

    def global_to_local(self, tensor: "torch.Tensor", dim: int = 0, expand: bool = False) -> "torch.Tensor":
        # Not needed.
        raise NotImplementedError()


class MultiModalBaseModel[ConfigType: MultiModalBaseModelConfig](
    GPTBaseModel[ConfigType], VisionMultiModalModel[ConfigType]
):
    """
    A transformer-based language model generalizing the GPT model architecture.
    """

    _config: ConfigType


class MultiModalModel[ConfigType: MultiModalModelConfig](GPTModel[ConfigType]):
    # TODO: Can we drop class?
    pass


class MultiModalInferenceRunner(InferenceRunner):
    model_class: typing.ClassVar[type[MultiModalModel]] = MultiModalModel
