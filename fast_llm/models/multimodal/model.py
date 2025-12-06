import logging
import typing

import torch

from fast_llm.core.distributed import all_gather_scalar
from fast_llm.data.sample.language_model import LanguageModelBatch
from fast_llm.data.sample.patch import PatchBatch
from fast_llm.engine.config_utils.tensor_dim import TensorDim, scalar_dim
from fast_llm.engine.distributed.config import DistributedDim, DistributedDimNames, PhaseType
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.block.config import BlockDimNames, BlockKwargs
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.vision.config import VisionKwargs
from fast_llm.layers.vision.vision_encoder import VisionMultiModalModel
from fast_llm.models.gpt.config import GPTBatchConfig
from fast_llm.models.gpt.model import GPTBaseModel, GPTModel
from fast_llm.models.multimodal.config import MultiModalBaseModelConfig, MultiModalBatchConfig, MultiModalModelConfig
from fast_llm.tensor import TensorMeta

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

    def preprocess_meta(
        self, batch_meta: GPTBatchConfig | torch.Tensor, phase: PhaseType
    ) -> list[tuple[TensorMeta, dict]]:
        preprocessed_meta = []
        for tokens, kwargs in super().preprocess_meta(batch_meta, phase):
            kwargs[LanguageModelKwargs.token_ids] = tokens
            kwargs[LanguageModelKwargs.mask_inputs] = True
            # TODO: What about sequence data?
            batch_data_dim = self._distributed_config.get_distributed_dim(DistributedDimNames.batch_data)

            micro_sequence_length = tokens.global_shape.numel()

            batch_and_sequence_q_dim = PatchSequenceTensorDim(
                BlockDimNames.sequence_q,
                micro_sequence_length,
                self._distributed_config.get_distributed_dim(DistributedDimNames.data),
                batch_data_dim,
            )
            hidden_batch_and_sequence_q_dim = (
                PatchSequenceTensorDim(
                    BlockDimNames.sequence_q_tp,
                    micro_sequence_length,
                    self._distributed_config.get_distributed_dim(DistributedDimNames.tensor_and_data),
                    batch_data_dim,
                )
                if self._distributed_config.sequence_tensor_parallel
                else batch_and_sequence_q_dim
            )
            # These are used by the model (preprocessing) and shouldn't see the batch-parallel dim.
            sequence_q_dim = TensorDim(
                BlockDimNames.sequence_q,
                micro_sequence_length,
                self._distributed_config.get_distributed_dim(DistributedDimNames.sequence_data),
            )
            sequence_k_dim = TensorDim(BlockDimNames.sequence_k, micro_sequence_length)

            image_patches = TensorMeta.from_dims(
                (
                    # We combine the batch and sequence dims to allow for variable sequence lengths.
                    # Gives the same result, assuming we disable cross-image attention (TODO: Enforce)
                    batch_and_sequence_q_dim,
                    # TODO: Relate to tensor dims in patch convolution.
                    TensorDim("input_channels", self._config.vision_encoder.embeddings.input_channels),
                    TensorDim("patch_height", self._config.vision_encoder.embeddings.patch_height),
                    TensorDim("patch_width", self._config.vision_encoder.embeddings.patch_width),
                )
            )
            hidden_dims = (
                (hidden_batch_and_sequence_q_dim, scalar_dim, self.vision_encoder._hidden_dim)
                if (sequence_first := kwargs[LanguageModelKwargs.sequence_first])
                else (scalar_dim, hidden_batch_and_sequence_q_dim, self.vision_encoder._hidden_dim)
            )
            kwargs[self._vision_encoder_namespace] = {
                VisionKwargs.sequence_first: sequence_first,
                VisionKwargs.sequence_k_dim: sequence_k_dim,
                VisionKwargs.sequence_q_dim: sequence_q_dim,
                VisionKwargs.hidden_dims: hidden_dims,
            }

            preprocessed_meta.append((image_patches, kwargs))

        return preprocessed_meta

    def _get_empty_image_patches(self, tokens: torch.Tensor, kwargs: dict[str, typing.Any]) -> PatchBatch:
        patch_embeddings_config = self._config.vision_encoder.embeddings
        sequence_first = kwargs[AttentionKwargs.sequence_first]
        device = tokens.device
        dtype = self._distributed.config.compute_dtype.torch
        return PatchBatch(
            patches=torch.empty(
                (
                    0,
                    patch_embeddings_config.input_channels,
                    patch_embeddings_config.patch_height,
                    patch_embeddings_config.patch_width,
                ),
                device=device,
                dtype=dtype,
            ),
            sample_map=torch.empty(0, device=device, dtype=torch.int32),
            token_map=torch.empty(0, device=device, dtype=torch.int32),
            positions=torch.empty((0, 2), device=device, dtype=torch.int32),
            num_samples=tokens.shape[1] if sequence_first else tokens.shape[0],
            sample_size=kwargs[AttentionKwargs.sequence_q_dim].size,
            lengths=[],
        )

    def preprocess_batch(
        self,
        batch: LanguageModelBatch,
        preprocessed_meta: list[tuple[TensorMeta, dict]] | None = None,
        *,
        phase: PhaseType,
        iteration: int,
        metrics: dict | None = None,
    ) -> list[tuple[torch.Tensor, dict]]:
        preprocessed = super().preprocess_batch(
            batch, preprocessed_meta, phase=phase, iteration=iteration, metrics=metrics
        )
        # TODO: Support micro-sequences.
        assert len(preprocessed) == 1, "Micro-sequences not supported for MultiModalModel."
        tokens, kwargs = preprocessed[0]

        kwargs[LanguageModelKwargs.token_ids] = tokens

        # If document cropping is enabled, extra tokens may belong to images and need to be removed.
        # TODO: Handle earlier.
        tokens_end = kwargs[AttentionKwargs.sequence_k_dim].size
        tokens_begin = tokens_end - kwargs[AttentionKwargs.sequence_q_dim].size
        if batch.image_patches is None:
            cropped_image_patches = self._get_empty_image_patches(tokens, kwargs)
        else:
            cropped_image_patches = batch.image_patches.crop(tokens_begin, tokens_end)

        sequence_length = tokens.shape[:2].numel()
        pad_size = sequence_length - cropped_image_patches.patches.size(0)

        patches = cropped_image_patches.patches.to(self._distributed.config.compute_dtype.torch)
        patches = torch.cat([patches, patches.new_zeros((pad_size,) + patches.shape[1:])])

        positions = torch.cat(
            [
                cropped_image_patches.positions,
                cropped_image_patches.positions.new_zeros((pad_size,) + cropped_image_patches.positions.shape[1:]),
            ]
        )

        kwargs[self._vision_encoder_namespace] = {
            **kwargs[self._vision_encoder_namespace],
            VisionKwargs.patch_positions: positions,
            VisionKwargs.sequence_lengths: [cropped_image_patches.lengths + [pad_size]],
            VisionKwargs.sequence_length: sequence_length,
            VisionKwargs.device: self._distributed.device,
            BlockKwargs.output_hidden_states: kwargs.get(BlockKwargs.output_hidden_states, []),
            BlockKwargs.hidden_states: kwargs[BlockKwargs.hidden_states],
        }
        # We need to modify `local_unpadded_size` directly in `preprocessed_meta` since it's the one used by the engine.
        # Unsafe, but only needed for testing.
        # TODO: Doesn't work with gradient accumulation (only sees the last value).
        hidden_batch_and_sequence_q_dim = kwargs[self._vision_encoder_namespace][VisionKwargs.hidden_dims][
            0 if kwargs[self._vision_encoder_namespace][VisionKwargs.sequence_first] else 1
        ]
        assert isinstance(hidden_batch_and_sequence_q_dim, PatchSequenceTensorDim)
        PatchSequenceTensorDim.local_unpadded_size = cropped_image_patches.patches.size(0)

        kwargs[LanguageModelKwargs.embedding_map] = (
            (cropped_image_patches.token_map, cropped_image_patches.sample_map)
            if kwargs[LanguageModelKwargs.sequence_first]
            else (cropped_image_patches.sample_map, cropped_image_patches.token_map)
        )

        super().preprocess(kwargs)

        return [(patches, kwargs)]

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        # Hack to delay preprocessing in super().preprocess_batch (TODO: Improve)
        pass


class MultiModalModel[ConfigType: MultiModalModelConfig](GPTModel[ConfigType]):
    # TODO: Can we drop class?
    pass


class MultiModalInferenceRunner(InferenceRunner):
    model_class: typing.ClassVar[type[MultiModalModel]] = MultiModalModel
    batch_config_class: typing.ClassVar[type[MultiModalBatchConfig]] = MultiModalBatchConfig
