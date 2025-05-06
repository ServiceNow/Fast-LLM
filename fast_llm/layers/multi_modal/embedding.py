import typing

import torch

from fast_llm.core.distributed import set_generator
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.language_model.config import LanguageModelBaseConfig, LanguageModelKwargs
from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.layers.vision_encoder.config import VisionEncoderKwargs
from fast_llm.layers.vision_encoder.preprocessing import get_num_patches
from fast_llm.tensor import TensorMeta


class MultiModalEmbedding(LanguageModelEmbedding):
    """
    Multi-modal embedding layer to combine embeddings from text, image and more modalities.
    """

    def __init__(
        self,
        config: LanguageModelBaseConfig,
        tensor_space: TensorSpace,
    ):
        super().__init__(config, tensor_space)

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict | None = None,
    ) -> torch.Tensor:
        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(
                kwargs[TransformerKwargs.hidden_dims],
                tensor_name="Embedding output",
                dtype=self._residual_dtype,
            )
        # image_embeddings = kwargs.pop(VisionEncoderKwargs.patch_embeddings)
        position_ids = kwargs.get(LanguageModelKwargs.position_ids)
        image_sizes = kwargs.get(VisionEncoderKwargs.image_sizes)
        image_positions = kwargs.get(VisionEncoderKwargs.image_positions)
        tokens = kwargs.get(LanguageModelKwargs.tokens)
        # get text embeddings
        embeddings = super()._forward(tokens, position_ids)
        image_idx = 0
        for sample_idx, (positions, sizes) in enumerate(zip(image_positions, image_sizes)):
            image_embedding_offset = 0
            for position, size in zip(positions, sizes):
                num_image_tokens = get_num_patches(*size, self._config.vision_encoder.patch_size)
                embeddings[sample_idx, position : position + num_image_tokens] = input_[
                    sample_idx, image_embedding_offset : image_embedding_offset + num_image_tokens
                ]
                image_embedding_offset += num_image_tokens
                image_idx += 1

        with set_generator(
            self._tensor_space.distributed.tp_generator
            if self._sequence_parallel
            else self._tensor_space.distributed.pp_generator
        ):
            embeddings = torch.dropout(embeddings, self._dropout_p, self.training)

        return embeddings.to(self._residual_dtype)
